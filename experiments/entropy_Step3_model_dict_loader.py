# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-model.py
@time:2021/9/15 16:33
"""

import os
import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
import copy
import time
import pickle
import math


import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import dropout, nn
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
import random


from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE
from model_helper import Encoder_MultipleLayers, Embeddings
from Step2_DataEncoding import DataEncoding
from torchmetrics import R2Score
import sklearn
from torch.nn.functional import softmax
from torch.profiler import profile, record_function, ProfilerActivity
from tensordict import TensorDict
from ..loss import MOORLELoss

device = torch.device('cuda:0')
num_workers = 0
dtype = torch.float


def score_preds(y_true, domain_ids, y_pred):
    scores = {}
    scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
    scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(
        y_true=y_true, y_pred=y_pred)
    scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
    scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
    scores['mse_entropy_reg'] = MOORLELoss(
        y_true, domain_ids, y_pred)

    return scores


def R2Loss(predicted, drug_ids, label):
    r2_score = R2Score().to(device)
    return 1 - r2_score(predicted, label)


def MSELossRegular(predicted, drug_ids, label):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(predicted, label)


def DrugwiseMSELoss(predicted, drug_ids, label):
    def weighted_mse_loss(input, target, weight):
        return torch.sum(weight * (input - target) ** 2)

    cross_drug_weight = 0.1
    base_level_of_interest = 1
    interest_scale = 0.1

    mse_loss = torch.nn.MSELoss()
    unique_drugs = torch.unique(drug_ids)
    mse_step = torch.empty(unique_drugs.size())
    i = 0
    full_mse = mse_loss(predicted, label)

    for unique_id in unique_drugs:
        unique_id_idx = (drug_ids == unique_id).nonzero(
            as_tuple=True)[0].cuda()
        # print(predicted)
        drug_preds = torch.gather(predicted, 0, unique_id_idx)
        drug_labels = torch.gather(label, 0, unique_id_idx)
        mse_step[i] = weighted_mse_loss(drug_preds, drug_labels, torch.add(
            torch.multiply(drug_labels, interest_scale), base_level_of_interest))

    return torch.abs(torch.mean(mse_step)) + cross_drug_weight * full_mse


class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, response_df, drug_df, rna_df, binding_df=None, add_noise=False):
        'Initialization'
        self.labels = torch.tensor(labels, dtype=dtype, device=device)
        self.list_IDs = list_IDs
        drug_df = drug_df.set_index('DrugID')

        drug_df['DrugID_int'] = [
            int(x.split('_')[-1]) for x in drug_df.index]  # 'DrugID'
        drug_df.set_index('DrugID_int', inplace=True, drop=True)

        rna_col_number = rna_df.shape[1]

        rna_dict = rna_df.to_dict('index')
        drug_dict = drug_df.to_dict('index')

        rna_torch_dict = {str(key): torch.tensor(
            [list(value.values())], dtype=dtype) for key, value in rna_dict.items()}
        self.rna_torch_dict = TensorDict(
            rna_torch_dict, batch_size=[1], device=device)

        drug_torch_dict = {str(key): torch.tensor(
            [value['drug_encoding']], dtype=torch.int) for key, value in drug_dict.items()}

        self.drug_torch_dict = TensorDict(
            drug_torch_dict, batch_size=[1], device=device)

        response_df['DrugID'] = [
            int(str(x).split('_')[-1]) for x in response_df['DrugID']]

        keys = list(self.drug_torch_dict.keys())
        self.drug_ids = torch.tensor(
            response_df['DrugID'].values, device=device)
        self.cancer_ids = torch.tensor(
            response_df['CancID'].values, device=device)
        self.add_noise = add_noise
        self.rna_diameters = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        drug_id = self.drug_ids[index].item()  # response['DrugID']
        cancer_id = self.cancer_ids[index].item()  # response['CancID']

        v_d = self.drug_torch_dict[str(drug_id)][0]  # ['drug_encoding']
        v_p = self.rna_torch_dict[str(cancer_id)][0]
        v_b = torch.zeros(1)  # np.array(self.binding_df.iloc[index])

        if self.add_noise:
            std = 0.05
            v_p = v_p + \
                np.random.normal(0, std, size=len(
                    self.rna_diameters))*self.rna_diameters
            v_p = np.array(v_p)
        y = self.labels[index]  # np.array(self.labels[index])

        return v_d, v_p, v_b, y, drug_id, cancer_id


class GBN(torch.nn.Module):
    # Adopted from TabNet
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=64, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class transformer(nn.Sequential):
    def __init__(self, input_dim_drug,
                 transformer_emb_size_drug, dropout,
                 transformer_n_layer_drug,
                 transformer_intermediate_size_drug,
                 transformer_num_attention_heads_drug,
                 transformer_attention_probs_dropout,
                 transformer_hidden_dropout_rate):
        super(transformer, self).__init__()

        self.emb = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              dropout)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

    def forward(self, v):
        e = v[:, 0]  # .long().to(device)
        e_mask = v[:, 1]  # .long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb, ex_e_mask)
        return encoded_layers[:, 0]


class MLP(nn.Sequential):
    def __init__(self, input_dim, mlp_hidden_dims=[1024, 256, 64], hidden_dim_out=256):
        input_dim_gene = input_dim
        hidden_dim_gene = hidden_dim_out
        mlp_hidden_dims_gene = mlp_hidden_dims
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList(
            [GBN(input_dim_gene)] +
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])  # [nn.Sequential(nn.Dropout(0.1), nn.BatchNorm1d(dims[0]))] +
        self.dropout = nn.Dropout(0)

    def forward(self, v):
        # predict
        # v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            if i == len(self.predictor) - 1:
                v = l(v)
            else:
                v = F.relu(l(self.dropout(v)))
        return v

    def set_dropout(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)


class Classifier(nn.Sequential):
    def __init__(self, args, model_drug, model_gene, model_binding=None):
        super(Classifier, self).__init__()
        # self.use_binding = args.use_binding
        if model_binding is None:
            self.use_binding = False
        self.input_dim_drug = args.input_dim_drug_classifier
        self.input_dim_gene = args.input_dim_gene_classifier
        self.input_dim_binding_classifier = args.input_dim_binding_classifier
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.model_binding = model_binding
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_dims = [1024, 1024, 512]  # [2048, 1024, 1024, 512, 256]
        # self.alpha = None
        layer_size = len(self.hidden_dims) + 1
        if not self.use_binding:
            self.input_dim_binding_classifier = 0

        dims = [self.input_dim_drug + self.input_dim_gene + self.input_dim_binding_classifier] + \
            self.hidden_dims + [1]
        self.dims = dims
        self.predictor = nn.ModuleList(
            [GBN(dims[0])] +
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_D, v_P, v_B):
        # each encoding

        v_D = self.model_drug(v_D)
        v_P = self.model_gene(v_P)

        # concatenate and classify
        v_f = None
        if self.use_binding:
            v_B = self.model_binding(v_B)
            v_f = torch.cat((v_D, v_P, v_B), 1)
        else:
            v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f


class DeepTTC:
    def __init__(self, modeldir, args):
        self.model_drug = transformer(args.input_dim_drug,
                                      args.transformer_emb_size_drug,
                                      args.dropout,
                                      args.transformer_n_layer_drug,
                                      args.transformer_intermediate_size_drug,
                                      args.transformer_num_attention_heads_drug,
                                      args.transformer_attention_probs_dropout,
                                      args.transformer_hidden_dropout_rate)
        # self.model_drug = MLP(input_dim=args.input_dim_drug, mlp_hidden_dims = [1024, 256, 64], hidden_dim_out=128)
        self.device = device
        self.modeldir = modeldir
        self.record_file = os.path.join(
            self.modeldir, "valid_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        self.args = args
        self.model = None
        self.data_cache = {}
        self.alpha = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def test(self, datagenerator, model):
        y_label = []
        y_pred = []
        drug_ids = []
        model.eval()
        for i, (v_drug, v_gene, v_binding, label, drug_id, cancer_id) in enumerate(datagenerator):
            score = model(v_drug, v_gene, v_binding)
            n = torch.squeeze(score, 1)
            dev_drug_id = drug_id  # Variable(torch.from_numpy(
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            drug_ids = drug_ids + dev_drug_id.flatten().tolist()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        loss = mean_squared_error(y_label, y_pred)
        model.train()

        return y_label, y_pred, \
            mean_squared_error(y_label, y_pred), \
            np.sqrt(mean_squared_error(y_label, y_pred)), \
            pearsonr(y_label, y_pred)[0], \
            pearsonr(y_label, y_pred)[1], \
            spearmanr(y_label, y_pred)[0], \
            spearmanr(y_label, y_pred)[1], \
            concordance_index(y_label, y_pred), \
            loss

    def train(self, train_response, train_drug, train_rna, val_response, val_drug, val_rna, domain_mode='drug'):
        domain = None
        if domain_mode is None:
            domain = None
        elif domain_mode == 'drug':
            domain = 'DrugID'
        elif domain_mode == 'cancer':
            domain_mode = 'CancID'
        else:
            raise Exception('Unknown domain mode')

        model_gene = MLP(input_dim=np.shape(train_rna)[1], mlp_hidden_dims=[
            1024, 256, 64], hidden_dim_out=self.args.input_dim_gene_classifier)
        model_gene.set_dropout(0.15)

        self.model = Classifier(self.args, self.model_drug, model_gene)
        learning_rate = self.args.learning_rate
        decay = 0  # 1e-3
        BATCH_SIZE = self.args.batch_size
        train_epoch = self.args.epochs
        earlier_stopping_num = 1500
        self.model = self.model.to(self.device)

        opt = torch.optim.Adam(self.model.parameters(),
                               lr=learning_rate, weight_decay=decay)

        def get_domain_weight(dataset, domain='DrugID'):
            if domain is None:
                return np.ones(np.shape(dataset)[0])
            weights = torch.empty(dataset.shape[0])
            unique_drugs, counts = np.unique(
                dataset[domain].values, return_counts=True)
            drug_map = {}
            for drug_id, count in zip(unique_drugs, counts):
                drug_map[drug_id] = 1./count

            i = 0
            for drug_id in dataset[domain]:
                weights[i] = drug_map[drug_id]
                i += 1

            return weights

        train_weights = get_domain_weight(train_response, domain)
        multiplication_coeff = 1
        train_sampler_weighted = torch.utils.data.sampler.WeightedRandomSampler(
            train_weights, multiplication_coeff*int(train_drug.shape[0]))  # 2*

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': False,
                  'num_workers': num_workers,
                  'drop_last': False}

        print(train_drug)
        print('DrugID' in train_drug.columns)
        train_drug = train_drug.reset_index()
        val_drug = val_drug.reset_index()
        training_generator_weighted = data.DataLoader(data_process_loader(
            train_response.index.values, train_response.Label.values, train_response, train_drug, train_rna, None, add_noise=False), sampler=train_sampler_weighted, **params)
        training_dataset = data_process_loader(
            train_response.index.values, train_response.Label.values, train_response, train_drug, train_rna, None, add_noise=False)
        training_generator_regular = data.DataLoader(
            training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=False)

        max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ['# epoch', "MSE", 'RMSE',
                               "Pearson Correlation", "with p-value",
                               'Spearman Correlation', "with p-value2",
                               "Concordance Index"]
        table = PrettyTable(valid_metric_header)
        def float2str(x): return '%0.4f' % x
        print('--- Go for Training ---')
        t_start = time.time()
        earlier_stopping_counter = 0
        min_loss_val = 1E10

        training_results = {}
        validation_results = {}

        def get_permuted_batches(training_generators):
            batches = []
            for training_generator in training_generators:
                for v_d, v_p, v_b, label, drug_ids, cancer_ids in training_generator:
                    batches.append(
                        (v_d, v_p, v_b, label, drug_ids, cancer_ids))
            random.shuffle(batches)
            return batches

        def train_step(opt, training_generator, loss_function=None):
            iteration_loss = 0
            for i, (v_d, v_p, v_b, label, drug_ids, cancer_ids) in enumerate(training_generator):
                domain_ids = None
                if domain_mode == 'drug':
                    domain_ids = drug_ids
                if domain_mode == 'cancer':
                    domain_ids = cancer_ids
                score = self.model(v_d, v_p, v_b)
                label = label
                if domain_ids is not None:
                    loss_fct = torch.nn.MSELoss()
                else:
                    loss_fct = torch.nn.MSELoss()
                if loss_function is not None:
                    loss_fct = loss_function
                n = torch.squeeze(score, 1)  # .float()
                loss = None
                if loss_fct is MOORLELoss:
                    loss = loss_fct(n, domain_ids, label, self.alpha)
                if type(loss_fct) is torch.nn.MSELoss:
                    loss = loss_fct(n, label)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                if (i % 1000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) +
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] +
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

        batches = None
        self.add_to_data_cache('training', train_response, train_drug,
                               train_rna)
        self.add_to_data_cache('validation', val_response, val_drug, val_rna)
        for epo in range(train_epoch):
            # if epo < (train_epoch / 2):
            #    train_step(opt, training_generator_regular, MSELossRegular)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #    with record_function("batches_sampling"):

            permuted_batches = """
            if batches is None or epo % 10 == 0:
                # batches = get_permuted_batches(
                #    [training_generator_weighted, training_generator_regular])
                batches = get_permuted_batches(
                    [training_generator_regular, training_generator_regular])
            else:
                random.shuffle(batches)
            #"""

            train_step(opt, training_generator_regular, None)

            loss_val = None
            with torch.set_grad_enabled(False):
                score_domain = domain
                if score_domain is None:
                    score_domain = 'DrugID'
                print('Starting evaluations...')
                training_preds = self.predict_cached('training')[1]
                training_results[epo] = (
                    train_response.Label, train_response['DrugID'], train_response['CancID'], training_preds)
                y_true, validation_preds, mse, rmse, \
                    person, p_val, \
                    spearman, s_p_val, CI, loss_val = self.predict_cached(
                        'validation')
                validation_results[epo] = (
                    val_response.Label, val_response['DrugID'], val_response['CancID'], validation_preds)

                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val, CI]))
                loss_val = mse
                valid_metric_record.append(lst)
                general_loss = str(loss_val.item())[:7]

                print(loss_val)
                if loss_val < min_loss_val:
                    model_max = copy.deepcopy(self.model)
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' with loss:' + general_loss +
                          ', MSE: ' + str(mse)[:7] +
                          ' , Pearson Correlation: ' + str(person)[:7] +
                          ' with p-value: ' + str(p_val) +
                          ' Spearman Correlation: ' + str(spearman)[:7] +
                          ' with p_value: ' + str(s_p_val) +
                          ' , Concordance Index: ' + str(CI)[:7])

            earlier_stopping_counter += 1
            if min_loss_val > loss_val:
                print(loss_val)
                min_loss_val = loss_val
                earlier_stopping_counter = 0
            if earlier_stopping_counter >= earlier_stopping_num:
                break

        self.model = model_max

        print('--- Training Finished ---')
        return training_results, validation_results

    def add_to_data_cache(self, dataset_name, response_data, drug_data, rna_data):
        drug_data = drug_data.reset_index(drop=True)
        print('predicting...')
        self.model.to(device)
        print('\tModel is on the device')
        info = data_process_loader(response_data.index.values,
                                   response_data.Label.values,
                                   response_data, drug_data, rna_data, None)
        print('\tData process loader is ready')
        params = {'batch_size': 512,
                  'shuffle': False,
                  'num_workers': num_workers,
                  'drop_last': False
                  }  # ,
        self.data_cache[dataset_name] = info  # batches

    def predict_cached(self, dataset_name):
        params = {'batch_size': 512,
                  'shuffle': False,
                  'num_workers': num_workers,
                  'drop_last': False
                  }
        if dataset_name not in self.data_cache:
            raise Exception(f'Dataset {dataset_name} is not stored in cache!')
        print('\tPredicting cached...')
        data_generator = data.DataLoader(
            self.data_cache[dataset_name], **params)
        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
            self.test(data_generator, self.model)
        print('\tReturning results')

        return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val

    def predict(self, response_data, drug_data, rna_data, binding_data):
        drug_data = drug_data.reset_index(drop=True)
        print('predicting...')
        self.model.to(device)
        print('\tModel is on the device')
        info = data_process_loader(response_data.index.values,
                                   response_data.Label.values,
                                   response_data, drug_data, rna_data, binding_data)
        print('\tData process loader is ready')
        params = {'batch_size': 512,
                  'shuffle': False,
                  'num_workers': num_workers,
                  'drop_last': False
                  }  # ,
        generator = data.DataLoader(info, **params)
        print('\tGenerator constructed')

        print('\tPrediction started')
        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
            self.test(generator, self.model)
        print('\tReturning results')

        return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def clean_model(self):
        import gc
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

    def preprocess(self, response_data, rna_data, drug_data, binding_data, response_metric='AUC', use_map=True):
        args = self.args
        obj = DataEncoding(args.vocab_dir, args.cancer_id,
                           args.sample_id, args.target_id, args.drug_id)
        drug_col = 'DrugID'
        if use_map:
            from cross_study_validation import get_drug_map
            drug_map = get_drug_map()
            to_drop = []
            for i in range(drug_data.shape[0]):
                drug_id = drug_data.loc[drug_data.index[i], drug_col]
                if drug_id in drug_map:
                    drug_data.loc[drug_data.index[i],
                                  drug_col] = drug_map[drug_id]
                else:
                    to_drop.append(i)
            if len(to_drop) > 0:
                drug_data = drug_data.drop(drug_data.index[to_drop], axis=0)

        drug_smiles = drug_data

        drugid2smile = dict(
            zip(drug_smiles[drug_col], drug_smiles['SMILES']))
        smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
            obj._drug2emb_encoder)
        uniq_smile_dict = dict(
            zip(drug_smiles['SMILES'].unique(), smile_encode))

        drug_data.drop(['SMILES'], inplace=True, axis=1)
        drug_data['smiles'] = [drugid2smile[i] for i in drug_data['DrugID']]
        drug_data['drug_encoding'] = [uniq_smile_dict[i]
                                      for i in drug_data['smiles']]
        drug_data = drug_data.reset_index()

        print(response_data.columns)
        response_data = response_data[['CancID', 'DrugID', response_metric]]
        response_data['Label'] = response_data[response_metric]

        response_data = response_data[['CancID', 'DrugID', response_metric]]
        response_data.columns = ['CancID', 'DrugID', 'Label']

        print('Preprocessing...!!!')
        return response_data, rna_data, drug_data, binding_data


if __name__ == '__main__':

    # step1 数据切分
    vocab_dir = '.'
    obj = DataEncoding(vocab_dir=vocab_dir)

    # 切分完成
    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)
    # encoding 完成
    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)

    # step2：构造模型
    modeldir = './Model_80'
    modelfile = modeldir + '/model.pt'
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    net = DeepTTC(modeldir=modeldir)
    net.train(train_drug=traindata, train_rna=train_rnadata,
              val_drug=testdata, val_rna=test_rnadata)
    net.save_model()
    print("Model Saved :{}".format(modelfile))
