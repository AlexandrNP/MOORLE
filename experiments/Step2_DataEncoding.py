# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-DataEncoding.py
@time:2021/9/7 10:04 
"""
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE
from Step1_getData import GetData


class DataEncoding:
    def __init__(self, vocab_dir, cancer_id, sample_id, target_id, drug_id):
        self.vocab_dir = vocab_dir
        # 一个获取数据的函数类
        self.Getdata = GetData(cancer_id, sample_id, target_id, drug_id)

    def _drug2emb_encoder(self, smile):
        vocab_path = "{}/ESPF/drug_codes_chembl_freq_1500.txt".format(
            self.vocab_dir)
        sub_csv = pd.read_csv(
            "{}/ESPF/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def encode(self, traindata, testdata, args=None):
        drug_smiles = self.Getdata.getDrug()
        print(drug_smiles)
        drugid2smile = dict(
            zip(drug_smiles['drug_id'], drug_smiles['CanonicalSMILES']))
        smile_encode = pd.Series(drug_smiles['CanonicalSMILES'].unique()).apply(
            self._drug2emb_encoder)
        uniq_smile_dict = dict(
            zip(drug_smiles['CanonicalSMILES'].unique(), smile_encode))

        traindata['smiles'] = [drugid2smile[i] for i in traindata['DRUG_ID']]
        testdata['smiles'] = [drugid2smile[i] for i in testdata['DRUG_ID']]
        traindata['drug_encoding'] = [uniq_smile_dict[i]
                                      for i in traindata['smiles']]
        testdata['drug_encoding'] = [uniq_smile_dict[i]
                                     for i in testdata['smiles']]
        traindata = traindata.reset_index()
        traindata['Label'] = traindata[args.target_id]
        testdata = testdata.reset_index()
        testdata['Label'] = testdata[args.target_id]

        train_rnadata, test_rnadata = self.Getdata.getRna(
            traindata=traindata,
            testdata=testdata,
            args=args)

        cols = train_rnadata.index

        print('TRAIN RNA DATA')
        print(train_rnadata)
        print('TRAIN RNA DATA IDX')
        print(cols)
 

        train_rnadata = train_rnadata.T
        train_rnadata.columns = cols
        test_rnadata = test_rnadata.T
        test_rnadata.columns = cols

        train_rnadata.index = range(train_rnadata.shape[0])
        test_rnadata.index = range(test_rnadata.shape[0])


        return traindata, train_rnadata, testdata, test_rnadata


if __name__ == '__main__':
    vocab_dir = '.'
    obj = DataEncoding(vocab_dir=vocab_dir)
    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)

    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)

    print(traindata, train_rnadata, testdata, test_rnadata)
