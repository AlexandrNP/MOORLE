import os
import pickle
import numpy as np
import pandas as pd
import improve_utils
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from improve_utils import improve_globals as ig
# from cross_study_validation import prepare_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split, GroupShuffleSplit, StratifiedShuffleSplit, KFold, ShuffleSplit


def generate_cv_partition(source, groups, n_splits=10, random_state=1, validation_size=0.1, main_cv_type='shuffle', validation_split_type='stratified', out_dir=None):
    out_filename = f'CV_partitions_{source}.pickle'
    print('CV partitions!!!')
    cv_path = None
    if out_dir is not None:
        cv_path = os.path.join(out_dir, out_filename)
        print(cv_path)
        if os.path.isfile(cv_path):
            print(cv_path)
            return pickle.load(open(cv_path, 'rb'))
    X = np.array(range(len(groups)))
    groups = np.array(groups)
    test_size = 1. / n_splits
    train_size = 1 - test_size
    validation_size = validation_size / train_size

    main_cv = None
    if main_cv_type == 'stratified':
        main_cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
    elif main_cv_type == 'shuffle':
        main_cv = KFold(n_splits=n_splits, shuffle=True,
                        random_state=random_state)
    elif main_cv_type == 'grouped':
        main_cv = GroupKFold(n_splits=n_splits)
    else:
        raise Exception('Unknown Main CV type!')

    validation_split = None
    if validation_split_type == 'stratified':
        validation_split = StratifiedShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=random_state)
    elif validation_split_type == 'shuffle':
        validation_split = ShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=random_state)
    elif validation_split_type == 'grouped':
        validation_split = GroupShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=random_state)
    else:
        raise Exception('Unknown Validation CV type!')

    cv_idx_splits = []
    for train_index_outer, test_index in main_cv.split(X, groups, groups):
        X_train = X[train_index_outer]
        groups_train = groups[train_index_outer]

        for train_index, validation_index in validation_split.split(X_train, groups_train, groups_train):
            # X contains indices of the original dataset
            train_index = X_train[train_index]
            validation_index = X_train[validation_index]
            cv_idx_splits.append(
                (train_index, validation_index, test_index))
            break

    if out_dir is not None:
        pickle.dump(cv_idx_splits, open(cv_path, 'wb'))
    return cv_idx_splits


def clean_meatadata_columns(columns):
    clean_columns = []
    removed = []
    metadata_columns = ['sample', 'cell', 'cancid', 'drugid', 'source']
    for col in columns:
        for meatadata_col in metadata_columns:
            if not meatadata_col in col.lower():
                clean_columns.append(col)
            else:
                removed.append(col)
    return clean_columns


def get_drug_map():
    import os
    drug_map_file = 'drug_map.pickle'
    if os.path.isfile(drug_map_file):
        return pickle.load(open(drug_map_file, 'rb'))
    drug_list = pd.read_csv('DrugLists/Comprehensive_Drug_List.txt', sep='\t')
    drug_map = {}
    for col in drug_list.columns[1:6]:
        if col == 'UniqueID':
            continue
        for unique_id, current_id in zip(drug_list['UniqueID'], drug_list[col]):
            drug_map[current_id] = unique_id
    pickle.dump(drug_map, open(drug_map_file, 'wb'))
    return drug_map


def calculate_scores(y_true, y_pred):
    performance = np.empty(7)
    performance.fill(np.nan)
    performance = pd.Series(performance, index=[
                            'R2', 'MSE', 'MAE', 'pearsonCor', 'pearsonCorPvalue', 'spearmanCor', 'spearmanCorPvalue'])
    performance.loc['R2'] = r2_score(y_true, y_pred)
    performance.loc['MSE'] = mean_squared_error(y_true, y_pred)
    performance.loc['MAE'] = mean_absolute_error(y_true, y_pred)
    rho, pval = pearsonr(y_true, y_pred)
    performance.loc['pearsonCor'] = rho
    performance.loc['pearsonCorPvalue'] = pval
    rho, pval = spearmanr(y_true, y_pred)
    performance.loc['spearmanCor'] = rho
    performance.loc['spearmanCorPvalue'] = pval
    return performance


def load_data_deep(source, datadir, gene_set=None):
    bindings = None
    pretty_indent = '#' * 10
    print(f'{pretty_indent} {source.upper()} {pretty_indent}')
    source = source.lower()

    # Load data
    from preprocess import get_gene_expression_data, get_drug_data, get_response_data, check_file_and_run
    gene_set_name = None
    if gene_set == 'ALL':
        gene_set = None
    if gene_set is not None:
        gene_set_name = gene_set.split('/')[-1].split('.')[0]
    else:
        gene_set_name = 'ALL'
    gene_data_filename = f'{source}_gene_expression_{gene_set_name.lower()}_binding_affinity_genes.tsv'
    drug_data_filename = 'drug_data_imputed_1.tsv'
    gene_expression = check_file_and_run(gene_data_filename, get_gene_expression_data, [
                                         'Combat_AllGenes_UniqueSample.txt', source, gene_set_name])

    gene_expression.columns = [
        'CancID' if x == 'CELL' else f'ge_{x.lower()}' for x in gene_expression.columns]

    drug_descriptors = check_file_and_run(drug_data_filename, get_drug_data, [
                                          'JasonPanDrugsAndNCI60_dragon7_descriptors.tsv'])
    responses = get_response_data('drug_response_data.txt', source)
    responses.columns = ['CancID', 'DrugID', 'AUC']

    drug_info = pd.read_csv('DrugLists/Comprehensive_Drug_List.txt', sep='\t')
    smiles_cols = ['SMILES(drug_info)', 'SMILES(Jason_SMILES)',
                   'SMILES(NCI60_drug)', 'smiles(Broad)']

    smiles = drug_info[smiles_cols].groupby(
        {x: 'SMILES' for x in drug_info[smiles_cols].columns}, axis=1).first().dropna()
    smiles['DrugID'] = drug_info[source.upper()]
    smiles = smiles.dropna()

    # Use landmark genes
    if gene_set is not None:
        gene_set = pd.read_csv(gene_set, sep='\t').transpose().astype(
            str).values.squeeze().tolist()
        genes = gene_set + [str(x).lower() for x in gene_set]
        genes = ["ge_" + str(g) for g in genes]
    else:
        genes = gene_expression.columns[1:]

    genes = list(set(genes).intersection(set(gene_expression.columns[1:])))
    cols = ["CancID"] + genes
    gene_expression = gene_expression[cols]

    return gene_expression, drug_descriptors, None, smiles, responses, bindings


def prepare_dataframe(gene_expression, smiles, bindings, responses, model):
    print(f'@@@ ORIGINAL DRUG DATA: {smiles.shape}')
    response_metric = 'AUC'
    gene_expression, drug_data, binding_data = model.preprocess(
        gene_expression, smiles, bindings, responses, response_metric, use_map=True)

    drug_data = drug_data.drop(['index'], axis=1)

    if 'DrugID' in binding_data.columns:
        drug_data = pd.merge(drug_data, binding_data, on='DrugID', how='inner')
        binding_data = binding_data.drop(['DrugID'], axis=1)
    binding_columns = binding_data.columns

    gene_expression = gene_expression.loc[:, ~
                                          gene_expression.columns.duplicated()].copy()
    drug_data = drug_data.loc[:, ~drug_data.columns.duplicated()].copy()
    drug_columns = drug_data.columns
    drug_columns = list(drug_columns) + ['CancID']

    data = pd.merge(gene_expression, drug_data, on='CancID', how='inner')
    print(f'@@@@@ MERGED BINDING DATA SHAPE: {data.shape} @@@@@')

    gene_expression = gene_expression.drop(['CancID'], axis=1)
    gene_expression_columns = gene_expression.columns

    return data, gene_expression_columns, drug_columns, binding_columns


def prepare_dataframe_separated(responses, gene_expression, smiles, bindings, model):
    print(f'@@@ ORIGINAL DRUG DATA: {smiles.shape}')
    response_metric = 'AUC'
    responses, gene_expression, drug_data, binding_data = model.preprocess(
        responses, gene_expression, smiles, bindings, response_metric, use_map=True)

    drug_data = drug_data.drop(['index'], axis=1)

    binding_columns = binding_data.columns

    gene_expression = gene_expression.loc[:, ~
                                          gene_expression.columns.duplicated()].copy()
    drug_data = drug_data.loc[:, ~drug_data.columns.duplicated()].copy()
    drug_columns = drug_data.columns
    drug_columns = list(drug_columns) + ['CancID']

    print(f'@@@@@ MERGED BINDING DATA SHAPE: {drug_data.shape} @@@@@')

    gene_expression.set_index('CancID', drop=True, inplace=True)
    gene_expression_columns = gene_expression.columns

    return responses, gene_expression, drug_data, gene_expression_columns, drug_columns, binding_columns


def run_cross_benchmark(model):
    domains_mode = 'drug'  # None
    domains_col = 'DrugID'
    do_not_recompute_cv = False
    if domains_mode == 'drug':
        domains_col = 'DrugID'
    elif domains_mode == 'cancer':
        domains_col = 'CancID'
    # Regular sampling refers to sequential sampling during training phase of the model
    # with a shuffle. Batch size 64.
    #
    # Mixed sampling referes to the combined sampling approach that includes all batches
    # derived from sequential sampling plus stratified batches obtained from weighted
    # sampling. Batches are regenerated each 10 epochs and for each subsequent epoch they
    # are permuted. Batch size 64.
    #
    # ['ccle', 'ctrp']
    sources = ['ctrp']
    data_dir = 'raw_data'
    deepttc = True
    drug_list = pd.read_csv('DrugLists/Comprehensive_Drug_List.txt', sep='\t')
    drug_list = drug_list[['UniqueID', 'CTRP', 'GDSC', 'CCLE']]

    binding_affinity_data = pd.read_csv(
        'genes_gausschem4_scores_stat.tsv', sep='\t')
    binding_affinity_data.columns = [
        'Drug_UniqueID'] + list(binding_affinity_data.columns[1:])

    gene_set_name = 'oncogenes_gausschem4'

    sets = {}
    all_results = {}
    for alpha in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        results_dir = f'Results/DeepTTC_drug_blind_seq_sampling_entropy_alpha_{alpha}_epochs_100'
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        for source in sources:
            source = source.lower()

            datadir = f"{data_dir}/data.{source}"
            gene_expression, descriptors, morgan, smiles, responses, bindings = load_data_deep(
                source, datadir, gene_set='ALL')

            preprocessor = model
            data = None
            if deepttc:
                responses, gene_expression, drug_data, gene_expression_columns, drug_columns, binding_columns = prepare_dataframe_separated(
                    responses, gene_expression, smiles, bindings, preprocessor)
                drug_data = drug_data.set_index('DrugID')

            suffix = ''
            if deepttc:
                suffix = 'DeepTTC'
            suffix = f'{suffix}_{domains_mode}'
            label_encoder_path = os.path.join(
                results_dir, f'label_encoder_{source}_{gene_set_name.lower()}_{suffix}.pickle')
            cancer_label_encoder = LabelEncoder()
            gene_expression.index = cancer_label_encoder.fit_transform(
                gene_expression.index)
            responses['CancID'] = cancer_label_encoder.transform(
                responses['CancID'])

            pickle.dump(cancer_label_encoder, open(label_encoder_path, 'wb'))

            groups = responses['DrugID']
            cv_type = 'grouped'
            cv_partitions = generate_cv_partition(source, groups, n_splits=10, random_state=1, validation_size=0.1,
                                                  main_cv_type=cv_type, validation_split_type='stratified', out_dir=results_dir)

            set_name = f'{source}'
            print(f'Dataset size: {responses.shape}')
            for split_idx in range(len(cv_partitions)):
                train_idx, val_idx, test_idx = cv_partitions[split_idx]
                training_results_path = os.path.join(
                    results_dir, f'training_results_{set_name}_{suffix}_split_{split_idx}.pickle')
                validation_results_path = os.path.join(
                    results_dir, f'validation_results_{set_name}_{suffix}_split_{split_idx}.pickle')

                print(f'CV iteration {split_idx}')
                if os.path.isfile(training_results_path) and do_not_recompute_cv:
                    if split_idx < 10:
                        continue

                if deepttc:
                    from copy import deepcopy
                    print(f'Dataset size: {responses.shape[0]}')
                    train_response = responses.loc[responses.index[train_idx]]
                    train_drug_id = np.unique(train_response['DrugID'])
                    train_rna_id = np.unique(train_response['CancID'])
                    train_drug = drug_data.loc[train_drug_id]
                    train_rna = gene_expression.loc[train_rna_id]

                    val_response = responses.loc[responses.index[val_idx]]
                    val_drug_id = np.unique(val_response['DrugID'])
                    val_rna_id = np.unique(val_response['CancID'])
                    val_drug = drug_data.loc[val_drug_id]
                    val_rna = gene_expression.loc[val_rna_id]
                    test_response = responses.loc[responses.index[test_idx]]
                    test_drug_id = np.unique(test_response['DrugID'])
                    test_rna_id = np.unique(test_response['CancID'])
                    test_drug = drug_data.loc[test_drug_id]
                    test_rna = gene_expression.loc[test_rna_id]

                    rna_scaler = StandardScaler()
                    train_rna[train_rna.columns] = rna_scaler.fit_transform(
                        train_rna.values)
                    val_rna[val_rna.columns] = rna_scaler.transform(
                        val_rna.values)
                    test_rna[test_rna.columns] = rna_scaler.transform(
                        test_rna.values)

                    sets[set_name] = (test_response, test_drug, test_rna, None)
                    model.set_alpha(alpha)
                    training_results, validation_results = model.train(train_response, train_drug, train_rna,
                                                                       val_response, val_drug, val_rna, domain_mode=domains_mode)
                    pickle.dump(training_results, open(
                        training_results_path, 'wb'))
                    print('!!!!!!!!!!!!!!!!!')
                    print(training_results_path)
                    print('!!!!!!!!!!!!!!!!!')
                    pickle.dump(validation_results, open(
                        validation_results_path, 'wb'))

                    response_test_set, drug_test_set, rna_test_set, binding_test_set = sets[
                        set_name]
                    drug_test_set['DrugID'] = drug_test_set.index.values
                    y_test = response_test_set['Label']
                    # breakpoint()
                    _, y_pred, _, _, _, _, _, _, _, _ = model.predict(
                        response_test_set, drug_test_set, rna_test_set, binding_test_set)
                    res = pd.DataFrame.from_dict(
                        {'Test': y_test, 'Pred': y_pred})
                    test_results_path = os.path.join(
                        results_dir, f'test_results_{set_name}_{suffix}_split_{split_idx}.pickle')
                    test_results = {}
                    test_results[0] = (y_test, response_test_set['DrugID'],
                                       response_test_set['CancID'], y_pred)
                    pickle.dump(test_results, open(test_results_path, 'wb'))
                    print(res)
                cv_scores = calculate_scores(y_test.values, y_pred)
                all_results[set_name][split_idx] = cv_scores

                out_path = os.path.join(
                    results_dir, f'{set_name}_{set_name}_{suffix}_split_{split_idx}.tsv')
                cv_scores = pd.DataFrame(cv_scores)
                cv_scores.to_csv(out_path, sep='\t')
                print(cv_scores)

            for set_name in sets:
                results = pd.DataFrame.from_dict(
                    all_results[set_name], orient='index')
                summarized_results = results.mean(axis=0)
                print(results)
                print(summarized_results)

                full_out_file_name = f'test_{source}_{gene_set_name.lower()}_{suffix}_full_cv_scores.tsv'
                full_out_path = os.path.join(results_dir, full_out_file_name)
                summarized_out_file_name = f'test_{source}_{gene_set_name.lower()}_{suffix}_mean_cv_scores.tsv'
                summarized_out_path = os.path.join(
                    results_dir, summarized_out_file_name)

                results.to_csv(full_out_path, sep='\t')
                summarized_results.to_csv(
                    summarized_out_path, sep='\t', header=None)
