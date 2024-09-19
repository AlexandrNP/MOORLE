import pickle
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sources = ['ccle', 'ctrp', 'gcsi', 'gdsc1', 'gdsc2']

def get_drug_map():
    import os
    drug_map_file = 'drug_map.pickle'
    if os.path.isfile(drug_map_file):
        return pickle.load( open(drug_map_file, 'rb') )
    drug_list = pd.read_csv('DrugLists/Comprehensive_Drug_List.txt', sep='\t')
    drug_map = {}
    for col in drug_list.columns[1:6]:
        if col == 'UniqueID':
            continue
        for unique_id, current_id in zip(drug_list['UniqueID'], drug_list[col]):
            drug_map[current_id] = unique_id
    pickle.dump(drug_map, open(drug_map_file, 'wb'))
    return drug_map

def groupby_src_and_print(df, print_fn=print):
    print_fn(df.groupby('SOURCE').agg(
        {'CancID': 'nunique', 'DrugID': 'nunique'}).reset_index())


def load_source(src, datadir, use_lincs=True, gene_set=None):
    pretty_indent = '#' * 10
    print('Load source datadir: ', datadir)
    print(f'{pretty_indent} {src.upper()} {pretty_indent}')

    

    drug_map = get_drug_map()
    # Load data
    responses = pd.read_csv(
        f"{datadir}/rsp_{src}.csv")      # Drug response
    
    dropped_idx = []
    for i in range(responses.shape[0]):
        drug_id = responses.loc[responses.index[i], 'DrugID']
        if drug_id not in drug_map:
            dropped_idx.append(i)
            continue
        responses.loc[responses.index[i], 'DrugID'] = drug_map[ drug_id ]

    responses.drop(responses.index[dropped_idx], axis=0, inplace=True)
    
    gene_expression = pd.read_csv(
        f"{datadir}/ge_{src}.csv")        # Gene expressions
    mordred_descriptors = pd.read_csv(
        f"{datadir}/mordred_{src}.csv")  # Mordred descriptors
    morgan_fingerprints = pd.read_csv(
        f"{datadir}/ecfp2_{src}.csv")    # Morgan fingerprints
    smiles = pd.read_csv(f"{datadir}/smiles_{src}.csv")   # SMILES



    # Use landmark genes
    if gene_set is not None:
        gene_set = pd.read_csv(gene_set, sep='\t').transpose().astype(str).values.squeeze().tolist()
        print(np.shape(gene_set))
        genes = gene_set + [str(x).lower() for x in gene_set]
        print(np.shape(genes))
        print('GENE SET')
        print(genes)
    if use_lincs:
        with open(f"{datadir}/../landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
    genes = ["ge_" + str(g) for g in genes]
    print(len(set(genes).intersection(set(gene_expression.columns[1:]))))
    genes = list(set(genes).intersection(set(gene_expression.columns[1:])))
    cols = ["CancID"] + genes
    gene_expression = gene_expression[cols]

    groupby_src_and_print(responses)
    print("Unique cell lines with gene expressions",
          gene_expression["CancID"].nunique())
    print("Unique drugs with Mordred",
          mordred_descriptors["DrugID"].nunique())
    print("Unique drugs with ECFP2",
          morgan_fingerprints["DrugID"].nunique())
    print("Unique drugs with SMILES", smiles["DrugID"].nunique())
    #bindings = pd.read_csv('docking_data.tsv', sep='\t')
    bindings = pd.read_csv('binding_affinity_postprocessed.tsv', sep='\t')
    return gene_expression, mordred_descriptors, morgan_fingerprints, smiles, bindings, responses


def score(y_true, y_pred):
    scores = {}
    scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
    scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(
        y_true=y_true, y_pred=y_pred)
    scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
    scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
    print(scores)
    return scores


def prepare_dataframe(gene_expression, smiles, bindings, responses, model):
    print(f'@@@ ORIGINAL DRUG DATA: {smiles.shape}')
    gene_expression, drug_data, binding_data = model.preprocess(
        gene_expression, smiles, bindings, responses, 'AUC')

    print(gene_expression.shape)
    print(smiles.shape)
    print(bindings.shape)
    print(responses.shape)
    print(gene_expression['CancID'])
    print(drug_data['CancID'])
    

    drug_data = drug_data.drop(['index'], axis=1)
    drug_columns = drug_data.columns #[x for x in drug_data.columns if x not in ['CancID', 'DrugID']]   

    #binding_data.columns = ['DrugID'] + list(binding_data.columns[1:])
    #data = pd.merge(drug_data, binding_data, on='DrugID', how='inner')
    
    drug_data = pd.merge(drug_data, binding_data, on='DrugID', how='inner')
    if 'DrugID' in binding_data.columns:
        binding_data = binding_data.drop(['DrugID'], axis=1)
    binding_columns = binding_data.columns
    
 
    print(drug_data)

    data = pd.merge(gene_expression, drug_data, on='CancID', how='inner')
    #data = pd.merge(gene_expression, data, on='CancID', how='inner')
    print(f'@@@@@ MERGED BINDING DATA SHAPE: {data.shape} @@@@@')
    

    #print(binding_data['DrugID'])
    #print('Data DrugID')
    #print(np.unique(data['DrugID']))

    gene_expression = gene_expression.drop(['CancID'], axis=1)
    gene_expression_columns = gene_expression.columns

    return data, gene_expression_columns, drug_columns, binding_columns


def run_cross_study_analysis(model, data_dir, results_dir, n_splits=10, use_lincs=True):
    for src in sources:
        datadir = f"{data_dir}/ml.dfs/July2020/data.{src}"
        splitdir = f"{datadir}/splits"

        print('Datadir: ', datadir)
        gene_expression, _, _, smiles, responses = load_source(
            src, datadir, use_lincs)
        print(gene_expression.shape)
        print(smiles.shape)
        print(responses.shape)
        
        data, gene_expression_columns, drug_columns, binding_columns = prepare_dataframe(
            gene_expression, smiles, responses, model)

        print(data)

        # -----------------------------------------------
        #   Train model
        # -----------------------------------------------
        # Example of training a DRP model with gene expression and Mordred descriptors
        print("\nGet the splits.")

        for split_id in range(n_splits):
            with open(f"{splitdir}/split_{split_id}_tr_id") as f:
                train_id = [int(line.rstrip()) for line in f]

            with open(f"{splitdir}/split_{split_id}_te_id") as f:
                test_id = [int(line.rstrip()) for line in f]

            # Train and test data
            train_data = data.loc[train_id]
            test_data = data.loc[test_id]
            test_cancid = test_data['CancID']
            test_drugid = test_data['DrugID']

            # Val data from tr_data
            train_data, validation_data = train_test_split(
                train_data, test_size=0.12)
            print("Train", train_data.shape)
            print("Val  ", validation_data.shape)
            print("Test ", test_data.shape)

            scaler = StandardScaler()
            train_data_gene_expression_scaled = pd.DataFrame(scaler.fit_transform(train_data[gene_expression_columns]))
            validation_data_gene_expression_scaled = pd.DataFrame(scaler.transform(validation_data[gene_expression_columns]))
            test_data_gene_expression_scaled = pd.DataFrame(scaler.transform(test_data[gene_expression_columns]))

            scaler = StandardScaler()
            train_data_binding_scaled = pd.DataFrame(scaler.fit_transform(train_data[binding_columns]))
            validation_data_binding_scaled = pd.DataFrame(scaler.transform(validation_data[binding_columns]))
            test_data_binding_scaled = pd.DataFrame(scaler.transform(test_data[binding_columns]))

            # Scale
            # Train model
            train_data.index = range(train_data.shape[0])
            validation_data.index = range(validation_data.shape[0])
            test_data.index = range(test_data.shape[0])
            model.train(train_drug=train_data[drug_columns], train_rna=train_data_gene_expression_scaled, train_binding=train_data_binding_scaled,
                        val_drug=validation_data[drug_columns], val_rna=validation_data_gene_expression_scaled, val_binding=validation_data_binding_scaled)

            # Predict
            # DeepTTC-specific prediction format
            _, y_pred, _, _, _, _, _, _, _ = model.predict(
                test_data[drug_columns], test_data_gene_expression_scaled, test_data_binding_scaled)
            y_true = test_data['Label']
            print('Y_true:')
            print(y_true)

            # Scores
            scores = score(y_true, y_pred)
            print(f'CancID: {np.shape(test_cancid)}, DrugID: {np.shape(test_drugid)}, y_true: {np.shape(y_true)}, y_pred: {np.shape(y_pred)}, train_data: {np.shape(train_data)}, test_data: {np.shape(test_data)}')
            result = {'CancID': test_cancid,
                      'DrugID': test_drugid,
                      'y_true': y_true,
                      'y_pred': y_pred}
            result_df = pd.DataFrame()
            for key in result:
                print(key)
                result_df[key] = np.array(result[key])
            print('Y_true 2:')
            print(np.shape(result_df))
            print(np.shape(y_true))
            print(result_df)
            #result_df = pd.DataFrame.from_dict(result)
            result_df.to_csv(f'{results_dir}/{src}_{src}_split_{split_id}.csv', sep=',', index=None)
            pickle.dump(result, open(
                f'{results_dir}/predictions_{src}_cv_split_{split_id}.pickle', 'wb'))
            #pickle.dump(scores, open(
            #    f'{results_dir}/scores_{src}_cv_split_{split_id}.pickle', 'wb'))

            # Test on unrelated datasets
            for test_src in sources:
                if test_src == src:
                    continue
                test_datadir = f"{data_dir}/ml.dfs/July2020/data.{test_src}"
                test_gene_expression, _, _, test_smiles, test_responses = load_source(
                    test_src, test_datadir, use_lincs)
                test_src_data, test_gene_expression_columns, test_drug_columns = prepare_dataframe(
                    test_gene_expression, test_smiles, test_responses, model)
                # Subsetting to the current set of expressed genes!!!
                test_src_data_gene_expression_scaled = pd.DataFrame(scaler.transform(test_src_data[gene_expression_columns]))
                _, y_pred, _, _, _, _, _, _, _ = model.predict(
                    test_src_data[test_drug_columns], test_src_data_gene_expression_scaled)
                y_true = test_src_data['Label']
                test_cancid = test_src_data['CancID']
                test_drugid = test_src_data['DrugID']

                #print(f'{src} on {test_src} y_true: ')

                result = {'CancID': test_cancid,
                          'DrugID': test_drugid,
                          'y_true': y_true,
                          'y_pred': y_pred}
                result_df = pd.DataFrame.from_dict(result)
                result_df.to_csv(f'{results_dir}/{src}_{test_src}_split_{split_id}.csv', sep=',', index=None)
                scores = score(y_pred, y_true)
                #pickle.dump(result, open(
                #    f'{results_dir}/predictions_{src}_split_{split_id}_on_{test_src}.pickle', 'wb'))
                pickle.dump(scores, open(
                    f'{results_dir}/scores_{src}_split_{split_id}_on_{test_src}.pickle', 'wb'))
