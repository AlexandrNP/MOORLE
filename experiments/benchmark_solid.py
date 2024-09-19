import os
import numpy as np
import pandas as pd
import improve_utils
from copy import deepcopy
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from improve_utils import improve_globals as ig
from cross_study_validation import prepare_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def prepare_dataframe(gene_expression, smiles, responses, model):
    response_metric = 'AUC'
    pseudobindings = deepcopy(gene_expression[gene_expression.columns[-3:-1]])
    pseudobindings.columns = ['a', 'b']
    # breakpoint()
    #gene_expression, drug_data, binding_data = model.preprocess(
    #    gene_expression, smiles, pseudobindings, responses, response_metric, use_map=False)
    gene_expression, drug_data = model.preprocess(
        gene_expression, smiles, responses, response_metric)
    if response_metric in drug_data.columns:
        drug_data.drop([response_metric], axis=1, inplace=True)
    drug_data = drug_data.drop(['index'], axis=1)
    drug_columns = [
        x for x in drug_data.columns if x not in ['CancID']]
    # data = pd.merge(gene_expression, drug_data, on='DrugID', how='inner')
    data = pd.merge(gene_expression, drug_data, on='CancID', how='inner')
    gene_expression = gene_expression.drop(['CancID'], axis=1)
    gene_expression_columns = gene_expression.columns

    return data, gene_expression_columns, drug_columns


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


def run_cross_benchmark(model):
    results_dir = os.path.join('Results', 'CTRPv2')
    sets = set()
    all_results = {}
    datasets = ["CTRPv2"]  # "CCLE", "CTRPv2", "GDSCv2"
    test_datasets = ['CCLE', 'GDSCv1', 'GDSCv2', 'gCSI']

    for dataset in datasets:
        all_results[dataset] = {}
        sets.add(dataset)
        set_name = f'{dataset}'
        for split in range(1):

            split_idx = split
            out_path = os.path.join(
                results_dir, f'{set_name}_{set_name}_split_{split_idx}.tsv')

            source_data_name = dataset
            y_col_name = "auc1"
            fdir = Path(__file__).resolve().parent

            train_ids = improve_utils.load_split_ids(
                [f"{source_data_name}_split_{split}_train.txt"])
            validation_ids = improve_utils.load_split_ids(
                [f"{source_data_name}_split_{split}_val.txt"])
            test_ids = improve_utils.load_split_ids(
                [f"{source_data_name}_split_{split}_test.txt"])

            # Load train
            rs_tr = improve_utils.load_single_drug_response_data_v2(
                source=source_data_name,
                split_file_name=f"{source_data_name}_split_{split}_train.txt",
                y_col_name=y_col_name)

            # Load val
            rs_vl = improve_utils.load_single_drug_response_data_v2(
                source=source_data_name,
                split_file_name=f"{source_data_name}_split_{split}_val.txt",
                y_col_name=y_col_name)

            # Load test
            rs_te = improve_utils.load_single_drug_response_data_v2(
                source=source_data_name,
                split_file_name=f"{source_data_name}_split_{split}_test.txt",
                y_col_name=y_col_name)

            print("\nResponse train data", rs_tr.shape)
            print("Response val data", rs_vl.shape)
            print("Response test data", rs_te.shape)

            # Load omic feature data
            # cv = improve_utils.load_copy_number_data(gene_system_identifier="Gene_Symbol")
            ge = improve_utils.load_gene_expression_data(
                gene_system_identifier="Gene_Symbol")
            # mt = improve_utils.load_dna_methylation_data(gene_system_identifier="TSS")

            # Load drug feature data
            sm = improve_utils.load_smiles_data()
            dd = improve_utils.load_mordred_descriptor_data()
            fp = improve_utils.load_morgan_fingerprint_data()
            sm.set_index(sm.columns[0], inplace=True)

            # import pdb; pdb.set_trace()
            print(f"Total unique cells: {rs_tr[ig.canc_col_name].nunique()}")
            print(f"Total unique drugs: {rs_tr[ig.drug_col_name].nunique()}")
            assert len(set(rs_tr[ig.canc_col_name]).intersection(
                set(ge.index))) == rs_tr[ig.canc_col_name].nunique(), "Something is missing..."
            assert len(set(rs_tr[ig.drug_col_name]).intersection(
                set(fp.index))) == rs_tr[ig.drug_col_name].nunique(), "Something is missing..."
            # print(rs_tr[ig.drug_col_name])
            print(sm.index)
            print(
                len(set(rs_tr[ig.drug_col_name]).intersection(set(sm.index))))
            assert len(set(rs_tr[ig.drug_col_name]).intersection(set(
                sm.index))) == rs_tr[ig.drug_col_name].nunique(), "Something is missing..."  # TODO: check this!

            # Preprocess
            from copy import deepcopy
            sm.reset_index(inplace=True)
            ge.reset_index(inplace=True)
            sm.columns = ['DrugID', 'SMILES']
            sm_orig = deepcopy(sm)
            rs_tr.columns = ['source', 'DrugID', 'CancID', 'AUC']
            rs_vl.columns = ['source', 'DrugID', 'CancID', 'AUC']
            rs_te.columns = ['source', 'DrugID', 'CancID', 'AUC']
            ge_cols = list(ge.columns[1:])
            print(len(ge_cols))
            print(['CancID'] + ge_cols)
            ge.columns = ['CancID'] + ge_cols
            print(sm.columns)
            print(rs_tr.columns)
            data_train, gene_expression_columns, drug_columns = prepare_dataframe(
                ge, sm, rs_tr, model)
            sm = deepcopy(sm_orig)
            data_val, gene_expression_columns, drug_columns = prepare_dataframe(
                ge, sm, rs_vl, model)
            sm = deepcopy(sm_orig)
            data_test, gene_expression_columns, drug_columns = prepare_dataframe(
                ge, sm, rs_te, model)
            # Train model here
            scaler = StandardScaler()
            data_train[gene_expression_columns] = scaler.fit_transform(
                data_train[gene_expression_columns])
            data_val[gene_expression_columns] = scaler.transform(
                data_val[gene_expression_columns])
            data_test[gene_expression_columns] = scaler.transform(
                data_test[gene_expression_columns])
            print(data_val[drug_columns])
            model.train(train_drug=data_train[drug_columns], train_rna=data_train[gene_expression_columns],
                        val_drug=data_val[drug_columns], val_rna=data_val[gene_expression_columns])
            #model.train(train_drug=data_train[drug_columns], train_rna=data_train[gene_expression_columns], train_binding=data_train[gene_expression_columns],
            #            val_drug=data_val[drug_columns], val_rna=data_val[gene_expression_columns], val_binding=data_val[gene_expression_columns])

            print("\nAdd model predictions to dataframe and save")
            preds_df = rs_te.copy()
            print(preds_df.head())
            # DL model predictions
            y_col_name = 'AUC'
            #_, model_preds, _, _, _, _, _, _, _ = model.predict(
            #    data_test[drug_columns], data_test[gene_expression_columns], data_test[gene_expression_columns])  # rs_te[y_col_name]  # + \
            _, model_preds, _, _, _, _, _, _, _ = model.predict(
                data_test[drug_columns], data_test[gene_expression_columns])  # rs_te[y_col_name]  # + \
            #    np.random.normal(loc=0, scale=0.1, size=rs_te.shape[0])
            preds_df[y_col_name + ig.pred_col_name_suffix] = model_preds
            print(preds_df.head())
            # import pdb
            # pdb.set_trace()
            # TODO: we will determine later what should be the output dir for model predictions
            outdir_preds = fdir/"model_preds"
            os.makedirs(outdir_preds, exist_ok=True)
            outpath = outdir_preds/"test_preds.csv"
            preds_df.columns = ['source', 'improve_chem_id',
                                'improve_sample_id', 'auc1', 'auc1_pred']
            y_col_name = 'auc1'
            cols_drugs_df = pd.DataFrame(drug_columns)
            cols_df = pd.DataFrame(gene_expression_columns)
            cols_df.to_csv('gene_expression_cols.csv')
            cols_drugs_df.to_csv('drug_cols.csv')
            y_test = data_test['Label']
            y_pred = model_preds
            split_idx = split
            r2 = r2_score(data_test['Label'], model_preds)
            mse = mean_squared_error(data_test['Label'], model_preds)
            print('TEST RESULTS')
            print(f'R2: {r2}')
            print(f'MSE: {mse}')
            improve_utils.save_preds(preds_df, y_col_name, outpath)

            cv_scores = calculate_scores(y_test, y_pred)
            all_results[set_name][split_idx] = cv_scores
            out_path = os.path.join(
                results_dir, 
                f'{set_name}_{set_name}_split_{split_idx}.tsv')
            cv_scores = pd.DataFrame(cv_scores)
            cv_scores.to_csv(out_path, sep='\t')
            print(cv_scores)

            for test_dataset in test_datasets:
                source_data_name = test_dataset
                y_col_name = "auc1"
                fdir = Path(__file__).resolve().parent
                
                rs_te = improve_utils.load_single_drug_response_data_v2(
                    source=source_data_name,
                    split_file_name=f"{source_data_name}_all.txt",
                    y_col_name=y_col_name)

                ge = improve_utils.load_gene_expression_data(
                    gene_system_identifier="Gene_Symbol")

                # Load drug feature data
                sm = improve_utils.load_smiles_data()
                dd = improve_utils.load_mordred_descriptor_data()
                fp = improve_utils.load_morgan_fingerprint_data()
                sm.set_index(sm.columns[0], inplace=True)

                # Preprocess
                sm.reset_index(inplace=True)
                ge.reset_index(inplace=True)
                sm.columns = ['DrugID', 'SMILES']
                sm_orig = deepcopy(sm)
                rs_tr.columns = ['source', 'DrugID', 'CancID', 'AUC']
                rs_vl.columns = ['source', 'DrugID', 'CancID', 'AUC']
                rs_te.columns = ['source', 'DrugID', 'CancID', 'AUC']
                ge_cols = list(ge.columns[1:])
                ge.columns = ['CancID'] + ge_cols

                data_test, gene_expression_columns, drug_columns = prepare_dataframe(
                    ge, sm, rs_te, model)
                data_test[gene_expression_columns] = scaler.transform(
                    data_test[gene_expression_columns])
                preds_df = rs_te.copy()
                # DL model predictions
                y_col_name = 'AUC'
                _, model_preds, _, _, _, _, _, _, _ = model.predict(
                    data_test[drug_columns], data_test[gene_expression_columns])  # rs_te[y_col_name]  # + \
                preds_df[y_col_name + ig.pred_col_name_suffix] = model_preds
                # TODO: we will determine later what should be the output dir for model predictions
                outdir_preds = fdir/"model_preds"
                os.makedirs(outdir_preds, exist_ok=True)
                outpath = outdir_preds/f"{test_dataset}_test_preds.csv"
                preds_df.columns = ['source', 'improve_chem_id',
                                'improve_sample_id', 'auc1', 'auc1_pred']
                y_col_name = 'auc1'
                # print(rs_te)
                # print(data_test.columns)
                cols_drugs_df = pd.DataFrame(drug_columns)
                cols_df = pd.DataFrame(gene_expression_columns)
                cols_df.to_csv('gene_expression_cols.csv')
                cols_drugs_df.to_csv('drug_cols.csv')
                y_test = data_test['Label']
                y_pred = model_preds
                
                r2 = r2_score(data_test['Label'], model_preds)
                mse = mean_squared_error(data_test['Label'], model_preds)
                print(test_dataset)
                print('TEST RESULTS')
                print(f'R2: {r2}')
                print(f'MSE: {mse}')
                print('######################')
                improve_utils.save_preds(preds_df, y_col_name, outpath)


        for set_name in sets:
            results = pd.DataFrame.from_dict(
                all_results[set_name], orient='index')
            summarized_results = results.mean(axis=0)
            print(results)
            print(summarized_results)

            full_out_file_name = f'test_{dataset}_all_genes_full_cv_scores.tsv'
            full_out_path = os.path.join(results_dir, full_out_file_name)
            summarized_out_file_name = f'test_{dataset}_all_genes_mean_cv_scores.tsv'
            summarized_out_path = os.path.join(
                results_dir, summarized_out_file_name)

            results.to_csv(full_out_path, sep='\t')
            summarized_results.to_csv(
                summarized_out_path, sep='\t', header=None)

        print("Finished.")
