from entropy_performance_evaluation import run_cross_benchmark
# from benchmark_solid import run_cross_benchmark
from cross_study_validation import run_cross_study_analysis
from Step2_DataEncoding import DataEncoding
# from new_Step3_model import *
from entropy_Step3_model_dict_loader import *
import subprocess
from candle.file_utils import get_file
import candle
import torch
import os
import sys
import cProfile
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
print(sys.path)


file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = [
    {
        "name": "save_data",
        "type": bool,
        "help": "Whether to save loaded data in pickle files",
    },
    {
        "name": "use_lincs",
        "type": bool,
        "help": "Whether to use a LINCS subset of genes ONLY",
    },
    {
        "name": "benchmark_dir",
        "type": str,
        "help": "Directory with the input data for benchmarking",
    },
    {
        "name": "benchmark_result_dir",
        "type": str,
        "help": "Directory for benchmark output",
    },
    {
        "name": "generate_input_data",
        "type": bool,
        "help": "'True' for generating input data anew, 'False' for using stored data",
    },
    {
        "name": "mode",
        "type": str,
        "help": "Execution mode. Available modes are: 'run', 'benchmark'",
    },
    {
        "name": "cancer_id",
        "type": str,
        "help": "Column name for cancer",
    },
    {
        "name": "drug_id",
        "type": str,
        "help": "Column name for drug",
    },
    {
        "name": "sample_id",
        "type": str,
        "help": "Column name for samples/cell lines",
    },
    {
        "name": "target_id",
        "type": str,
        "help": "Column name for target",
    },
    {
        "name": "vocab_dir",
        "type": str,
        "help": "Directory with ESPF vocabulary",
    },
    {
        "name": "transformer_num_attention_heads_drug",
        "type": int,
        "help": "number of attention heads for drug transformer",
    },
    {
        "name": "input_dim_drug",
        "type": int,
        "help": "Input size of the drug transformer",
    },
    {
        "name": "transformer_emb_size_drug",
        "type": int,
        "help": "Size of the drug embeddings",
    },
    {
        "name": "transformer_n_layer_drug",
        "type": int,
        "help": "Number of layers for drug transformer",
    },
    {
        "name": "transformer_intermediate_size_drug",
        "type": int,
        "help": "Intermediate size of the drug layers",
    },
    {
        "name": "transformer_attention_probs_dropout",
        "type": int,
        "help": "number of layers for drug transformer",
    },
    {
        "name": "transformer_hidden_dropout_rate",
        "type": int,
        "help": "dropout rate for transformer hidden layers",
    },
    {
        "name": "input_dim_drug_classifier",
        "type": int,
        "help": "dropout rate for classifier hidden layers",
    },
    {
        "name": "input_dim_gene_classifier",
        "type": int,
        "help": "dropout rate for classifier hidden layers",
    },
]

required = None


class DeepTTCCandle(candle.Benchmark):
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


class DataLoader:
    args = None

    def __init__(self, args):
        self.args = args

    def load_data(self):
        train_drug, test_drug, train_rna, test_rna = self._process_data(
            self.args)
        return train_drug, test_drug, train_rna, test_rna

    def save_data(self, train_drug, test_drug, train_rna, test_rna):
        args = self.args

        pickle.dump(train_drug, open(args.train_data_drug, 'wb'), protocol=4)
        pickle.dump(test_drug, open(args.test_data_drug, 'wb'), protocol=4)
        pickle.dump(train_rna, open(args.train_data_rna, 'wb'), protocol=4)
        pickle.dump(test_rna, open(args.test_data_rna, 'wb'), protocol=4)

        self.args = args

    def _download_default_dataset(self, default_data_url):
        url = default_data_url
        # candle_data_dir = os.getenv("CANDLE_DATA_DIR")
        # if candle_data_dir is None:
        #    candle_data_dir = '.'
        data_dir = self.args.data_dir

        # import shutil
        # import pathlib
        # cwd = os.path.dirname(os.path.abspath(__file__))
        # shutil.copy(f'{cwd}/landmark_genes', data_dir)

        OUT_DIR = os.path.join(data_dir, 'GDSC_data')
        # get_file(fname='DeepTTC_data.tar.gz', origin=url, unpack=True, data_dir=data_dir)

        url_length = len(url.split('/'))-4
        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)
        import wget
        subprocess.run(['wget', '--recursive', '--no-clobber', '-nH',
                        f'--cut-dirs={url_length}', '--no-parent', f'--directory-prefix={OUT_DIR}', f'{url}'])
        wget.download(url, out=OUT_DIR)

    def _process_data(self, args):
        train_drug = test_drug = train_rna = test_rna = None

        args.train_data_drug = os.path.join(
            args.data_dir, 'train_data_drug.pickle')
        args.test_data_drug = os.path.join(
            args.data_dir, 'test_data_drug.pickle')
        args.train_data_rna = os.path.join(
            args.data_dir, 'train_data_rna.pickle')
        args.test_data_rna = os.path.join(
            args.data_dir, 'test_data_rna.pickle')

        self.args = args

        if not os.path.exists(args.train_data_rna) or \
                not os.path.exists(args.test_data_rna) or \
                args.generate_input_data:

            self._download_default_dataset(args.default_data_url)

            obj = DataEncoding(args.vocab_dir, args.cancer_id,
                               args.sample_id, args.target_id, args.drug_id)
            train_drug, test_drug = obj.Getdata.ByCancer(
                random_seed=args.rng_seed)

            train_drug, train_rna, test_drug, test_rna = obj.encode(
                traindata=train_drug,
                testdata=test_drug,
                args=args)

            print('Train Drug:')
            print(train_drug)
            print('Train RNA:')
            print(train_rna)

            if args.save_data:
                self.save_data(train_drug, test_drug, train_rna, test_rna)
        else:
            train_drug = pickle.load(open(args.train_data_drug, 'rb'))
            test_drug = pickle.load(open(args.test_data_drug, 'rb'))
            train_rna = pickle.load(open(args.train_data_rna, 'rb'))
            test_rna = pickle.load(open(args.test_data_rna, 'rb'))
        return train_drug, test_drug, train_rna, test_rna


def initialize_parameters(default_model='DeepTTC.default'):
    # Build benchmark object
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    # default_model = os.path.join(candle_data_dir, default_model)
    common = DeepTTCCandle(file_path,
                           default_model,
                           'torch',
                           prog='deep_ttc',
                           desc='DeepTTC drug response prediction model')

    # Initialize parameters
    gParameters = candle.finalize_parameters(common)
    relative_paths = ['vocab_dir',
                      'output_dir']

    for path in relative_paths:
        gParameters[path] = os.path.join(candle_data_dir, gParameters[path])

    if 'data_dir' not in gParameters:
        gParameters['data_dir'] = candle_data_dir
    dirs_to_check = ['results', gParameters['output_dir'],
                     gParameters['data_dir']]
    for directory in dirs_to_check:
        path = os.path.join(candle_data_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)

    gParameters['candle_data_dir'] = candle_data_dir
    return gParameters


def get_model(args):
    net = DeepTTC(modeldir=args.output_dir, args=args)
    return net


def run(args):
    loader = DataLoader(args)
    train_drug, test_drug, train_rna, test_rna = loader.load_data()
    modeldir = args.output_dir
    modelfile = os.path.join(modeldir, args.model_name)
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    model = get_model(args)
    model.train(train_drug=train_drug, train_rna=train_rna,
                val_drug=test_drug, val_rna=test_rna)
    model.save_model()
    print("Model Saved :{}".format(modelfile))


def benchmark(args):
    model = get_model(args)
    print(sys.path)
    run_cross_benchmark(model)


def main():
    import os
    import sys
    sys.path.insert(0, os.path.abspath("."))
    sys.path.insert(0, os.path.abspath(".."))
    gParameters = initialize_parameters()
    args = candle.ArgumentStruct(**gParameters)
    print(args)

    if args.mode == 'run':
        run(args)
    elif args.mode == 'benchmark':
        benchmark(args)
    else:
        raise Exception('Unknown mode. Please check configuration file')


if __name__ == "__main__":
    main()
    # cProfile.run(main())
    try:
        torch.cuda.empty_cache()
    except AttributeError:
        pass
