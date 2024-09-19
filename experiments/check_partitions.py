import pickle

partitions = pickle.load(open('Results/DeepTTC_drug_blind_seq_sampling_mse_reg_drug/CV_partitions.pickle','rb'))
for split in range(len(partitions)):
    train = partitions[split][0]
    val = partitions[split][1]
    test = partitions[split][2]

