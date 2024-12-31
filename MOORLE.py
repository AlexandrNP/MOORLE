import torch
from torch.nn.functional import softmax
from torch.utils import data
# This script defines a PyTorch Dataset class for handling drug response data.
# The DrugResponseDataset class is designed to work with drug response datasets,
# where each sample consists of a drug ID, a cancer ID, and a corresponding label.

# The dataset is initialized with lists of IDs, labels, and dataframes containing
# response, drug, and RNA data. The data is preprocessed and stored in TensorDicts
# for efficient access during training.

# The __len__ method returns the total number of samples in the dataset, while
# the __getitem__ method retrieves a single sample, including the drug tensor,
# cancer tensor, and label, based on the provided index.

# Define a custom dataset class for drug response data
class DrugResponseDataset(data.Dataset):
    """A PyTorch Dataset class for handling drug response data."""
    
    def __init__(self, labels, response_df, rna_df, drug_df, device, dtype=torch.float):
        """
        Initialization of the DrugResponseDataset.

        Parameters:
        - labels: List of labels corresponding to each sample.
        - response_df: DataFrame containing response data.
        - drug_df: DataFrame containing drug data.
        - rna_df: DataFrame containing RNA data.
        - device: The device (CPU or GPU) where tensors will be stored.
        - dtype: The data type for the tensors (default is torch.float).
        """
        import TensorDict
        'Initialization'
        self.labels = torch.tensor(labels, dtype=dtype, device=device)
        self.list_IDs = list_IDs
        drug_df = drug_df.set_index('DrugID')

        # Convert DrugID to integer and set as index
        drug_df['DrugID_int'] = [
            int(x.split('_')[-1]) for x in drug_df.index]
        drug_df.set_index('DrugID_int', inplace=True, drop=True)

        # Convert RNA and drug dataframes to dictionaries for tensor conversion
        rna_dict = rna_df.to_dict('index')
        drug_dict = drug_df.to_dict('index')

        # Convert RNA data to tensors and store in a TensorDict
        rna_torch_dict = {str(key): torch.tensor(
            [list(value.values())], dtype=dtype) for key, value in rna_dict.items()}
        self.rna_torch_dict = TensorDict(
            rna_torch_dict, batch_size=[1], device=device)

        # Convert drug data to tensors and store in a TensorDict
        drug_torch_dict = {str(key): torch.tensor(
            [value['drug_encoding']], dtype=torch.int) for key, value in drug_dict.items()}

        self.drug_torch_dict = TensorDict(
            drug_torch_dict, batch_size=[1], device=device)

        # Convert DrugID in response_df to integer
        response_df['DrugID'] = [
            int(str(x).split('_')[-1]) for x in response_df['DrugID']]

        # Store drug and cancer IDs as tensors
        keys = list(self.drug_torch_dict.keys())
        self.drug_ids = torch.tensor(
            response_df['DrugID'].values, device=device)
        self.cancer_ids = torch.tensor(
            response_df['CancID'].values, device=device)

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Generates one sample of data.

        Parameters:
        - index: The index of the sample to retrieve.

        Returns:
        - A tuple containing the drug tensor, cancer tensor, and label for the given index.
        """
        drug_id = self.drug_ids[index].item()
        gene_id = self.cancer_ids[index].item()

        drug_tensor = self.drug_torch_dict[str(drug_id)][0]
        gene_tensor = self.rna_torch_dict[str(gene_id)][0]
        y = self.labels[index]  

        return gene_tensor, drug_tensor, y, drug_id, gene_id

# The MOORLELoss function calculates a regularized loss for drug predictions.
# It computes the mean squared error (MSE) for each unique drug and applies
# an entropy-based regularization to encourage diverse predictions.

def MOORLELoss(predicted, drug_ids, label, device, alpha=1):
    """
    Calculate the MOORLE loss, which is a combination of mean squared error
    and an entropy-based regularization component.

    Parameters:
    - predicted: Tensor of predicted values.
    - drug_ids: Tensor of drug identifiers corresponding to each prediction.
    - label: Tensor of true labels.
    - device: The device (CPU or GPU) to perform computations on.
    - alpha: Regularization strength parameter (default is 1).

    Returns:
    - drugwise_loss_reg: The computed regularized loss value.
    """
    unique_drugs = torch.unique(drug_ids)
    unique_drugs_num = unique_drugs.size()
    step = torch.empty(unique_drugs_num, requires_grad=False)
    
    # If there are fewer than 2 predictions, return a default loss value of 1
    if (len(predicted) < 2):
        return 1
    
    mse_loss = torch.nn.MSELoss()

    for i in range(len(unique_drugs)):
        unique_id = unique_drugs[i]
        unique_id_idx = (drug_ids == unique_id).nonzero(
            as_tuple=True)[0].cuda(device)
        drug_preds = torch.gather(predicted, 0, unique_id_idx)
        drug_labels = torch.gather(label, 0, unique_id_idx)
        
        # If there are no predictions for a drug, set the step to 0
        if drug_preds.size(dim=0) < 1:
            step[i] = 0
        else:
            # Calculate the MSE for the current drug
            score = mse_loss(drug_preds, drug_labels)
            step[i] = score


    # Calculate the distribution of steps and its entropy
    step_distribution = softmax(step, dim=-1)
    entropy = torch.sum(torch.special.entr(step_distribution))

    # Calculate the maximum possible entropy
    max_entropy = torch.log(torch.Tensor([unique_drugs_num])).squeeze().data
    
    # Compute the regularization component
    regularization_component = alpha * (max_entropy - entropy)
    
    # Calculate the mean value of the steps
    mean_value = torch.mean(step)
    
    # Combine the mean value and regularization component to get the final loss
    drugwise_loss_reg = torch.add(mean_value, regularization_component).to(
        device)

    return drugwise_loss_reg 
