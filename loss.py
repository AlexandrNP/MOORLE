import torch
from torch.nn.functional import softmax


def MOORLELoss(predicted, drug_ids, label, device, alpha=1):
    unique_drugs = torch.unique(drug_ids)
    unique_drugs_num = unique_drugs.size()
    step = torch.empty(unique_drugs_num, requires_grad=False)
    if (len(predicted) < 2):
        return 1
    mse_loss = torch.nn.MSELoss()

    for i in range(len(unique_drugs)):
        unique_id = unique_drugs[i]
        unique_id_idx = (drug_ids == unique_id).nonzero(
            as_tuple=True)[0].cuda(device)
        drug_preds = torch.gather(predicted, 0, unique_id_idx)
        drug_labels = torch.gather(label, 0, unique_id_idx)
        if drug_preds.size(dim=0) < 1:
            step[i] = 0
        else:
            score = mse_loss(drug_preds, drug_labels)
            step[i] = score

    step_distribution = softmax(step, dim=-1)
    entropy = torch.sum(torch.special.entr(step_distribution))

    max_entropy = torch.log(torch.Tensor([unique_drugs_num])).squeeze().data
    regularization_component = alpha * (max_entropy - entropy)
    mean_value = torch.mean(step)
    drugwise_loss_reg = torch.add(mean_value, regularization_component).to(
        device)

    return drugwise_loss_reg
