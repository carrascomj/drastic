# Different analysis functions to evaluate the perfomance of a model

import torch
import numpy as np


def binary_accuracy(preds, y):
    """ 
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.zeros(preds.size())
    for i in range(len(preds)):
        idx_max = torch.where(preds[i] == preds[i].max())
        rounded_preds[i][idx_max] = 1
    rounded_preds = torch.FloatTensor(rounded_preds).int()
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = (correct.sum() / len(preds[0])) / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    """Evaluation loop.

    Parameters
    ----------
    model: torch.Module
        trained neural network
    iterator torch.utils.data.DataLoader:
        dynamic data loader object
    criterion nn.criterion:
        criterion used to train the model
    
    Return
    ------
    loss : float
    accuracy: float
    predictions_all: list
        each prediction of the model
    labels_all: list
        each real label of the model (index). Just for convenience, but it
        could be extracted from the model
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    print("Evaluating...")
    predictions_all = []
    labels_all = []

    for i, batch in enumerate(iterator):

        inputs, labels_onehot = batch

        inputs = Parameter(inputs.float(), requires_grad=False)

        predictions = model(inputs)
        labels_idx = torch.LongTensor(
            [np.where(label == 1)[0][0] for label in labels_onehot]
        )

        loss = criterion(predictions, labels_idx)
        acc = binary_accuracy(predictions, labels_onehot)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        predictions_all.append(predictions)
        labels_all.append(labels_idx)
    return (
        epoch_loss / len(iterator),  # loss
        epoch_acc / len(iterator),  # accuracy
        predictions_all,  # the predictions of the model
        labels_all,  # the real labels of the model
    )


def to_confussion_matrix(preds, y, conf={}):
    return NotImplementedError("First we need to have the trained models!")
