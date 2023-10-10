import huggingface_hub as hf_hub
import numpy as np
import os
import torch
import torchmetrics
import wandb
import yaml

def map_classes(y, class_mapping, boolean=False):
    """
    Takes class_mapping, a list which indicates the position of each class in
    the input y to the output y_prime. e.g...

    class_mapping = [0, 0, 0, 1, 1, 2, 2]

    would produce a y_prime with final dimension of size 3.

    If an input dimension has no target, use np.nan in that position.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor of shape (:, n_classes, ...)
    class_mapping : list
        List of length n_classes, indicating the position of each class in the
        output tensor.
    boolean : bool, optional
        If True, the input will be treated as a boolean mask, meaning that the 
        values are not summed, but instead the maximum value is taken. This is
        useful for labels. The default is False, for when the input is a 
        probability/confidence distribution.
    """

    #Get all the target_classes (excluding nans).
    target_classes = np.unique(class_mapping)
    target_classes = target_classes[~np.isnan(target_classes)]
    tensors = []
    for target in target_classes:
        # Find all input classes that correspond to the target
        base_positions = [i for i,c in enumerate(class_mapping) if c==target]

        # stack all input classes corresponding to the target
        gathered_slices = y[:, base_positions]

        if boolean:
            class_tensor,_ = torch.max(gathered_slices,dim=1)
        else:
            class_tensor = torch.sum(gathered_slices,dim=1)

        tensors.append(class_tensor)
    return torch.stack(tensors, dim=1)


def ambiguous_mse_loss(y_pred, y_true):
    """
    Does not expect one-hot encoded labels. Instead, multiple labels are marked as possible.

    This allows the model to learn specific labels when possible, whilst also learning something
    when multiple possibilities exist.

    When multiple possible labels exist, the softmax outputs for those labels are replaced by
    their mean value. For example, for a softmax output, p, and a label vector, y, such as:

    p = [0.1 , 0.6 , 0.2 , 0.1 , 0.0]
    y = [  1 ,   1 ,   0 ,   0 ,   0]

    First calculate the mean of values p_i for which y_i==1:

    (0.1 + 0.6) / 2 = 0.7 / 2 = 0.35

    Replace values accordingly:

    p = [0.35 , 0.35 , 0.2 , 0.1 , 0.0]


    Then compute mse loss for:

    p = [0.35, 0.35, 0.2 , 0.1 , 0.0]
    y = [ 0.5,  0.5,   0 ,   0 ,   0]


    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor of shape (:, n_classes, ...)
    y_true : torch.Tensor
        Tensor of shape (:, n_classes, ...)
    """
    # return torch.nn.functional.mse_loss(y_pred, y_true)
    N = y_true.shape[1]
    dims = len(y_true.shape)

    # Get summed possible probabilities
    proxy_pred_vals = torch.sum(y_true * y_pred, axis=1, keepdims=True)

    # Get number of positive classes per pixel, to weight loss on possible classes
    num_positives = torch.sum(y_true, axis=1, keepdims=True)

    # New predictions are: proxy value in all possible labels, and original values in impossible labels
    new_preds = torch.tile(proxy_pred_vals / num_positives,[1,N]+[1]*(dims-2)) * y_true + y_pred * (1 - y_true) #'pi' in paper

    l = torch.nn.functional.mse_loss(
            new_preds, y_true / (num_positives + torch.finfo(torch.float32).eps)
            )

    return l


def ambiguous_crossentropy_loss(y_pred, y_true):
    """
    Does not expect one-hot encoded labels. Instead, multiple labels are marked as possible.

    This allows the model to learn specific labels when possible, whilst also learning something
    when multiple possibilities exist.

    When multiple possible labels exist, the softmax outputs for those labels are replaced by
    their mean value. For example, for a softmax output, p, and a label vector, y, such as:

    p = [0.1 , 0.6 , 0.2 , 0.1 , 0.0]
    y = [  1 ,   1 ,   0 ,   0 ,   0]

    First calculate the sum of values p_i for which y_i==1:

    0.1 + 0.6 = 0.7

    Replace values accordingly:

    p = [0.7 , 0.2 , 0.1 , 0.0]


    Then compute crossentryopy loss for:

    p = [ 0.7, 0.35, 0.2 , 0.1 , 0.0]
    y = [   1,    0,   0 ,   0 ,   0]


    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor of shape (:, n_classes, ...)
    y_true : torch.Tensor
        Tensor of shape (:, n_classes, ...)
    """

    # Get summed possible probabilities
    possibility_score = torch.sum(y_true * y_pred, axis=1, keepdims=True) # 'phi' in paper

    # positive entropy term
    l = -torch.log2(possibility_score + torch.finfo(torch.float32).eps)

    return torch.mean(l)

class InformationMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("truth_info", default=torch.tensor(0,dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("pred_info", default=torch.tensor(0,dtype=torch.float), dist_reduce_fx="sum")

    def update(self, y_pred, y_true):
        true_probs = y_true/torch.sum(y_true,axis=1,keepdims=True)
        truth_info = torch.mean(torch.sum(true_probs**2,axis=1))
        pred_info = torch.mean(torch.sum(true_probs*y_pred,axis=1))

        self.truth_info += truth_info
        self.pred_info += pred_info

    def compute(self):
        return torch.divide(self.pred_info,self.truth_info)

class Entropy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("entropy", default=torch.tensor(0,dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("N", default=torch.tensor(0,dtype=torch.float), dist_reduce_fx="sum")

    def update(self, y_pred, y_true):
        pred_pos = torch.sum(y_true*y_pred,axis=1)
        pred_neg = torch.sum((1-y_true)*y_pred,axis=1)

        entropy = -pred_pos*torch.log2(pred_pos+torch.finfo(torch.float32).eps) - pred_neg*torch.log2(pred_neg+torch.finfo(torch.float32).eps)
        self.entropy += torch.mean(entropy)
        self.N += 1

    def compute(self):
        return self.entropy/self.N

class MappedClassMetric(torchmetrics.Metric):

    def __init__(self, metric, class_mapping, class_names=None, argmax_preds=False, argmax_labels=False):
        super().__init__()
        self.metric = metric
        self.class_mapping = class_mapping
        self.class_names = class_names
        self.argmax_labels = argmax_labels
        self.argmax_preds = argmax_preds

    def update(self, y_pred, y_true):

        y_pred = torch.moveaxis(y_pred,1,-1)
        y_true = torch.moveaxis(y_true,1,-1)

        y_pred = torch.reshape(y_pred,[-1,y_pred.shape[-1]])
        y_true = torch.reshape(y_true,[-1,y_true.shape[-1]])

        y_pred = map_classes(y_pred,self.class_mapping,boolean=False)
        y_true = map_classes(y_true,self.class_mapping,boolean=True)

        if self.argmax_preds:
            y_pred = torch.argmax(y_pred,axis=1)
        if self.argmax_labels:
            y_true = torch.argmax(y_true,axis=1)

        self.metric.update(y_pred, y_true)

    def compute(self):

        # Hacky way of getting round wandb's confusion matrix implementation.
        if isinstance(self.metric,torchmetrics.classification.MulticlassConfusionMatrix):
            conf_mat = self.metric.compute()
            return convert_confusion_matrix(conf_mat,self.class_names)
        else:
            return self.metric.compute()

    def reset(self):
        self.metric.reset()

def convert_confusion_matrix(conf_mat,class_names):
    # wandb wants labels and predictions and computes everything internally,
    # but its too much data for the whole validation/training set, so we
    # compute the confusion matrix ourselves (batchwise) and then convert it 
    # to the format wandb wants.
    n_classes = len(class_names)

    # Adapted from: https://github.com/wandb/wandb/blob/main/wandb/plot/confusion_matrix.py
    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([class_names[i], class_names[j], conf_mat[i, j]])

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
        fields,
        {"title": 'Confusion Matrix'},
    )


def get_model_files(model_name):
    path = os.path.join(os.path.dirname(__file__), '..', 'hf_models')
    path = os.path.abspath(path)
    for f in ['config.yaml','weights.pt']:
        if not os.path.exists(os.path.join(path,'full-models',model_name,f)):
            print(f'Downloading {f} for {model_name} from https://huggingface.co/aliFrancis/SEnSeIv2')
            hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename=f,subfolder=f'full-models/{model_name}', local_dir=path)
    
    config_path = os.path.join(path,'full-models',model_name,'config.yaml')
    weights_path = os.path.join(path,'full-models',model_name,'weights.pt')

    config = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    if not os.path.exists(config['SEnSeIv2']):
        sensei_version = config['SEnSeIv2'].split('/')[-1]
        if not os.path.exists(os.path.join(path,'sensei-configs',sensei_version)):
            print(f'Downloading {sensei_version} from https://huggingface.co/aliFrancis/SEnSeIv2')
            hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename=sensei_version,subfolder=f'sensei-configs', local_dir=path)
    return config_path, weights_path