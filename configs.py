"""
Configs for Models & Tokenizers
Configs for Arguments (argparse options)
"""
import torch
import torch.nn as nn
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, DistilBertTokenizer
from transformers import BertConfig, RobertaConfig, AlbertConfig, DistilBertConfig
from transformers import BertModel, RobertaModel, AlbertModel, DistilBertModel
from torchvision.models import resnet18, resnet152, densenet161
from transformers.modeling_bert import BertEmbeddings
from transformers.modeling_roberta import RobertaEmbeddings
from transformers.modeling_albert import AlbertEmbeddings
from transformers.modeling_distilbert import Embeddings as DistillBertEmbeddings

MODEL_NAMES = ['bert', 'roberta', 'albert', 'distilbert']
CNN_NAMES = ['resnet18', 'resnet152', 'densenet161']


class IdentityLayer(nn.Module):
    """
    Identity: output is same as input.

    Helpful in 'deleting' layers from pre-defined models.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def remove_fc_from_state_dict(state_dict, cnn_model_name="resnet18"):
    if 'resnet' in cnn_model_name:
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
    elif 'densenet' in cnn_model_name:
        del state_dict["classifier.weight"]
        del state_dict["classifier.bias"]
    else:
        pass 
    return state_dict


def load_cnn(name, is_pretrained=True, num_cls=-1):
    """
    Given CNN model name, loads model (optional: pretrained).

    The pretrained weights (as provided by torchvision.models)
    are loaded from TORCH_HOME directory (~/.cache/torch)

    Removes the final FC layers of the model to
    use it for feature extraction.

    :param str name: CNN model name (from torchvision.models)
    :param bool is_pretrained: flag to load pre-trained weights
    :param int num_cls: no. of classes in the final FC layer.
                        If num_cls == -1: don't append an fc layer.
    :returns: CNN model, embedding dim of final FC layer
    """
    cnn_dict = {'resnet18': resnet18,
                'resnet152': resnet152,
                'densenet161': densenet161}

    # If no checkpoint provided, load pre-trained weights (torchvision.models)
    model = cnn_dict[name](is_pretrained)

    # If num_cls is provided, then we replace with a new FC layer; else delete existing FC layer
    is_new_fc = True if num_cls != -1 else False

    # Also compute the embedding dim of the final FC layer
    emb_dim = None

    if 'resnet' in name:
        emb_dim = model.fc.in_features

        # Either replace with new FC layer
        if is_new_fc:
            model.fc = nn.Linear(emb_dim, num_cls)

        # Or delete the current FC layer
        else:
            model.fc = IdentityLayer()

    # elif 'densenet' in name:
    else:
        emb_dim = model.classifier.in_features

        # Either replace with new FC layer
        if is_new_fc:
            model.classifier = nn.Linear(emb_dim, num_cls)

        # Or delete the current FC layer
        else:
            model.classifier = IdentityLayer()

    return model, emb_dim


# TODO:: Add Common Argparse options
