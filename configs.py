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


def load_tokenizer(name, config_name_or_dict):
    """
    Loads transformer tokenizer
    """
    tokenizer_dict = {'bert': BertTokenizer,
                      'roberta': RobertaTokenizer,
                      'albert': AlbertTokenizer,
                      'distilbert': DistilBertTokenizer}

    if type(config_name_or_dict) == dict:
        tokenizer = tokenizer_dict[name].from_dict(config_name_or_dict)

    elif type(config_name_or_dict) == str:
        tokenizer = tokenizer_dict[name].from_pretrained(config_name_or_dict)

    else:
        print(type(config_name_or_dict))
        raise TypeError("The configs param can either be of str or dict dtype!")

    return tokenizer


def load_config(name, config_name_or_dict):
    """
    Loads transformer config

    :param str name: transformer model name (e.g bert, roberta, etc)
    :param config_name_or_dict: either pre-trained config name or custom params dict
    :type config_name_or_dict: str or dict
    :return: config object
    """
    configs_dict = {'bert': BertConfig,
                    'roberta': RobertaConfig,
                    'albert': AlbertConfig,
                    'distilbert': DistilBertConfig}

    if type(config_name_or_dict) == dict:
        config = configs_dict[name](**config_name_or_dict)

    elif type(config_name_or_dict) == str:
        config = configs_dict[name].from_pretrained(config_name_or_dict)

    else:
        raise TypeError("The configs param can either be of str or dict dtype!")

    return config


def load_embedding_fn(name, config_name_or_dict):
    """
    Load the Embedding function
    (Used for mapping token IDs --> Embeddings)

    The Embedding function internally requires
    the following config params:

    `vocab_size, hidden_size, type_vocab_size, max_position_embeddings,
    layer_norm_eps, hidden_dropout_prob`

    :param str name: model name (e.g. bert, roberta)
    :param config_name_or_dict: config pre-trained config name or custom params
    :type config_name_or_dict: str or dict
    """
    embedding_fn_dict = {'bert': BertEmbeddings,
                         'roberta': RobertaEmbeddings,
                         'albert': AlbertEmbeddings,
                         'distilbert': DistillBertEmbeddings}

    # Load the config object
    config = load_config(name, config_name_or_dict)

    # Initialize the embedding function using the config
    embedding_fn = embedding_fn_dict[name](config)

    return embedding_fn


def load_model(name, config_dict=None, config_name='bert-base-uncased', use_pretrained=False):
    """
    Loads transformer model

    :param str name: transformer model name
    :param dict config_dict: custom config params dict
    :param str config_name: pre-trained model config name
    :param bool use_pretrained: flag for using pre-trained transformer
    :return: transformer model from HuggingFace
    """
    model_dict = {'bert': BertModel,
                  'roberta': RobertaModel,
                  'albert': AlbertModel,
                  'distilbert': DistilBertModel}

    if use_pretrained:
        if config_dict:
            # Load BertConfig from the custom configs dict
            config = load_config(name, config_dict)

            model = model_dict[name].from_pretrained(config_name, config=config)

        else:
            model = model_dict[name].from_pretrained(config_name)

    else:
        model = model_dict[name](config_dict)

    return model


# TODO:: Add Common ArgParse options
