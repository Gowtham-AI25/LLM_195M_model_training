import math
import torch 
import torch.nn as nn
import pytest

from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.model.Embeddings import Embeddings

def get_emb_layer():
    model_config = LLM_model_config.load_from_yaml("/workspaces/LLM_195M_model_training/llm_training_project/config/configs/model_config.yaml")
    emb_layer = Embeddings(model_config)

    return emb_layer, model_config

def test_emb_layer_shape_dtype():
    emb_layer, model_config = get_emb_layer()
    batch_size = 4
    seq_len = 1024
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len)) # batch_size= 4 and seq_len = 1024
    embeddings = emb_layer(input_ids)
    # checking embedding shape
    assert embeddings.shape == (batch_size, seq_len, model_config.emb_dim)
    # checking embedding dtype
    assert embeddings.dtype == torch.float32


def test_initalization():
    emb_layer, model_config = get_emb_layer()
    # Check that the weights are initialized correctly (normal distribution)
    weight_mean  = emb_layer.weight.mean().item()
    weight_std = emb_layer.weight.std().item()
    expected_std = 1.0 / math.sqrt(model_config.emb_dim)
    assert abs(weight_mean) < 0.1, f"Weight mean {weight_mean} is too far from 0"
    assert abs(weight_std - expected_std) < 0.1 * expected_std, f"Weight std {weight_std} deviates from expected {expected_std}"



if __name__ == "__main__":
    test_emb_layer_shape_dtype()
    print("All tests passed.")

    