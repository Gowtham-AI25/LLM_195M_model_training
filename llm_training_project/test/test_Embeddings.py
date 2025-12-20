import math
import torch 
import torch.nn as nn
import pytest

from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.model.Embeddings import Embeddings

def get_emb_layer():
    model_config = LLM_model_config.load_from_yaml("/workspaces/LLM_195M_model_training/llm_training_project/config/configs/model_config.yaml")
    emb_layer = Embeddings(model_config)

    return emb_layer

def test_emb_layer():
    