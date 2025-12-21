from llm_training_project.model.model import LLM 
from llm_training_project.config.model_config import LLM_model_config
import torch
import pytest

def get_model():
    model_config = LLM_model_config.load_from_yaml("/workspaces/LLM_195M_model_training/llm_training_project/config/configs/model_config.yaml")
    model = LLM(model_config)
    return model, model_config

@pytest.mark.parametrize("batch_size, seq_len", [
    (1, 1),       # Smallest possible input (Single token inference)
    (1, 128),     # Standard single-batch prompt
    (4, 128),      # Small multi-batch
    (8, 128),     # Medium batch/sequence
])

def test_model_initialization(batch_size, seq_len):
    model, model_config = get_model()
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        outputs = model(input_ids)
    assert outputs.shape == (batch_size, seq_len, model_config.vocab_size), f"Output shape {outputs.shape} does not match expected {(batch_size, seq_len, model_config.vocab_size)}"

