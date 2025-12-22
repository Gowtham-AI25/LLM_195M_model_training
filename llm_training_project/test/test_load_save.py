
import torch
import torch.distributed as dist
import os
import shutil
import pytest
import math
from pathlib import Path
import gc

# --- Internal Imports ---
from llm_training_project.model.model import LLM 
from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.config.train_config import LLM_training_config
from llm_training_project.training.TraningStateManager import TrainingStateManager
from llm_training_project.checkpoints.checkpoint import CheckpointManager


@pytest.fixture
def test_setup():
    device = torch.device("cpu")
    model_config = LLM_model_config.load_from_yaml("llm_training_project/config/configs/model_config.yaml")
    train_config = LLM_training_config.load_from_yaml("llm_training_project/config/configs/train_config.yaml")
    ckpt_manager = CheckpointManager(train_config.checkpoint_path, device)
    state_manager = TrainingStateManager(train_config, model_config, ckpt_manager, device)

    return state_manager, ckpt_manager, train_config.checkpoint_path

@pytest.fixture
def get_state(test_setup):
    state_manager, ckpt_manager, ckpt_path = test_setup
    state  = state_manager.load_training_state(LLM)
    return state, ckpt_manager

def test_load_training_state(get_state):
    state, _ = get_state
    assert isinstance(state['model'], torch.nn.Module)
    assert isinstance(state['optimizer'], torch.optim.AdamW)

def test_checkpoint_saving(get_state):
    state, ckpt_manager = get_state
    # Simulate saving a checkpoint
    path = ckpt_manager.save_checkpoint(**state, name="checkpoint")
    # Check if checkpoint was saved
    assert os.path.exists(path)

if __name__ == "__main__":
    pytest.main([__file__])
    gc.collect()

