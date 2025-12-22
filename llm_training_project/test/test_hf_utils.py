from llm_training_project.utils.HF_utils import HFUtils
import pytest 
import os

@pytest.fixture
def get_hf_utils():
   checkpoint_path = "llm_training_project/checkpoints_dir/checkpoints/checkpoint.pt"
   dataset_dir = "llm_training_project/shards/dataset_dir"
   checkpoint_dir = "llm_training_project/checkpoints_dir"

   hf_utils = HFUtils.load_config_from_yaml(
       "llm_training_project/config/configs/hf_config.yaml")
   return hf_utils, checkpoint_path, dataset_dir, checkpoint_dir

def test_load_checkpoint_from_hf(get_hf_utils):
    hf_utils, _, _, checkpoint_dir = get_hf_utils

    local_path = hf_utils.load_checkpoint_from_hf(
        local_checkpoint_dir=checkpoint_dir
    )
    assert isinstance(local_path, str), "Returned path is not a string"
    assert local_path.endswith("checkpoint.pt"), "Incorrect checkpoint filename"
    assert os.path.isfile(local_path), "Checkpoint file was not downloaded"

def test_save_checkpoint_to_hf(get_hf_utils):
    hf_utils, checkpoint_path, _, _ = get_hf_utils

    try:
        hf_utils.save_checkpoint_to_hf(
            checkpoint_path=checkpoint_path,
            commit_message="Test upload checkpoint"
        )
    except Exception as e:
        pytest.fail(f"save_checkpoint_to_hf raised an exception: {e}")


def test_download_hf_file_from_url(get_hf_utils):
    hf_utils, _, dataset_dir, _ = get_hf_utils

    file_url = f"https://huggingface.co/datasets/gowthamgoli/LLM_3B_tokens/resolve/main/tokens/batch_0000.pt"

    local_path = hf_utils.download_hf_file_from_url(
        file_url=file_url,
        local_datasetdir=dataset_dir
    )
    print(local_path)
    assert isinstance(local_path, str), "Returned path is not a string"
    assert local_path.endswith("batch_0000.pt"), "Incorrect filename"
    assert os.path.isfile(local_path), "File was not downloaded"