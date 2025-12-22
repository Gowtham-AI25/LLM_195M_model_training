from llm_training_project.shards.ShardManager import ShardManager
import pytest
import json


def shard_json(tmp_path):
    """
    Creates a temporary shard JSON file for testing.
    """
    shard_data = {
        "remaining_shards": [
            "shard_001.pt",
            "shard_002.pt",
            "shard_003.pt"
        ]
    }

    json_path = tmp_path + "/" + "shards.json"
    with open(json_path, "w") as f:
        json.dump(shard_data, f, indent=4)

    return str(json_path)
@pytest.fixture
def get_shard_manager():
   json_path = shard_json("llm_training_project/test/temp")
   shard_manager = ShardManager(json_path)
   return shard_manager

def test_load_shard_files(get_shard_manager):
    shard_manager = get_shard_manager
    assert shard_manager.shard_files == ["shard_001.pt", "shard_002.pt", "shard_003.pt"]

def test_has_next(get_shard_manager):
    shard_manager = get_shard_manager
    # Initially, there are shards
    assert shard_manager.has_next() is True
    # Remove all shards
    shard_manager.remove_shard()  
    shard_manager.remove_shard()
    shard_manager.remove_shard()
    # Now, there should be no shards left
    assert shard_manager.has_next() is False 

def test_get_next_shard(get_shard_manager):
   shard_manager = get_shard_manager
   next_shard = shard_manager.get_next_shard()
   assert next_shard == "shard_001.pt"
   # assert that the shard list is unchanged
   assert len(shard_manager.shard_files) == 3

def test_remove_shard(get_shard_manager):
    shard_manager = get_shard_manager
    removed_shard = shard_manager.remove_shard()
    assert removed_shard == "shard_001.pt"
    assert shard_manager.shard_files == ["shard_002.pt", "shard_003.pt"]
    # Remove another shard
    removed_shard = shard_manager.remove_shard()
    assert removed_shard == "shard_002.pt"
    assert shard_manager.shard_files == ["shard_003.pt"]
