from llm_training_project.dataloader.DataLoader import get_dataloader
from llm_training_project.config.train_config import LLM_training_config
import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

def shard_file(tmp_path):
    """
    Creates a fake shard file with deterministic data.
    Shape: [num_sequences, seq_len]
    """
    num_sequences = 32   # must be divisible by world_size=2
    seq_len = 128

    data = torch.arange(num_sequences * seq_len).view(num_sequences, seq_len)
    shard_path = tmp_path + "/" + "shard.pt"
    torch.save(data, shard_path)

    return str(shard_path)

@pytest.fixture
def get_data_loader():
    shard_path = shard_file("llm_training_project/test/temp")
    config = LLM_training_config.load_from_yaml("llm_training_project/config/configs/train_config.yaml")
    dataloader = get_dataloader(
        shard_file=shard_path,
        batch_size=config.batch_size,
        world_size =config.num_devices,
        rank= 0,
        num_workers=config.num_workers
    )
    return dataloader


def test_dataloader_is_iterable(get_data_loader):
    dataloader = get_data_loader
    it = iter(dataloader)
    batch = next(it)

    assert isinstance(batch, torch.Tensor)

def test_dataloader_batch_shape(get_data_loader):
    dataloader = get_data_loader
    batch = next(iter(dataloader))

    assert batch.shape[0] == dataloader.batch_size
    assert batch.dim() == 2


def test_dataloader_dtype(get_data_loader):
    dataloader = get_data_loader
    batch = next(iter(dataloader))

    assert batch.dtype == torch.long

def test_dataloader_no_empty_batches(get_data_loader):
    dataloader = get_data_loader

    for batch in dataloader:
        assert batch.numel() > 0

def test_dataloader_drop_last(get_data_loader):
    dataloader = get_data_loader

    total_batches = len(list(dataloader))
    total_samples = total_batches * dataloader.batch_size

    assert total_samples <= len(dataloader.dataset)

def test_dataloader_multiple_iterations(get_data_loader):
    dataloader = get_data_loader

    run1 = [batch.clone() for batch in dataloader]
    run2 = [batch.clone() for batch in dataloader]

    assert sum(b.shape[0] for b in run1) == sum(b.shape[0] for b in run2)


def test_dataloader_device_cpu(get_data_loader):
    dataloader = get_data_loader

    for batch in dataloader:
        assert batch.device.type == "cpu"


def test_dataloader_dataset_length(get_data_loader):
    dataloader = get_data_loader

    assert len(dataloader.dataset) > 0


def test_dataloader_uses_distributed_sampler(get_data_loader):
    dataloader = get_data_loader

    if dataloader.sampler is not None:
        assert isinstance(dataloader.sampler, DistributedSampler)
