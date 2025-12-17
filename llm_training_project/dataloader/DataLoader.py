import torch 
from torch.utils.data import DataLoader, DistributedSampler
from llm_training_project.dataloader.Dataset import SingleShardDataset
from typing import Optional

def get_dataloader(
        shard_file: str,
        batch_size: int,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        num_workers: int = 0,
        validate: bool = True ) -> DataLoader:
    
    # Create the SingleShardDataset instance for the given shard file
    dataset = SingleShardDataset(
                shard_file=shard_file,
                validate=validate
            )
    
    # Create DistributedSampler if world_size and rank are provided this is for distributed training to ensure each process gets a different subset of data
    sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True
            ) if world_size is not None and rank is not None else None
    
    # Create the DataLoader with the dataset and sampler
    dataloader = DataLoader(
                    dataset,
                    batch_size = batch_size,
                    sampler = sampler,
                    shuffle = True if sampler is None else False,
                    num_workers = num_workers,
                    pin_memory = True,
                    drop_last = True
                )
    return dataloader
