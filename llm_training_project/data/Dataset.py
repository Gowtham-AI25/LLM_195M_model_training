import torch 
from torch.utils.data import Dataset
from typing import Optional 

class SingleShardDataset(Dataset):
    """
    Map-style Dataset exposing exactly ONE shard (.pt file).

    Responsibilities:
    - Load tensor from disk
    - Provide __len__ and __getitem__

    """
    def __init__(self, shard_file: str, validate: bool = True):
        """
        Args:
            shard_file (str): Path to the .pt file containing the tensor data.
            device (str): Device to load the tensor onto (e.g., "cpu", "cuda").
            validate (bool): Whether to validate the loaded tensor.
        """
        self.shard_file = shard_file
        # Load the tensor from the .pt file
        self.data = self._load_shard(self.shard_file)

        if validate:
            self._validate_data()
    
    def _validate_data(self):
        """
            Validate the loaded tensor data(type of data, dtype of data and dim of data).
            Raises ValueError if validation fails.
        """

        if not isinstance(self.data, torch.Tensor):
            raise ValueError(f"Loaded data is not a torch.Tensor, got {type(self.data)} instead.")
        
        if self.data.dtype != torch.long:
            raise ValueError(f"Loaded data does not have a valid torch.dtype, got {self.data.dtype} instead.")
        
        if self.data.dim() != 2:
            raise ValueError(f"Loaded data does not have 2 dimensions [ Num_Seq, Seq_len ], got {self.data.dim()} instead.")
    
    def _load_shard(self, shard_file: str) -> torch.Tensor:
        """
            Load the tensor from the .pt file and move it to the specified device.
            Args:
                shard_file (str): Path to the .pt file.
            Returns:
                torch.Tensor: Loaded tensor on the specified device.
        """

        tensor = torch.load(shard_file, map_location="cpu")
        return tensor
    
    def __len__(self) -> int:
        """
        Returns:
            int: Number of sequences in the dataset.
        """
        return self.data.size(0)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx (int): Index of the sequence to retrieve.
        
        Returns:
            torch.Tensor: The sequence at the specified index.
        """
        return self.data[idx]   
        


