import os
import torch 
from typing import Optional, List
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file

class HFUtils:

    def __init__(self,
                 hf_token: str,
                 checkpoint_repo_id: str,
                 dataset_repo_id: Optional[str] = None,
                 checkpoint_repo_type: str = "model",
                 dataset_repo_type: str = "dataset"
            ):
        """
        Args:
            hf_token (str): Hugging Face authentication token.
            checkpoint_repo_id (str): Repository ID for model checkpoints.
            dataset_repo_id (Optional[str]): Repository ID for datasets.
            checkpoint_repo_type (str): Type of the checkpoint repository ("model" or "dataset").
            dataset_repo_type (str): Type of the dataset repository ("model" or "dataset").
        """
        self.hf_token = hf_token
        self.checkpoint_repo_id = checkpoint_repo_id
        self.dataset_repo_id = dataset_repo_id 
        self.checkpoint_repo_type = checkpoint_repo_type
        self.dataset_repo_type = dataset_repo_type

        # 1. CHECKPOINT SAVE (LOCAL → HF)

        def save_checkpoint_to_hf(self,
                                  checkpoint_path: str,
                                  hf_checkpoint_path: str,
                                  commit_message: str = "Upload checkpoint"
            ) -> None:
            """
            Uploads a local checkpoint file to the Hugging Face Hub.

            Args:
                local_checkpoint_path (str): Path to the local checkpoint file.
                hf_checkpoint_path (str): Path in the HF repo where the file will be saved.
                commit_message (str): Commit message for the upload.
            """
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
            
            upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=hf_checkpoint_path,
                repo_id=self.checkpoint_repo_id,
                repo_type=self.checkpoint_repo_type,
                token=self.hf_token,
                commit_message=commit_message
            )
        
        # 2. CHECKPOINT LOAD (HF → LOCAL)

        def load_checkpoint_from_hf(self,
                                    hf_checkpoint_path: str,
                                    local_checkpoint_path: str
            ) -> str:
            """
            Downloads a checkpoint file from the Hugging Face Hub to a local path.

            Args:
                hf_checkpoint_path (str): Path in the HF repo where the file is located.
                local_checkpoint_path (str): Local path where the file will be saved.
            Returns:
                str: Path to the downloaded local checkpoint file.
            """
            local_checkpoint_path = Path(local_checkpoint_path)
            local_checkpoint_path.mkdir(parents=True, exist_ok=True)

            local_path = hf_hub_download(
                repo_id=self.checkpoint_repo_id,
                repo_type=self.checkpoint_repo_type,
                filename=hf_checkpoint_path,
                local_dir=str(local_checkpoint_path.parent),
                local_dir_use_symlinks=False,
                token=self.hf_token
            )
            return local_path
    
        # 3. DATASET LOAD (HF → LOCAL)

        def load_dataset_from_hf(self,
                                 hf_dataset_path: str,
                                 local_dataset_path: str
            ) -> str:
            """
            Downloads a dataset file from the Hugging Face Hub to a local path.

            Args:
                hf_dataset_path (str): Path in the HF repo where the dataset file is located.
                local_dataset_path (str): Local path where the dataset file will be saved.
            """
            local_dataset_path = Path(local_dataset_path)
            local_dataset_path.mkdir(parents=True, exist_ok=True)

            # example link https://huggingface.co/datasets/gowthamgoli/LLM_3B_tokens/resolve/main/tokens/batch_0000.pt
            local_path = hf_hub_download(
                repo_id=self.dataset_repo_id,
                repo_type=self.dataset_repo_type,
                filename=hf_dataset_path,
                local_dir=str(local_dataset_path.parent),
                local_dir_use_symlinks=False,
                token=self.hf_token
            )
            return local_path

