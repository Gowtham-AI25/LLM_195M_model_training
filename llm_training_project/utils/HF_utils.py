import os
import torch 
from typing import Optional, List
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file

class HFUtils:
    def __init__(
        self,
        hf_token: str,
        checkpoint_repo_id: str,
        dataset_repo_id: Optional[str] = None,
        checkpoint_repo_type: str = "model",
        dataset_repo_type: str = "dataset"
    ):
        self.hf_token = hf_token
        self.checkpoint_repo_id = checkpoint_repo_id
        self.dataset_repo_id = dataset_repo_id 
        self.checkpoint_repo_type = checkpoint_repo_type
        self.dataset_repo_type = dataset_repo_type

    # --- 1. CHECKPOINT SAVE (LOCAL → HF) ---
    def save_checkpoint_to_hf(
        self,
        checkpoint_path: str,
        hf_checkpoint_path: str = "/checkpoint",
        commit_message: str = "Upload checkpoint"
    ) -> None:
        """
        Uploads a local checkpoint file to the Hugging Face Hub.
        Args:
            checkpoint_path (str): Local path to the checkpoint file.
            hf_checkpoint_path (str): Path in the HF repo to save the checkpoint.
            commit_message (str): Commit message for the upload.

        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
        
        print(f"[HFUtils] Uploading {path.name} to {self.checkpoint_repo_id}...")
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=hf_checkpoint_path,
            repo_id=self.checkpoint_repo_id,
            repo_type=self.checkpoint_repo_type,
            token=self.hf_token,
            commit_message=commit_message
        )

    # --- 2. CHECKPOINT LOAD (HF → LOCAL) ---
    def load_checkpoint_from_hf(
        self,
        hf_checkpoint_path: str = "/checkpoint",
        local_checkpoint_dir: str = "./checkpoints"
    ) -> str:
        
        """
        Downloads a checkpoint to a specific local directory.
        Args:
            hf_checkpoint_path (str): Path in the HF repo to the checkpoint file.
            local_checkpoint_dir (str): Local directory to save the downloaded checkpoint.
        Returns:
            str: Local path to the downloaded checkpoint file.
        """
        local_dir = Path(local_checkpoint_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"[HFUtils] Downloading checkpoint {hf_checkpoint_path}...")
        local_path = hf_hub_download(
            repo_id=self.checkpoint_repo_id,
            repo_type=self.checkpoint_repo_type,
            filename=hf_checkpoint_path,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False, # Important for Kaggle environments
            token=self.hf_token
        )
        return local_path

    # --- 3. DATASET LOAD (HF → LOCAL) ---
    def load_dataset_from_hf(
        self,
        hf_dataset_path: str,
        local_dataset_dir: str
    ) -> str:
        """
        Downloads a dataset chunk to a specific local directory.
        Args:
            hf_dataset_path (str): Path in the HF repo to the dataset chunk.
            local_dataset_dir (str): Local directory to save the downloaded dataset chunk.
        Returns:
            str: Local path to the downloaded dataset chunk.
        """
        if not self.dataset_repo_id:
            raise ValueError("dataset_repo_id was not provided during initialization.")

        local_dir = Path(local_dataset_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"[HFUtils] Downloading dataset chunk {hf_dataset_path}...")
        local_path = hf_hub_download(
            repo_id=self.dataset_repo_id,
            repo_type=self.dataset_repo_type,
            filename=hf_dataset_path,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=self.hf_token
        )
        return local_path