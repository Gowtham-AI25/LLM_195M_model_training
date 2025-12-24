import yaml
from typing import Optional
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file
from urllib.parse import urlparse
import shutil
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

class HFUtils:
    def __init__(
        self,
        hf_token: str,
        checkpoint_repo_id: str,
        dataset_repo_id: Optional[str] = None,
        checkpoint_repo_type: str = "model",
        dataset_repo_type: str = "dataset",
        hf_checkpoint_dir: str = "checkpoints", # Removed leading slash for better path joining
        hf_dataset_dir: str = "tokens"
    ):
        self.hf_token = hf_token
        self.checkpoint_repo_id = checkpoint_repo_id
        self.dataset_repo_id = dataset_repo_id 
        self.checkpoint_repo_type = checkpoint_repo_type
        self.dataset_repo_type = dataset_repo_type
        self.hf_checkpoint_dir = hf_checkpoint_dir
        self.hf_dataset_dir = hf_dataset_dir
        
    @classmethod
    def load_config_from_yaml(cls, yaml_path: str) -> "HFUtils":
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    # --- 1. CHECKPOINT SAVE (LOCAL → HF) ---
    def save_checkpoint_to_hf(
            self,
            checkpoint_path: str,
            commit_message: str = "Upload checkpoint"
        ) -> None:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint file not found at {path}")
            
            
            print(f"[HFUtils] Uploading file to {self.checkpoint_repo_id}")
            
            upload_file(
                path_or_fileobj=str(path),
                path_in_repo=self.hf_checkpoint_dir,
                repo_id=self.checkpoint_repo_id,
                repo_type=self.checkpoint_repo_type,
                token=self.hf_token,
                commit_message=commit_message,
            )
            
            print(f"[HFUtils] ✅ Successfully uploaded {path.name} to Hugging Face.")


    def load_checkpoint_from_hf(
        self,
        checkpoint_dir: str,
    ) -> str | None:
        """
        Download checkpoint from Hugging Face (via HF cache)
        and place it into the canonical checkpoint directory.
    
        Returns:
            Path to checkpoint.pt if found, otherwise None.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
        final_ckpt_path = checkpoint_dir / "checkpoint.pt"
        repo_file_path = self.hf_checkpoint_dir.strip("/")
    
        print(f"[HFUtils] Checking checkpoint '{repo_file_path}' on Hugging Face...")
        try:
            cached_ckpt_path = hf_hub_download(
                repo_id=self.checkpoint_repo_id,
                filename=repo_file_path,
                repo_type=self.checkpoint_repo_type,
                token=self.hf_token,
                force_download=False,
            )
    
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
            print("[HFUtils] No checkpoint found on Hugging Face. Starting fresh.")
            return None
    
        except Exception as e:
            print(f"[HFUtils] Unexpected error while downloading checkpoint: {e}")
            return None
    
        if not final_ckpt_path.exists():
            print("[HFUtils] Copying checkpoint into training directory...")
            shutil.copy2(cached_ckpt_path, final_ckpt_path)
    
        print(f"[HFUtils] Checkpoint ready at: {final_ckpt_path}")
    
        return str(final_ckpt_path)
        
    # --- 3. DATASET LOAD (HF → LOCAL) ---

    def download_hf_file_from_url(
        self,
        file_url: str,
        local_datasetdir: str
    ) -> str:
        """
        Download a single Hugging Face file using a direct `resolve/main/...` URL.
        Args:
            file_url (str): The full URL to the file on Hugging Face.
            local_dir (str): The local directory to save the downloaded file.
        Returns:
            str: The local path to the downloaded file.
        """

        # -------------------------------
        # 1. Prepare local directory
        # -------------------------------
        local_dir = Path(local_datasetdir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------
        # 2. Parse URL
        # -------------------------------
        parsed = urlparse(file_url)
        parts = parsed.path.strip("/").split("/")

        # Expected structure:
        # datasets/{repo_id}/resolve/{revision}/{file_path}
        if "resolve" not in parts:
            raise ValueError("Invalid Hugging Face resolve URL")

        resolve_idx = parts.index("resolve")
        repo_id = "/".join(parts[1:resolve_idx])   # gowthamgoli/LLM_3B_tokens
        filename = "/".join(parts[resolve_idx + 2:])  # tokens/batch_0000.pt

        # -------------------------------
        # 3. Download file
        # -------------------------------
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=self.dataset_repo_type,
            local_dir=str(local_dir),
            force_download=False
        )

        return str(local_path)
