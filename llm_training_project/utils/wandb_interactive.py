import torch
import torch.distributed as dist
from huggingface_hub import HfApi

# Initialize the API once outside the function
hf_api_client = HfApi()

def should_stop_from_hf(rank, world_size, device):
    """
    Checks for stop signal using the official Hugging Face Hub API.
    - File exists -> Continue
    - File missing -> Stop
    """
    REPO_ID = "gowthamgoli/llm_test_datarepo"
    FILENAME = "STOP_TRAINING"
    
    # 0 = Continue, 1 = Stop
    stop_flag = torch.tensor(0, device=device)

    if rank == 0:
        try:
            # Native check for file existence
            exists = hf_api_client.file_exists(
                repo_id=REPO_ID,
                filename=FILENAME,
                repo_type="dataset"
            )

            if not exists:
                print(f"\n[CONTROL] 🛑 {FILENAME} not found in {REPO_ID}. Stopping training.", flush=True)
                stop_flag.fill_(1)
            
        except Exception as e:
            # If the API call fails (e.g., timeout or network), default to continuing
            # so we don't accidentally kill a long training run.
            pass 

    # Broadcast decision to all ranks
    if world_size > 1:
        dist.broadcast(stop_flag, src=0)

    return bool(stop_flag.item())
