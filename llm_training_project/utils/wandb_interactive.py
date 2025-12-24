import torch
import torch.distributed as dist
import requests

def should_stop_from_hf(rank, world_size, device, timeout=5.0):
    """
    Checks if a control file exists on Hugging Face.
    - File exists (200 OK): Continue training.
    - File missing (404): Stop training.
    """
    # Hard-coded control URL
    CONTROL_URL = "https://huggingface.co/datasets/gowthamgoli/llm_test_datarepo/resolve/main/STOP_TRAINING"
    # 0 = Continue, 1 = Stop
    stop_flag = torch.tensor(0, device=device)
    if rank == 0:
        try:
            # We use HEAD to check existence without downloading the content
            response = requests.head(CONTROL_URL, timeout=timeout)
            if response.status_code == 404:
                print(f"\n[CONTROL] File NOT found at {CONTROL_URL}. Triggering STOP.", flush=True)
                stop_flag.fill_(1)
            elif response.status_code == 200:
                # File exists, keep training
                pass
            else:
                print(f"[CONTROL] Warning: Received status {response.status_code}. Continuing training by default.", flush=True)
        except requests.RequestException as e:
            # If the internet/HF is down, don't crash the training. Just continue.
            print(f"[CONTROL] Network error checking HF control file: {e}. Continuing...", flush=True)
    # Broadcast the decision from Rank 0 to all other GPUs
    if world_size > 1:
        dist.broadcast(stop_flag, src=0)
    return bool(stop_flag.item())
