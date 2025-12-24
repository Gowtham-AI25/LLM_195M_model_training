import torch
import torch.distributed as dist
import requests
import logging

# Silence all intermediate redirect logs from the underlying libraries
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# Global session to keep the connection alive across shards
_hf_session = requests.Session()

def should_stop_from_hf(rank, world_size, device):
    """
    Checks for stop signal using GET (safe for 0KB files).
    - File exists (200 OK) -> Continue
    - File missing (404 Not Found) -> Stop
    """
    CONTROL_URL = "https://huggingface.co/datasets/gowthamgoli/llm_test_datarepo/resolve/main/STOP_TRAINING"
    
    # 0 = Continue, 1 = Stop
    stop_flag = torch.tensor(0, device=device)

    if rank == 0:
        try:
            # We use GET here. Since the file is 0KB, it's virtually free.
            # allow_redirects=True is critical to resolve 307 hops in the background.
            response = _hf_session.get(
                CONTROL_URL, 
                allow_redirects=True, 
                timeout=10
            )

            # Check the FINAL status code after all redirects
            if response.status_code == 404:
                print("\n[CONTROL] 🛑 STOP_TRAINING file missing! Ending session...", flush=True)
                stop_flag.fill_(1)
            
            # If 200, we stay silent. 
            # If we get a 307 here, it means redirects are failing to follow, 
            # but usually, with allow_redirects=True, the final code will be 200.

        except Exception as e:
            # If the internet fails, we don't want to kill the training.
            # We assume it's a glitch and continue.
            pass 

    # Synchronize the decision across all GPUs
    if world_size > 1:
        dist.broadcast(stop_flag, src=0)

    return bool(stop_flag.item())
