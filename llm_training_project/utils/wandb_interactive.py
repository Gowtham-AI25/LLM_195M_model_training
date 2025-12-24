import torch
import torch.distributed as dist
import requests

def should_stop_from_hf(rank, world_size, device):
    """
    Remote stop control:
    - File exists (200 OK) -> Continue
    - File missing (404) -> Stop
    """
    # Hard-coded URL for your specific repository
    CONTROL_URL = "https://huggingface.co/datasets/gowthamgoli/llm_test_datarepo/resolve/main/STOP_TRAINING"
    # 0 = Continue, 1 = Stop
    stop_flag = torch.tensor(0, device=device)
    if rank == 0:
        try:
            r = requests.head(
                CONTROL_URL,
                allow_redirects=True, # Resolves the 307 Redirect issue
                timeout=5,
            )

            if r.status_code == 404:
                print("\n[CONTROL] 🛑 HF STOP_TRAINING file missing → Stopping training.", flush=True)
                stop_flag.fill_(1)
            elif r.status_code == 200:
                # File exists, everything is normal.
                pass
            else:
                # Handle 500, 503, etc. - assume continue if HF is having issues
                print(f"[CONTROL] Warning: HF returned status {r.status_code}. Continuing...", flush=True)

        except requests.RequestException as e:
            # Prevents crashing if the internet connection blips
            print(f"[CONTROL] HF check failed ({e}), continuing by default.", flush=True)

    # All ranks wait here for Rank 0 to broadcast the decision
    if world_size > 1:
        dist.broadcast(stop_flag, src=0)

    return bool(stop_flag.item())
