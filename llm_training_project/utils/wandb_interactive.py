import torch
import torch.distributed as dist
import wandb

def should_stop_from_wandb(rank, world_size, device):
    """
    Checks W&B dashboard for a 'stop_training' signal.
    """
    stop_flag = torch.tensor(0, device=device)
    
    if rank == 0:
        if wandb.run is not None:
            # Force-sync local config with W&B cloud
            wandb.run.config.update({}, allow_val_change=True)
            stop_value = wandb.run.config.get("stop_training", 0)
            if int(stop_value) == 1:
                print("\n[W&B] Remote stop signal detected!", flush=True)
                stop_flag.fill_(1)
                
    # Sync decision across all ranks to prevent deadlocks
    if world_size > 1:
        dist.broadcast(stop_flag, src=0)
        
    return bool(stop_flag.item())
