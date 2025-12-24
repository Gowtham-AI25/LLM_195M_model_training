import torch
import torch.distributed as dist
import wandb

def should_stop_from_wandb(rank, world_size, device):
    stop_flag = torch.tensor(0, device=device)
    
    if rank == 0:
        if wandb.run is not None:
            # Refresh the local run object with cloud data
            wandb.run.api.flush() # Optional: ensure previous logs are sent
            # This is the standard way to pull the latest config from W&B
            wandb.run.refresh() 
            
            stop_value = wandb.run.config.get("stop_training", 0)
            if int(stop_value) == 1:
                print("\n[W&B] Remote stop signal detected!", flush=True)
                stop_flag.fill_(1)
                
    if world_size > 1:
        # This acts as both a communication and a synchronization point
        dist.broadcast(stop_flag, src=0)
        
    return bool(stop_flag.item())
