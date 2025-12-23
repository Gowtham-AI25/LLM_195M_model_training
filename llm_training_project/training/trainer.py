import torch 
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast
import math
import wandb

def train_on_shard(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: torch.nn.CrossEntropyLoss,  
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        start_global_step: int, 
        max_grad_norm: float,
        gradient_accumulation_steps: int = 25,
        writer = None,
        rank: int = 0,
        dtype: torch.dtype = None # Use dynamic dtype from config
    ):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_raw_loss = 0.0
    num_micro_batches = 0
    global_step = start_global_step
    
    # Track loss for the current accumulation cycle
    accum_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # Because config get string i needed to convert it
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        # 1. Forward pass with dynamic dtype
        with autocast(device_type=device.type, dtype=dtype):
            logits = model(inputs)
            raw_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Scale by actual steps to be safe if final batch is partial
            scaled_loss = raw_loss / gradient_accumulation_steps 

        # 2. Backward pass
        is_accum_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_final_batch = (batch_idx + 1) == num_batches

        if not (is_accum_step or is_final_batch):
            with model.no_sync():
                scaler.scale(scaled_loss).backward()
        else:
            scaler.scale(scaled_loss).backward()

        # Stats tracking
        raw_loss_val = raw_loss.item()
        accum_loss += raw_loss_val
        total_raw_loss += raw_loss_val
        num_micro_batches += 1

        # 3. Optimization Step (Triggered every 25 micro-batches OR at end of shard)
        if is_accum_step or is_final_batch:
            scaler.unscale_(optimizer)
            
            # Compute grad_norm outside diagnostic block to avoid NameError
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm).item()

            # 4. Diagnostic Logging (End of Shard: 100th step)
            if rank == 0 and writer is not None:
                if global_step % 50 == 0:
                    writer.log_histograms(step=global_step, model=model)
                    writer.log_diagnostics(step=global_step, model=model, optimizer=optimizer)

                # 5. Regular Metric Logging (Every step)
                avg_step_loss = accum_loss / ( (batch_idx % gradient_accumulation_steps) + 1 )
                perplexity = math.exp(avg_step_loss) if avg_step_loss < 20 else float("inf")

                print(f"Step {global_step} | Loss: {avg_step_loss:.4f} | Perplexity: {perplexity:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

                if wandb.run is not None:
                    wandb.log({
                        "performance/loss": avg_step_loss,
                        "performance/perplexity": perplexity,
                        "performance/learning_rate": optimizer.param_groups[0]["lr"],
                        "system/grad_norm": grad_norm,
                        "system/grad_scale": scaler.get_scale()
                    }, step=global_step)
                        
                writer.log_training_metric(
                    step=global_step,
                    loss=avg_step_loss,
                    perplexity=perplexity,
                    lr=optimizer.param_groups[0]["lr"],
                    grad_scale=scaler.get_scale(), # Matches logger parameter name
                    grad_norm=grad_norm
                )
            
            # 6. Weights update
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            # Reset accumulation trackers
            accum_loss = 0.0
            global_step += 1

    return {
        "global_step": global_step,
        "total_loss": total_raw_loss,
        "num_micro_batches": num_micro_batches
    }
