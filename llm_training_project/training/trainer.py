import torch 
from torch.nn.utils import clip_grad_norm_
from typing import Dict
from torch.amp import autocast
import math

def train_on_shard(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        sheduler,
        criterion: torch.nn.CrossEntropyLoss,  
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        start_global_step: int,
        max_grad_norm: float,
        gradient_accumulation_steps: int = 25,
        writer = None,
        rank: int = 0,
    ):
    """
        Train the model on a single data shard for one epoch.

        Args:
            model (torch.nn.Module): The model to be trained.
            dataloader (torch.utils.data.DataLoader): DataLoader for the shard.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            sheduler: Learning rate scheduler.
            device (torch.device): Device to perform training on.
            scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
            max_grad_norm (float): Maximum gradient norm for clipping.

        Returns:
            Dict[str, float]: Dictionary containing average loss and number of batchs.
    """
    # Set model to training mode
    model.train()
    # Initialize gradients to zero at the start of the shard training
    optimizer.zero_grad(set_to_none=True)

    total_raw_loss = 0.0
    num_micro_batchs = 0
    global_step = start_global_step

    for batch_idx, batch in enumerate(dataloader):

        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ---------- Forward pass with mixed precision ----------

        with autocast(device_type = device.type, dtype = torch.float16):
            logits = model(inputs)

            raw_loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            scaled_loss = raw_loss / gradient_accumulation_steps 

        # ---------- Backward pass with gradient scaling ----------
        scaler.scale(scaled_loss).backward()

        # ---------- collect raw loss for logging ----------

        raw_loss_value = raw_loss.item()
        total_raw_loss += raw_loss_value
        num_micro_batchs += 1

        # ---------- Gradient accumulation step ----------

        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            continue

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
 
        # Clip gradients to prevent exploding gradients
        grad_norm = clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        ).item()

        if rank == 0 and global_step % 100 == 0 and writer is not None:

            writer.log_histograms(
                step=global_step,
                model=model
            )

            writer.log_diagnostics(
                step=global_step,
                model=model,
                optimizer=optimizer
            )



        # optimize the model parameters
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        sheduler.step()
        global_step += 1

        # 1 Perplexity (numerically safe)
        perplexity = math.exp(raw_loss_value) if raw_loss_value < 20 else float("inf")
        # 2 Learning rate (current step)
        lr = optimizer.param_groups[0]["lr"]
        # 3 AMP gradient scaler value
        grad_scale = scaler.get_scale()

        if rank == 0 and writer is not None:
            writer.log_training_metric(
                step=global_step,
                loss=raw_loss_value,
                perplexity = perplexity,
                lr = lr,
                grad_scale = grad_scale,
                grad_norm = grad_norm,
            )
            print(f"Step {global_step}: Loss={raw_loss_value:.6f}, Perplexity={perplexity:.6f}, LR={lr:.6e}")

    return {
        "global_step": global_step,
        "total_loss": total_raw_loss,
        "num_micro_batchs": num_micro_batchs
    }


                                           



