import torch
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast
from typing import Dict

from llm_training_project.log.diagnostic_metrics import DiagnosticMetrics
from llm_training_project.log.wandb_logger import WandBLogger
from llm_training_project.log.async_wandb_logger import AsyncWandBLogger
from llm_training_project.training.evaluation import run_validation


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
        logger: WandBLogger = None,
        metrics_engine: DiagnosticMetrics = None,
        rank: int = 0,
        dtype: torch.dtype = None,
        val_local_path: str = None,
) -> Dict[str, float]:
    """
    Train model on a single shard with efficient metric computation and logging.

    Design:
    -------
    - Step metrics → computed & logged every step
    - Other metrics → computed ONLY when required (based on frequency)
    - No redundant computation
    - Fully decoupled: Trainer → Metrics → Logger

    Returns:
    --------
    Dict with:
        global_step
        total_loss
        num_micro_batches
    """

    # ---------------------------------------------------------
    # 🔹 INITIAL SETUP
    # ---------------------------------------------------------
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_raw_loss = 0.0
    num_micro_batches = 0
    global_step = start_global_step

    accum_loss = 0.0
    num_batches = len(dataloader)

    # Convert dtype string → torch dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # ---------------------------------------------------------
    # 🔁 MAIN TRAINING LOOP
    # ---------------------------------------------------------
    for batch_idx, batch in enumerate(dataloader):

        torch.compiler.cudagraph_mark_step_begin()

        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # -----------------------------------------------------
        # 🔹 FORWARD PASS (Mixed Precision)
        # -----------------------------------------------------
        with autocast(device_type=device.type, dtype=dtype):
            logits = model(inputs)

            raw_loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # Scale loss for gradient accumulation
            window_start = batch_idx - (batch_idx % gradient_accumulation_steps)
            remaining_in_shard = num_batches - window_start
            current_accum_size = min(gradient_accumulation_steps, remaining_in_shard)

            scaled_loss = raw_loss / current_accum_size

        # -----------------------------------------------------
        # 🔹 BACKWARD PASS (Accumulation Logic)
        # -----------------------------------------------------
        is_accum_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_final_batch = (batch_idx + 1) == num_batches

        should_update = is_accum_step or (is_final_batch and not is_accum_step)

        if not should_update:
            # Avoid gradient sync in DDP for micro-steps
            with model.no_sync():
                scaler.scale(scaled_loss).backward()
        else:
            scaler.scale(scaled_loss).backward()

            # -------------------------------------------------
            # 🔹 PRE-OPTIMIZATION (Unscale + Fix)
            # -------------------------------------------------
            scaler.unscale_(optimizer)

            # Fix scaling for partial accumulation batches
            actual_accum_count = (batch_idx % gradient_accumulation_steps) + 1
            if should_update and not is_accum_step:
                scale_fix = gradient_accumulation_steps / actual_accum_count
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(scale_fix)

            # -------------------------------------------------
            # 🔹 GRADIENT CLIPPING
            # -------------------------------------------------
            grad_norm = clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            ).item()

            # -------------------------------------------------
            # 🔹 METRICS COMPUTATION + LOGGING
            # -------------------------------------------------
            if rank == 0 and logger is not None and metrics_engine is not None:

                # =========================
                # 🟢 STEP METRICS (EVERY STEP)
                # =========================
                step_metrics = metrics_engine.compute_step_metrics(
                    loss=raw_loss,
                    optimizer=optimizer,
                    scaler=scaler,
                    grad_norm=grad_norm
                )

                logger.log_step(
                    step=global_step,
                    step_metrics=step_metrics
                )

                # =========================
                # 🟡 INTERVAL METRICS
                # =========================
                if global_step % logger.interval_freq == 0:

                    grad_metrics = metrics_engine.compute_grad_param_metrics(
                        model=model
                    )

                    opt_metrics = metrics_engine.compute_optimizer_metrics(
                        model=model,
                        optimizer=optimizer
                    )

                    logger.log_interval(
                        step=global_step,
                        grad_metrics=grad_metrics,
                        optimizer_metrics=opt_metrics
                    )

                # =========================
                # 🔷 LOGIT METRICS
                # =========================
                if global_step % logger.logit_freq == 0:

                    logit_metrics = metrics_engine.compute_logit_diagnostics(
                        logits=logits
                    )

                    logger.log_logits(
                        step=global_step,
                        logit_metrics=logit_metrics
                    )

                # =========================
                # 🔷 EMBEDDING METRICS
                # =========================
                if global_step % logger.embedding_freq == 0:

                    emb_metrics = metrics_engine.compute_embedding_diagnostics(
                        model=model
                    )

                    logger.log_embedding(
                        step=global_step,
                        embedding_metrics=emb_metrics
                    )

                # =========================
                # 🔵 HISTOGRAMS
                # =========================
                if global_step % logger.histogram_freq == 0:

                    logger.log_histograms(
                        step=global_step,
                        model=model
                    )
                
            # =========================================================
            # 🔴 VALIDATION (EVERY 100 STEPS)
            # =========================================================
            if (
                    rank == 0 and
                    logger is not None and
                    global_step % 100 == 0
                ):
                    run_validation(
                        model=model,
                        criterion=criterion,
                        local_path=val_local_path,         # local .pt path
                        device=device,
                        logger=logger,
                        step=global_step,
                        max_batches=50                     # keep it lightweight
                    )

            # -------------------------------------------------
            # 🔹 OPTIMIZER STEP
            # -------------------------------------------------
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # Reset accumulation trackers
            accum_loss = 0.0
            global_step += 1

        # -----------------------------------------------------
        # 🔹 GLOBAL STATS TRACKING
        # -----------------------------------------------------
        raw_loss_val = raw_loss.item()
        accum_loss += raw_loss_val
        total_raw_loss += raw_loss_val
        num_micro_batches += 1

    # ---------------------------------------------------------
    # 🔹 RETURN SHARD STATS
    # ---------------------------------------------------------
    return {
        "global_step": global_step,
        "total_loss": total_raw_loss,
        "num_micro_batches": num_micro_batches
    }