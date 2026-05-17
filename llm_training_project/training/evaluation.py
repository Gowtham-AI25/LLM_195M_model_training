import os
import torch
import math


def run_validation(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    hf_api,
    hf_file_url: str,
    local_path: str,
    device: torch.device,
    logger,
    step: int,
    max_batches: int = 50
):
    """
    Runs validation on a fixed dataset.

    Responsibilities:
    -----------------
    1. Ensure validation file exists (download if needed)
    2. Load validation data (.pt)
    3. Run model in eval mode (no grad)
    4. Compute average loss + perplexity
    5. Log to WandB

    Parameters:
    -----------
    model : torch.nn.Module
    criterion : loss function
    hf_api : HFUtils instance
    hf_file_url : str → remote validation file URL
    local_path : str → where to store/load file
    device : torch.device
    logger : AsyncWandBLogger or WandBLogger
    step : int → global step
    max_batches : int → limit eval compute
    """

    # =========================================================
    # 🔹 1. DOWNLOAD IF NOT EXISTS
    # =========================================================
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        hf_api.download_hf_file_from_url(
            file_url=hf_file_url,
            local_datasetdir=os.path.dirname(local_path)
        )

    # =========================================================
    # 🔹 2. LOAD VALIDATION DATA
    # =========================================================
    data = torch.load(local_path)

    # Expected format:
    # data = [(input_ids, target_ids), ...]
    # OR dict with tensors

    # Normalize format
    if isinstance(data, dict):
        inputs = data["input_ids"]
        targets = data["target_ids"]
        dataset = list(zip(inputs, targets))
    else:
        dataset = data

    # =========================================================
    # 🔹 3. EVAL MODE
    # =========================================================
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    # =========================================================
    # 🔹 4. VALIDATION LOOP (NO GRAD)
    # =========================================================
    with torch.no_grad():

        for i, (input_ids, target_ids) in enumerate(dataset):

            if i >= max_batches:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            total_loss += loss.item()
            total_tokens += input_ids.numel()
            num_batches += 1

    # =========================================================
    # 🔹 5. COMPUTE METRICS
    # =========================================================
    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    # =========================================================
    # 🔹 6. LOG TO WANDB
    # =========================================================
    if logger is not None:
        logger.log_validation(
            step=step,
            val_loss=avg_loss,
            val_perplexity=perplexity
        )

    # =========================================================
    # 🔹 7. BACK TO TRAIN MODE
    # =========================================================
    model.train()

    return {
        "validation/loss": avg_loss,
        "validation/perplexity": perplexity
    }