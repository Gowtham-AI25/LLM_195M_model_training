import torch
import torch.nn.functional as F
from typing import Dict
import math


class DiagnosticMetrics:
    """
    Centralized Metrics Computation Engine for LLM Training.

    This class provides a complete, structured set of diagnostic metrics covering:

        1. Learning Signals
        2. Training Stability
        3. Optimizer Health
        4. Embedding Representation (Magnitude + Geometry)
        5. Output Distribution (Logits)

    Design Principles:
    ------------------
    - Pure computation (no logging, no side effects)
    - WandB-ready output (flat dict: str → float)
    - Grouped by computational efficiency
    - Designed for periodic and step-based evaluation
    """

    # =========================================================
    # 🟢 1. EVERY-STEP METRICS
    # =========================================================
    def compute_step_metrics(
        self,
        *,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        grad_norm: float
    ) -> Dict[str, float]:
        """
        Computes core training metrics for every optimization step.

        Inputs:
        -------
        loss : torch.Tensor
            Scalar loss from forward pass.

        optimizer : torch.optim.Optimizer
            Used to extract current learning rate.

        scaler : torch.cuda.amp.GradScaler
            Used to obtain dynamic gradient scaling value.

        grad_norm : float
            Global gradient norm computed after unscaling.

        Returns:
        --------
        Dict[str, float]
            {
                "train/loss": float,
                "train/perplexity": float,
                "train/learning_rate": float,
                "train/grad_scale": float,
                "grad/global_norm": float
            }
        """

        loss_val = loss.item()
        perplexity = math.exp(loss_val) if loss_val < 20 else float("inf")
        lr = optimizer.param_groups[0]["lr"]
        grad_scale = scaler.get_scale()

        return {
            "train/loss": loss_val,
            "train/perplexity": perplexity,
            "train/learning_rate": lr,
            "train/grad_scale": grad_scale,
            "grad/global_norm": grad_norm,
        }

    # =========================================================
    # 🟡 2. GRADIENT + PARAMETER DIAGNOSTICS
    # =========================================================
    def compute_grad_param_metrics(
        self,
        *,
        model: torch.nn.Module,
    ) -> Dict[str, float]:
        """
        Computes relationships between gradients and parameters.

        Inputs:
        -------
        model : torch.nn.Module
            Provides access to parameters and gradients.

        Returns:
        --------
        Dict[str, float]
            {
                "grad/param_norm": float,
                "grad/grad_to_param_ratio": float
            }
        """

        param_norm_sq = 0.0
        grad_norm_sq = 0.0

        for p in model.parameters():
            param_norm_sq += p.data.norm(2).pow(2).item()
            if p.grad is not None:
                grad_norm_sq += p.grad.norm(2).pow(2).item()

        param_norm = param_norm_sq ** 0.5
        grad_norm = grad_norm_sq ** 0.5

        return {
            "grad/param_norm": param_norm,
            "grad/grad_to_param_ratio": grad_norm / (param_norm + 1e-8),
        }

    # =========================================================
    # 🟡 3. OPTIMIZER DIAGNOSTICS
    # =========================================================
    def compute_optimizer_metrics(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Computes optimizer-related diagnostic metrics.

        Inputs:
        -------
        model : torch.nn.Module

        optimizer : torch.optim.Optimizer

        Returns:
        --------
        Dict[str, float]
            {
                "optimizer/update_norm": float,
                "optimizer/update_to_param_ratio": float,
                "optimizer/weight_decay_effect": float,
                "optimizer/vt_mean": float
            }
        """

        lr = optimizer.param_groups[0]["lr"]
        wd = optimizer.param_groups[0].get("weight_decay", 0.0)

        update_norm_sq = 0.0
        param_norm_sq = 0.0
        wd_effect_sq = 0.0
        vt_sum = 0.0
        vt_count = 0

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                update = lr * p.grad
                update_norm_sq += update.norm(2).pow(2).item()
                param_norm_sq += p.data.norm(2).pow(2).item()

                if wd != 0.0:
                    wd_effect_sq += (wd * p.data).norm(2).pow(2).item()

                state = optimizer.state.get(p, {})
                if "exp_avg_sq" in state:
                    vt_sum += state["exp_avg_sq"].mean().item()
                    vt_count += 1

        update_norm = update_norm_sq ** 0.5
        param_norm = param_norm_sq ** 0.5

        return {
            "optimizer/update_norm": update_norm,
            "optimizer/update_to_param_ratio": update_norm / (param_norm + 1e-8),
            "optimizer/weight_decay_effect": wd_effect_sq ** 0.5,
            "optimizer/vt_mean": (vt_sum / vt_count) if vt_count > 0 else 0.0,
        }

    # =========================================================
    # 🔷 4. EMBEDDING DIAGNOSTICS (MAGNITUDE + GEOMETRY)
    # =========================================================
    def compute_embedding_diagnostics(
        self,
        *,
        model: torch.nn.Module,
        sample_size: int = 1024
    ) -> Dict[str, float]:
        """
        Computes embedding magnitude and geometry diagnostics.

        Inputs:
        -------
        model : torch.nn.Module
            Must contain embedding layer:
                model.tok_embeddings.weight

        sample_size : int
            Number of tokens sampled for computation.

        Returns:
        --------
        Dict[str, float]
            {
                "embedding/norm_mean": float,
                "embedding/norm_variance": float,
                "embedding/cosine_sim_mean": float,
                "embedding/cosine_sim_variance": float,
                "embedding/pairwise_dist_mean": float,
                "embedding/pairwise_dist_std": float,
                "embedding/anisotropy_score": float
            }
        """

        emb = model.tok_embeddings.weight.detach()
        vocab_size = emb.size(0)

        if sample_size < vocab_size:
            indices = torch.randperm(vocab_size, device=emb.device)[:sample_size]
            emb_sample = emb[indices]
        else:
            emb_sample = emb

        N = emb_sample.size(0)

        # Magnitude
        norms = emb_sample.norm(dim=1)
        norm_mean = norms.mean().item()
        norm_variance = norms.var().item()

        # Normalize
        emb_norm = F.normalize(emb_sample, dim=1)

        # Cosine similarity
        cosine_matrix = emb_norm @ emb_norm.T
        mask = ~torch.eye(N, dtype=torch.bool, device=emb.device)
        cosine_vals = cosine_matrix[mask]

        cosine_mean = cosine_vals.mean().item()
        cosine_var = cosine_vals.var().item()

        # Pairwise distance
        dist_matrix = torch.cdist(emb_sample, emb_sample, p=2)
        dist_vals = dist_matrix[mask]

        dist_mean = dist_vals.mean().item()
        dist_std = dist_vals.std().item()

        return {
            "embedding/norm_mean": norm_mean,
            "embedding/norm_variance": norm_variance,
            "embedding/cosine_sim_mean": cosine_mean,
            "embedding/cosine_sim_variance": cosine_var,
            "embedding/pairwise_dist_mean": dist_mean,
            "embedding/pairwise_dist_std": dist_std,
            "embedding/anisotropy_score": cosine_mean,
        }

    # =========================================================
    # 🔷 5. LOGIT DIAGNOSTICS (OUTPUT DISTRIBUTION)
    # =========================================================
    def compute_logit_diagnostics(
        self,
        *,
        logits: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes output distribution diagnostics from logits.

        Inputs:
        -------
        logits : torch.Tensor
            Shape: (batch_size, seq_len, vocab_size)

        Returns:
        --------
        Dict[str, float]
            {
                "model/logit_mean": float,
                "model/logit_variance": float,
                "model/logit_entropy": float,
                "model/logit_max_mean": float,
                "model/logit_margin_mean": float
            }
        """

        logits = logits.detach()
        logits_flat = logits.view(-1, logits.size(-1))

        logit_mean = logits_flat.mean().item()
        logit_var = logits_flat.var().item()

        probs = F.softmax(logits_flat, dim=-1)
        log_probs = torch.log(probs + 1e-9)

        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        max_logits = logits_flat.max(dim=-1).values
        max_mean = max_logits.mean().item()

        top2 = torch.topk(logits_flat, k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()

        return {
            "model/logit_mean": logit_mean,
            "model/logit_variance": logit_var,
            "model/logit_entropy": entropy,
            "model/logit_max_mean": max_mean,
            "model/logit_margin_mean": margin,
        }