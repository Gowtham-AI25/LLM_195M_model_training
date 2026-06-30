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
    def compute_grad_metrics(
        self,
        *,
        model: torch.nn.Module,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        Single-pass computation of both global gradient health
        and per-layer gradient flow diagnostics.

        Why combined:
        -------------
        Both metrics require iterating model parameters.
        Combining into one function avoids iterating the full
        parameter set twice — important for 195M param models.

        Inputs:
        -------
        model : torch.nn.Module
            Provides access to parameters, gradients, and
            transformer block structure.

        Returns:
        --------
        Tuple of two Dicts[str, float]:

        grad_metrics:
            {
                "grad/param_norm":          float,  # L2 norm of all parameters
                "grad/grad_to_param_ratio": float,  # ratio of grad norm to param norm
            }

        flow_metrics:
            {
                "grad_flow/first_block_norm":    float,  # grad norm at block 0
                "grad_flow/mid_block_norm":      float,  # grad norm at middle block
                "grad_flow/last_block_norm":     float,  # grad norm at last block
                "grad_flow/first_to_last_ratio": float,  # early/late gradient ratio
            }

        How to read flow_metrics:
        -------------------------
        ratio ~1.0  → healthy gradient flow through all layers
        ratio  0.1  → early layers getting 10x weaker gradients (vanishing)
        ratio trending down over training → intervene early
        """

        # -------------------------------------------------------
        # 🔹 ACCUMULATORS
        # -------------------------------------------------------
        param_norm_sq = 0.0   # sum of squared parameter norms
        grad_norm_sq  = 0.0   # sum of squared gradient norms

        # Track gradient norm squared for 3 representative blocks
        # first=block 0, mid=middle block, last=final block
        n = model.config.n_blocks
        block_indices = {
            "first": 0,
            "mid":   n // 2,
            "last":  n - 1,
        }
        # separate accumulator per block label
        block_grad_sq = {label: 0.0 for label in block_indices}

        # -------------------------------------------------------
        # 🔹 SINGLE PARAMETER PASS — no autograd graph needed
        # -------------------------------------------------------
        with torch.no_grad():

            # --- Pass 1: global norms across ALL parameters ---
            for p in model.parameters():
                # cast to float32 for numerical stability
                # bfloat16/float16 norm can accumulate error
                param_norm_sq += p.data.float().norm(2).pow(2).item()

                if p.grad is not None:
                    grad_norm_sq += p.grad.float().norm(2).pow(2).item()

            # --- Pass 2: attention grad norms for 3 blocks only ---
            # We use attention parameters specifically because:
            # - Attention layers are the most sensitive to vanishing gradients
            # - FFN grads are typically larger and mask the problem
            # - 3 blocks × ~4 attention params = 12 tensors total — very cheap
            for label, idx in block_indices.items():
                for p in model.transformer_blocks[idx].attn.parameters():
                    if p.grad is not None:
                        block_grad_sq[label] += (
                            p.grad.float().norm(2).pow(2).item()
                        )

        # -------------------------------------------------------
        # 🔹 FINALISE GLOBAL METRICS
        # -------------------------------------------------------
        param_norm = param_norm_sq ** 0.5
        grad_norm  = grad_norm_sq  ** 0.5

        grad_metrics = {
            # Total L2 norm of all model weights
            # Tracks whether weights are growing — should stay stable
            "grad/param_norm": param_norm,

            # Ratio of gradient magnitude to parameter magnitude
            # ~0.01 is healthy for LLM pretraining
            # approaching 0.0 → vanishing gradients globally
            # >> 0.1 → exploding gradients
            "grad/grad_to_param_ratio": grad_norm / (param_norm + 1e-8),
        }

        # -------------------------------------------------------
        # 🔹 FINALISE FLOW METRICS
        # -------------------------------------------------------
        first_norm = block_grad_sq["first"] ** 0.5
        mid_norm   = block_grad_sq["mid"]   ** 0.5
        last_norm  = block_grad_sq["last"]  ** 0.5

        flow_metrics = {
            # Gradient norm at block 0 — earliest layer
            # If this is near zero, block 0 is not learning
            "grad_flow/first_block_norm": first_norm,

            # Gradient norm at middle block
            # Useful midpoint reference
            "grad_flow/mid_block_norm": mid_norm,

            # Gradient norm at final block
            # Typically largest — backprop starts here
            "grad_flow/last_block_norm": last_norm,

            # Key diagnostic ratio: first block / last block
            # Tells you how much gradient signal survives 20 blocks of backprop
            # This is the metric that catches silent training failures early
            "grad_flow/first_to_last_ratio": first_norm / (last_norm + 1e-8),
        }

        return grad_metrics, flow_metrics
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
        with torch.no_grad():
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

        emb = model.emb_layer.emb_layer.weight.detach()
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
        del norms
    

        # Normalize
        emb_norm = F.normalize(emb_sample, dim=1)

        # Cosine similarity
        cosine_matrix = emb_norm @ emb_norm.T
        del emb_norm

        mask = ~torch.eye(N, dtype=torch.bool, device=emb.device)
        cosine_vals = cosine_matrix[mask]
        del cosine_matrix


        cosine_mean = cosine_vals.mean().item()
        cosine_var = cosine_vals.var().item()
        del cosine_vals

        # Pairwise distance
        dist_matrix = torch.cdist(emb_sample, emb_sample, p=2)
        del emb_sample

        dist_vals = dist_matrix[mask]
        del dist_matrix, mask

        dist_mean = dist_vals.mean().item()
        dist_std = dist_vals.std().item()
        del dist_vals

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

        if logits_flat.size(0) > 1024:
            idx = torch.randperm(
                logits_flat.size(0), device=logits_flat.device
            )[:1024]
            logits_flat = logits_flat[idx]
            del idx

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
    
    def compute_dead_neuron_metrics(
        self,
        *,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict[str, float]:

        dead_ratios = {}
        hooks = []
        n = model.config.n_blocks
        block_indices = [0, n // 2, n - 1]

        def make_hook(block_idx):
            def hook(module, inp, out):
                # inp is a tuple — inp[0] is the tensor going into w2
                # which is exactly F.silu(gate_proj) * up_proj
                # already computed by your FFN forward — zero extra work
                with torch.no_grad():
                    act = inp[0].float()
                    # shape: (B, T, ffn_hidden_dim//2)

                    # mean absolute activation per neuron
                    mean_abs = act.abs().mean(dim=(0, 1))
                    # shape: (ffn_hidden_dim//2,)

                    dead_ratios[block_idx] = (
                        mean_abs < threshold
                    ).float().mean().item()

                    del act, mean_abs

            return hook

        for idx in block_indices:
            h = model.transformer_blocks[idx].ffn.w2.register_forward_hook(
                make_hook(idx)
            )
            hooks.append(h)

        with torch.no_grad():
            model(sample_input)

        for h in hooks:
            h.remove()

        result = {
            f"dead_neurons/block_{idx}": ratio
            for idx, ratio in dead_ratios.items()
        }
        result["dead_neurons/mean"] = (
            sum(dead_ratios.values()) / max(1, len(dead_ratios))
        )
        return result