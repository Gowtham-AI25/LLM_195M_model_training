import wandb
from typing import Dict, Any, Optional
import torch


class WandBLogger:
    """
    Centralized WandB Logging Class for LLM Training Diagnostics.

    Responsibilities:
    -----------------
    - Initialize WandB run
    - Log metrics at correct frequency
    - Separate scalar, histogram, and structured logs
    - Maintain clean metric namespaces for dashboard clarity

    This class does NOT compute metrics.
    It only logs outputs from DiagnosticMetrics.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        scalar_freq: int = 1,
        interval_freq: int = 100,
        embedding_freq: int = 200,
        logit_freq: int = 100,
        histogram_freq: int = 500,
    ):
        """
        Initializes WandB run and logging configuration.

        Inputs:
        -------
        project : str
            WandB project name

        run_name : str
            Name of the experiment run

        config : Dict[str, Any]
            Training configuration (lr, batch size, etc.)

        scalar_freq : int
            Frequency for step metrics (every step usually)

        interval_freq : int
            Frequency for optimizer + gradient diagnostics

        embedding_freq : int
            Frequency for embedding diagnostics

        logit_freq : int
            Frequency for logit diagnostics

        histogram_freq : int
            Frequency for histogram logging
        """

        wandb.init(
            project=project,
            name=run_name,
            config=config
        )

        self.scalar_freq = scalar_freq
        self.interval_freq = interval_freq
        self.embedding_freq = embedding_freq
        self.logit_freq = logit_freq
        self.histogram_freq = histogram_freq

    # =========================================================
    # 🟢 STEP LOGGING (EVERY STEP)
    # =========================================================
    def log_step(
        self,
        *,
        step: int,
        step_metrics: Dict[str, float]
    ) -> None:
        """
        Logs core step metrics (loss, perplexity, gradients).

        Inputs:
        -------
        step : int
            Global training step

        step_metrics : Dict[str, float]
            Output from compute_step_metrics()
        """

        if step % self.scalar_freq == 0:
            wandb.log(step_metrics, step=step)

    # =========================================================
    # 🟡 INTERVAL LOGGING (GRAD + OPTIMIZER)
    # =========================================================
    def log_interval(
        self,
        *,
        step: int,
        grad_metrics: Dict[str, float],
        optimizer_metrics: Dict[str, float]
    ) -> None:
        """
        Logs gradient and optimizer diagnostics.

        Inputs:
        -------
        step : int

        grad_metrics : Dict[str, float]

        optimizer_metrics : Dict[str, float]
        """

        if step % self.interval_freq == 0:
            metrics = {**grad_metrics, **optimizer_metrics}
            wandb.log(metrics, step=step)

    # =========================================================
    # 🔷 EMBEDDING LOGGING
    # =========================================================
    def log_embedding(
        self,
        *,
        step: int,
        embedding_metrics: Dict[str, float]
    ) -> None:
        """
        Logs embedding diagnostics.

        Inputs:
        -------
        step : int

        embedding_metrics : Dict[str, float]
            Output from compute_embedding_diagnostics()
        """

        if step % self.embedding_freq == 0:
            wandb.log(embedding_metrics, step=step)

    # =========================================================
    # 🔷 LOGIT LOGGING
    # =========================================================
    def log_logits(
        self,
        *,
        step: int,
        logit_metrics: Dict[str, float]
    ) -> None:
        """
        Logs output distribution diagnostics.

        Inputs:
        -------
        step : int

        logit_metrics : Dict[str, float]
            Output from compute_logit_diagnostics()
        """

        if step % self.logit_freq == 0:
            wandb.log(logit_metrics, step=step)

    # =========================================================
    # 🔵 HISTOGRAM LOGGING
    # =========================================================
    def log_histograms(
        self,
        *,
        step: int,
        hist_data: dict,
    ) -> None:
        if step % self.histogram_freq != 0:
            return
        wandb.log(hist_data, step=step)


    # =========================================================
    # 🟣 VALIDATION LOGGING
    # =========================================================
    def log_validation(
        self,
        *,
        step: int,
        val_loss: float,
        val_perplexity: float
    ) -> None:
        """
        Logs validation metrics.

        Inputs:
        -------
        step : int

        val_loss : float

        val_perplexity : float
        """

        wandb.log({
            "validation/loss": val_loss,
            "validation/perplexity": val_perplexity
        }, step=step)

    #=================================================
    # Log Dead neurons
    #=================================================
        
    def log_dead_neurons(
        self,
        *,
        step: int,
        dead_metrics: Dict[str, float]
    ) -> None:
        wandb.log(dead_metrics, step=step)

    def log_gradient_flow(
        self,
        *,
        step: int,
        flow_metrics: Dict[str, float]
    ) -> None:
        """
        Logs per-layer gradient flow diagnostics.

        Inputs:
        -------
        step : int

        flow_metrics : Dict[str, float]
            Output from compute_grad_metrics() second return value.
            {
                "grad_flow/first_block_norm":    float,
                "grad_flow/mid_block_norm":      float,
                "grad_flow/last_block_norm":     float,
                "grad_flow/first_to_last_ratio": float,
            }
        """
        if step % self.interval_freq == 0:
            wandb.log(flow_metrics, step=step)

    # =========================================================
    # 🔚 FINALIZE RUN
    # =========================================================
    def finish(self) -> None:
        """
        Ends WandB run.
        """
        wandb.finish()