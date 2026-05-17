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
        model: torch.nn.Module,
        important_layers: Optional[list] = None
    ) -> None:
        """
        Logs weight and gradient distributions.

        Inputs:
        -------
        step : int

        model : torch.nn.Module

        important_layers : Optional[list]
            List of parameter names to filter logging
        """

        if step % self.histogram_freq != 0:
            return

        hist_data = {}

        for name, param in model.named_parameters():

            if important_layers and name not in important_layers:
                continue

            hist_data[f"weights/{name}"] = wandb.Histogram(
                param.data.detach().cpu().numpy()
            )

            if param.grad is not None:
                hist_data[f"grads/{name}"] = wandb.Histogram(
                    param.grad.detach().cpu().numpy()
                )

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

    # =========================================================
    # 🔚 FINALIZE RUN
    # =========================================================
    def finish(self) -> None:
        """
        Ends WandB run.
        """
        wandb.finish()