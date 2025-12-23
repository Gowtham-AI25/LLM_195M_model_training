import torch
from torch.utils.tensorboard import SummaryWriter
import os

class TensorBoardLogger:

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

        self.histogram_param_names = [
            "tok_embeddings.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.out_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ]

    def get_diagnostic_metrics(self, model, optimizer):
        """
        Returns a list of diagnostic metrics being logged.
        Args:
            model: The model being trained.
            optimizer: The optimizer being used.
        Returns:
            param_norm: L2 norm of model parameters.
            step_size: Current optimizer step size (learning rate).
            weight_decay: Weight decay value from the optimizer.
        """
        lr = optimizer.param_groups[0]["lr"]
        wd = optimizer.param_groups[0].get("weight_decay", 0.0)

        param_norm_sq = 0.0
        step_size_sq = 0.0
        wd_effect_sq = 0.0

        for p in model.parameters():
            # Parameter norm
            p_data = p.data
            param_norm_sq += p_data.norm(2).pow(2).item()

            # Weight decay effect
            if wd != 0.0:
                wd_effect_sq += (wd * p_data).norm(2).pow(2).item()

            # Step size (only if gradient exists)
            if p.grad is not None:
                step_size_sq += (lr * p.grad).norm(2).pow(2).item()


        return param_norm_sq ** 0.5, step_size_sq ** 0.5, wd_effect_sq ** 0.5

    def log_diagnostics(self, step: int,
                        model: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer
                    ) -> None :
        """
        Logs diagnostic metrics to TensorBoard.
        Args:
            step: Current training step
            model: The model being trained.
            optimizer: The optimizer being used.
        """
        # atucally step is global step we have to log every 100 steps
        step = step // 100
        param_norm, step_size, weight_decay = self.get_diagnostic_metrics(model, optimizer)

        self.writer.add_scalar('Diagnostics/Param_Norm', param_norm, step)
        self.writer.add_scalar('Diagnostics/Step_Size', step_size, step)
        self.writer.add_scalar('Diagnostics/Weight_Decay_Effect', weight_decay, step)

    def log_histogram(self,
                      step: int,
                      model: torch.nn.Module
                    ) -> None:
        """
        Logs histograms of model parameters and gradients to TensorBoard.
        Args:
            step: Current training step
            model: The model being trained.
        """
        hist_step = step // 100

        for name, param in model.named_parameters():

            # -------- FILTER IMPORTANT LAYERS ONLY --------
            if name not in self.histogram_param_names:
                continue

            # -------- Parameter histogram --------
            self.writer.add_histogram(
                tag=f"parameters/{name}",
                values=param.data,
                global_step=hist_step,
            )

            # -------- Gradient histogram --------
            if param.grad is not None:
                self.writer.add_histogram(
                    tag=f"gradients/{name}",
                    values=param.grad,
                    global_step=hist_step,
                )

    def log_training_metric(self,
                            step: int,
                            loss: float,
                            perplexity: float,
                            lr: float,
                            gard_scale: float,
                            grad_norm: float
                        ) -> None:
        """
        Logs training metrics to TensorBoard.
        Args:
            step: Current training step
            loss: Training loss
            perplexity: Training perplexity
            lr: Learning rate
            gard_scale: Gradient scaler value

            All values are final processed values for the current step.
        """
        max_gard_norm = grad_norm
        clip_ratio = min(1.0, max_gard_norm / (grad_norm + 1e-6))
        # main metrics logging
        self.writer.add_scalar('Tranining/Loss', loss, step)
        self.writer.add_scalar('Training/Perplexity', perplexity, step)
        self.writer.add_scalar('Training/lr', lr, step)
        # amp health metrics logging
        self.writer.add_scalar('Training/Grad_Scaler', gard_scale, step)   
        self.writer.add_scalar('Training/Grad_Norm', grad_norm, step) 
        self.writer.add_scalar('Training/Grad_Clip_Ratio', clip_ratio, step)
                            
                