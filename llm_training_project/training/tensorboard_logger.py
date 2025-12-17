import torch
from torch.utils.tensorboard import SummaryWriter
import os

class TensorboardLogger:

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_training_metric(self,
                            step: int,
                            loss: float,
                            perplexity: float,
                            lr: float,
                            grad_norm: float,
                            clip_ratio: float,
                            gard_scale: float
                        ):
        """
        Logs training metrics to TensorBoard.
        Args:
            step: Current training step
            loss: Training loss
            perplexity: Training perplexity
            lr: Learning rate
            grad_norm: Gradient norm
            clip_ratio: Gradient clipping ratio
            gard_scale: Gradient scaler value

            All values are final processed values for the current step.
        """
        
        # main metrics logging
        self.writer.add_scalar('Training/Loss', loss, step)
        self.writer.add_scalar('Training/Perplexity', perplexity, step)
        self.writer.add_scalar('Training/lr', lr, step)

        # Optimizer and amp health metrics logging
        self.writer.add_scalar('Training/Gradient_Norm', grad_norm, step)
        self.writer.add_scalar('Training/Clip_Ratio', clip_ratio, step)
        self.writer.add_scalar('Training/Grad_Scaler', gard_scale, step)    
                            
                