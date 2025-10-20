import math
from torch.optim.lr_scheduler import LambdaLR

def warmup_with_cosine_decay_lambda(current_step: int, *, warmup_steps: int, total_steps: int, decay_type: str = 'cosine'):
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    if decay_type == 'cosine':
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    elif decay_type == 'no_decay':
         return progress

def create_scheduler(optimizer, number_of_warm_up_steps, number_of_total_steps, decay_type):

    return LambdaLR(optimizer, lr_lambda=lambda step: warmup_with_cosine_decay_lambda(step, warmup_steps=number_of_warm_up_steps, total_steps=number_of_total_steps, decay_type=decay_type))