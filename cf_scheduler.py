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

## Restart decay thingo 

def warmup_with_periodic_restarts_lambda(current_step: int,*,warmup_steps: int,total_steps: int,steps_per_epoch: int,cycle_length: int, decay_type: str = "cosine"):

    progress = float(current_step) / float(max(1, total_steps))
    if decay_type == "cosine":
        global_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif decay_type == "no_decay":
        global_scale = 1.0
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    cycle_steps = steps_per_epoch * cycle_length
    cycle_step = current_step % cycle_steps

    if cycle_step < warmup_steps:
        # Linear warmup inside this cycle (to the *global* LR peak)
        return global_scale * (cycle_step / float(max(1, warmup_steps)))
    else:
        return global_scale

def create_cyclical_scheduler(optimizer,number_of_warm_up_steps,number_of_total_steps,decay_type,steps_per_epoch,cycle_length):
    
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_with_periodic_restarts_lambda(
            step,
            warmup_steps=number_of_warm_up_steps,
            total_steps=number_of_total_steps,
            steps_per_epoch=steps_per_epoch,
            cycle_length=cycle_length,
            decay_type=decay_type,
        ),
    )