import math


def threshold_schedule(epoch, schedule, start, end, warmup_epochs, total_epochs):

    # compute normalized progress ∈ [0, 1]
    e = max(0, epoch - warmup_epochs)
    T = max(1, total_epochs - warmup_epochs)
    progress = e / T

    if schedule == "constant":
        return end
    if schedule == "linear":
        return start - (start - end) * progress
    if schedule == "cosine":
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return end + (start - end) * cosine
    if schedule == "inverse_sqrt":
        if progress == 0.0:
            return start  # avoid sqrt(0)
        return end + (start - end) * (1 - math.sqrt(progress))    
    raise ValueError(f"Unknown threshold schedule: {schedule}")    
