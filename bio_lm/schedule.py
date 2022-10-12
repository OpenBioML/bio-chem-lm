import torch.optim.lr_scheduler as schedule


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step):
        learning_rate = max(
            0.0, 1.0 - (float(current_step) / float(num_training_steps))
        )
        learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
        return learning_rate

    return schedule.LambdaLR(optimizer, lr_lambda, last_epoch)
