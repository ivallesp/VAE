def run_epoch(model, batcher):
    """Run an epoch over a given model and return the loss

    Args:
        model (torch.nn.Module): model to train
        batcher (torch.utils.data.DataLoader): data batcher to train the model on

    Returns:
        float: average loss over the epoch, computed before running the step
    """
    epoch_loss = 0
    for i, (x, y) in enumerate(batcher):
        loss, x_hat = model.step(x)
        epoch_loss += loss.data.numpy().mean()
    epoch_loss /= i
    return epoch_loss
