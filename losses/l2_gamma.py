import torch

def l2_gamma_batch(y_pred, y_true, eps=1e-8):
    """
    L2-Gamma-Batch loss (PyTorch version of TF code)

    Args:
        y_pred: predicted image tensor (B, C, H, W)
        y_true: target image tensor (B, C, H, W)
        eps: numerical stability constant

    Returns:
        scalar loss
    """

    # gamma transform (sqrt in your TF code)
    y_pred_gamma = torch.sqrt(torch.clamp(y_pred, min=0.0) + eps)
    y_true_gamma = torch.sqrt(torch.clamp(y_true, min=0.0) + eps)

    # L2
    diff = y_pred_gamma - y_true_gamma
    loss = torch.mean(diff ** 2)

    # root mean (matches tf.sqrt(tf.reduce_mean(...)))
    return torch.sqrt(loss)