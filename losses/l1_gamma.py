import torch

def l1_gamma_batch(y_pred, y_true, eps=1e-8):
    """
    L1-Gamma-Batch loss 
    
    Replaces L2 (RMSE) with L1 (Mean Absolute Error) to act as a looser 
    structural leash, allowing high-frequency texture generation by the GAN.

    Args:
        y_pred: predicted image tensor (B, C, H, W)
        y_true: target image tensor (B, C, H, W)
        eps: numerical stability constant

    Returns:
        scalar loss
    """

    # gamma transform (sqrt) - kept identical to maintain the same perception space
    y_pred_gamma = torch.sqrt(torch.clamp(y_pred, min=0.0) + eps)
    y_true_gamma = torch.sqrt(torch.clamp(y_true, min=0.0) + eps)

    # L1 (Mean Absolute Error)
    diff = y_pred_gamma - y_true_gamma
    loss = torch.mean(torch.abs(diff))

    return loss