import torch

def hazen_williams(L, Q, C, D):
    """
    Computes the head loss due to friction in a pipe using the Hazen-Williams equation.
    
    h_f = 10.67 * L * |Q|^1.852 / (C^1.852 * D^4.87) * sign(Q)
    
    Args:
        L (Tensor): Length of the pipe (m)
        Q (Tensor): Volumetric flow rate (m^3/s)
        C (Tensor): Roughness coefficient
        D (Tensor): Diameter of the pipe (m)
        
    Returns:
        h_f (Tensor): Head loss (m)
    """
    # Small epsilon to avoid NaN gradients at Q=0 since derivative of |Q|^1.852 at 0 might be problematic
    eps = 1e-8
    
    abs_Q = torch.abs(Q) + eps
    
    numerator = 10.67 * L * torch.pow(abs_Q, 1.852)
    denominator = torch.pow(C, 1.852) * torch.pow(D, 4.87)
    
    h_f = (numerator / denominator) * torch.sign(Q)
    
    return h_f
