import torch

# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#%E2%84%93_=_1_2
SH_C0_0 = 0.28209479177387814  # Y0,0:  1/2*sqrt(1/pi)       plus
SH_C1_0 = -0.4886025119029199  # Y1,-1: sqrt(3/(4*pi))       minus
SH_C1_1 = 0.4886025119029199   # Y1,0:  sqrt(3/(4*pi))       plus
SH_C1_2 = -0.4886025119029199  # Y1,1:  sqrt(3/(4*pi))       minus


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Supports torch and NumPy coefficient arrays.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH degree. Supports degrees 0 and 1.
        sh: array SH coeffs [..., C, (deg + 1) ** 2]
        dirs: array unit directions [..., 3]
    Returns:
        [..., C]
    """

    assert deg in (0, 1), "This assignment uses only SH degrees 0 and 1"
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff
    result = SH_C0_0 * sh[..., 0]
    
    if deg == 1:
        # Begin code 3.1 ##
        result = (
            result
            + SH_C1_0 * dirs[..., 1:2] * sh[..., 1]
            + SH_C1_1 * dirs[..., 2:3] * sh[..., 2]
            + SH_C1_2 * dirs[..., 0:1] * sh[..., 3]
        )
        # End code 3.1 ##
    return result
