from skimage.metrics import structural_similarity as compare_ssim
from numpy import ndarray

# monai.losses.ssim_loss can be used as a metric
# but in order to match numbers with the fastMRI leaderboard,
# we use scikit-image ssim metric
def skimage_ssim(gt: ndarray, rec: ndarray) -> float:
    """
    Compute SSIM quality assessment metric
    inputs:
        gt: ground truth with the last two dims being spatial and the number of slices
        rec: reconstruction with the same dimensions as gt
    outputs:
        skimage SSIM score between gt and rec
    """
    # assumes 3D inputs
    return compare_ssim(gt.transpose(1, 2, 0), rec.transpose(1, 2, 0), multichannel=True, data_range=gt.max())
