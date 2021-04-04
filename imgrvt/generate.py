"""
Synthetic data generation for RVT demonstration.
"""

# Copyright (C) 2021  Alexey Shkarin

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.




import numpy as np
import scipy.ndimage




def low_pass(img, scale):
    """Perform Gaussian high-pass filter on the image"""
    img=img.astype(float)
    return scipy.ndimage.gaussian_filter(img,scale)

def noise_background(shape, scale=None):
    """
    Generate a normally-distributed noise background.

    If `scale` is not ``None``, it defines the background correlation length;
    otherwise, the background is white noise (correlation length is zero).
    """
    if scale:
        noise=np.random.normal(size=(shape[0]*4,shape[1]*4))
        noise=low_pass(noise,scale)
        return noise[:shape[0],:shape[1]]/noise.std()
    else:
        return np.random.normal(size=shape)



def distgrid(shape, offset):
    """Generate a grid of x and y distance from the PSF center, which is defined by the offset from the image center"""
    xs,ys=np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),indexing="ij")
    dxs=xs-(shape[0]/2+offset[0])
    dys=ys-(shape[1]/2+offset[1])
    return dxs,dys

def gpsf(shape, width, offset=(0,0)):
    """
    Generate a Gaussian PSF.
    
    Args:
        shape: image shape
        width: PSF width
        offset: PSF offset from the image center
    """
    dxs,dys=distgrid(shape,offset)
    return np.exp(-(dxs**2+dys**2)/(2*width**2))
def ipsf(shape, scale, width, phi=0, offset=(0,0)):
    """
    Generate an interferometric PSF.
    
    Args:
        shape: image shape
        scale: scale of the interference pattern (period of the rings)
        width: overall Gaussian envelope width
        phi: phase of the interference pattern at the PSF center
        offset: PSF offset from the image center
    """
    dxs,dys=distgrid(shape,offset)
    rs=(dxs**2+dys**2)**.5
    return np.exp(-rs**2/(2*width**2))*np.cos(2*np.pi*rs/scale+phi)



def single_gaussian(snr=1, size=64, width=4, noise_scale=None, offset=(0,0), seed=0):
    """
    Generate a single Gaussian PSF on a random noise background.

    Args:
        snr: image SNR, which is the ratio of the PSF magnitude to the background noise RMS
        size: image size
        width: PSF width
        noise_scale: background noise correlation length (``None`` means white noise)
        offset: PSF offset from the image center
        seed: NumPy random seed
    """
    np.random.seed(seed)
    background=noise_background((size,size),scale=noise_scale)
    psf=gpsf((size,size),width,offset=offset)
    return psf+background/snr

def single_ipsf(snr=1, size=64, scale=8, width=10, noise_scale=None, offset=(0,0), seed=0):
    """
    Generate a single interferometric PSF on a random noise background.

    Args:
        snr: image SNR, which is the ratio of the PSF magnitude to the background noise RMS
        size: image size
        scale: scale of the interference pattern (period of the rings)
        width: overall Gaussian envelope width
        noise_scale: background noise correlation length (``None`` means white noise)
        offset: PSF offset from the image center
        seed: NumPy random seed
    """
    np.random.seed(seed)
    background=noise_background((size,size),scale=noise_scale)
    psf=ipsf((size,size),scale,width,offset=offset)
    return psf+background/snr


def double_ipsf(snr, size=64, scale=8, width=10, noise_scale=None, offset=(0,0), seed=0):
    """
    Generate a pair of interferometric PSFs on a random noise background.

    Args:
        snr: image SNR, which is the ratio of the PSF magnitude to the background noise RMS
        size: image size
        scale: scale of the interference pattern (period of the rings)
        width: overall Gaussian envelope width
        noise_scale: background noise correlation length (``None`` means white noise)
        offset: PSF offset from the image center; the two PSFs offsets are symmetric relative to the image center
        seed: NumPy random seed
    """
    np.random.seed(seed)
    background=noise_background((size,size),scale=noise_scale)
    psf1=ipsf((size,size),scale,width,offset=offset)
    psf2=ipsf((size,size),scale,width,offset=(-offset[0],-offset[1]))
    return psf1+psf2+background/snr