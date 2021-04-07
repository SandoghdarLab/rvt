"""
Python implementation of the Radial Variance Transform.

The algorithm described in
A. D. Kashkanova, A. B. Shkarin, R. Gholami Mahmoodabadi, M. Blessing, Y. Tuna, A. Gemeinhardt, and V. Sandoghdar,
"Precision single-particle localization using radial variance transform," Opt. Express 29, 11070-11083 (2021)
(https://doi.org/10.1364/OE.420670)
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
import scipy.fftpack




def gen_r_kernel(r, rmax):
    """Generate a ring kernel with radius `r` and size ``2*rmax+1``"""
    a=rmax*2+1
    k=np.zeros((a,a))
    for i in range(a):
        for j in range(a):
            rij=((i-rmax)**2.+(j-rmax)**2)**.5
            if int(rij)==r:
                k[i][j]=1.
    return k/np.sum(k)

def generate_all_kernels(rmin, rmax, coarse_factor=1, coarse_mode="add"):
    """
    Generate a set of kernels with radii between `rmin` and `rmax` and sizes ``2*rmax+1``.

    ``coarse_factor`` and ``coarse_mode`` determine if the number of those kernels is reduced by either skipping or adding them
    (see :func:`rvt` for a more detail explanation).
    """
    kernels=[gen_r_kernel(r,rmax) for r in range(rmin,rmax+1)]
    if coarse_factor>1:
        if coarse_mode=="skip":
            kernels=kernels[::coarse_factor]
        else:
            kernels=[np.sum(kernels[i:i+coarse_factor],axis=0)/coarse_factor for i in range(0,len(kernels),coarse_factor)]
    return kernels


def _check_core_args(rmin, rmax, kind, coarse_mode="add"):
    """Check validity of the core algorithm arguments"""
    if rmin<0 or rmax<0:
        raise ValueError("radius should be non-negative")
    if rmin>=rmax:
        raise ValueError("rmax should be strictly greater than rmin")
    if kind not in {"basic","normalized"}:
        raise ValueError("unrecognized kind: {}; can be either 'basic' or 'normalized'")
    if coarse_mode not in {"add","skip"}:
        raise ValueError("unrecognized coarse mode: {}; can be either 'add' or 'skip'")
def _check_args(rmin, rmax, kind, coarse_mode, highpass_size, upsample):
    """Check validity of all the algorithm arguments"""
    _check_core_args(rmin,rmax,kind,coarse_mode)
    if upsample<1:
        raise ValueError("upsampling factor should be positive")
    if highpass_size is not None and highpass_size<0.3:
        raise ValueError("high-pass filter size should be >= 0.3")



## Prepare auxiliary parameters
def get_fshape(s1, s2, fast_mode=False):
    """Get the required shape of the transformed image given the shape of the original image and the kernel"""
    shape=s1 if fast_mode else s1+s2-1
    fshape=[scipy.fftpack.next_fast_len(int(d)) for d in shape]
    return tuple(fshape)
## Preparing FFTs of arrays
def prepare_fft(inp, fshape, pad_mode="constant"):
    """Prepare the image for a convolution by taking its Fourier transform, applying padding if necessary"""
    if pad_mode=="fast":
        return np.fft.rfftn(inp,fshape)
    else:
        pad=[((td-d)//2,(td-d+1)//2) for td,d in zip(fshape,inp.shape)]
        inp=np.pad(inp,pad,mode=pad_mode)
        return np.fft.rfftn(inp)
## Shortcut of SciPy fftconvolve, which takes already fft'd arrays on the input
def convolve_fft(sp1, sp2, s1, s2, fshape, fast_mode=False):
    """Calculate the convolution from the Fourier transforms of the original image and the kernel, trimming the result if necessary"""
    ret=np.fft.irfftn(sp1*sp2,fshape)
    if fast_mode:
        return np.roll(ret,(-(s2[0]//2),-(s2[1]//2)),(0,1))[:s1[0],:s1[1]].copy()
    else:
        off=(fshape[0]-s1[0])//2+s2[0]//2,(fshape[1]-s1[1])//2+s2[1]//2
        return ret[off[0]:off[0]+s1[0],off[1]:off[1]+s1[1]].copy()

_kernels_fft_cache={}
def rvt_core(img, rmin, rmax, kind="basic", rweights=None, coarse_factor=1, coarse_mode="add", pad_mode="constant"):
    """
    Perform core part of Radial Variance Transform (RVT) of an image.

    Args:
        img: source image (2D numpy array)
        rmin: minimal radius (inclusive)
        rmax: maximal radius (inclusive)
        kind: either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
            normalized version increases subpixel bias, but it works better at lower SNR
        rweights: relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``
        coarse_factor: the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision
        coarse_mode: the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)
        pad_mode: edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
            or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
            ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
            note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean

    Returns:
        transformed image with the same shape as upsample
    """
    _check_core_args(rmin,rmax,kind,coarse_mode) # check arguments validity
    s1=np.array(img.shape)
    s2=np.array([rmax*2+1,rmax*2+1])
    fast_mode=pad_mode=="fast"
    fshape=get_fshape(s1,s2,fast_mode=fast_mode) # calculate the padded image shape (add padding and get to the next "good" FFT size)
    cache_k=(rmin,rmax,coarse_factor,coarse_mode)+fshape
    if cache_k not in _kernels_fft_cache: # generate convolution kernels, if they are not in cache yet
        kernels=generate_all_kernels(rmin,rmax,coarse_factor=coarse_factor,coarse_mode=coarse_mode)
        _kernels_fft_cache[cache_k]=[prepare_fft(k,fshape,pad_mode="fast") for k in kernels] # pad_mode="fast" corresponds to the default zero-padding here
    kernels_fft=_kernels_fft_cache[cache_k] # get the convolution kernels (either newely generated, or cached)
    if rweights is not None:
        rweights=np.asarray(rweights)
        if len(rweights)!=len(kernels_fft):
            raise ValueError("the number of kernel weights {} is different from the number of kernels {}".format(len(rweights),len(kernels_fft)))
        rweights=rweights/rweights.sum() # normalize weights by their sum
    img=img-img.mean() # subtract mean (makes VoM calculation more stable and zero-padding more meaningful)
    img_fft=prepare_fft(img,fshape,pad_mode=pad_mode) # prepare image FFT (only needs to be done once)
    rmeans=np.array([convolve_fft(img_fft,k_fft,s1,s2,fshape,fast_mode=fast_mode) for k_fft in kernels_fft]) # calculate M_r for all radii
    if rweights is None:
        vom=np.var(rmeans,axis=0) # calculate VoM as a standard variance of M_r along the radius axis
    else:
        vom=np.sum(rmeans**2*rweights[:,None,None],axis=0)-np.sum(rmeans*rweights[:,None,None],axis=0)**2 # calculate VoM as a weighted variance of M_r along the radius axis
    if kind=="basic":
        return vom
    else: # calculate MoV for normalization
        imgsq_fft=prepare_fft(img**2,fshape,pad_mode=pad_mode) # prepare image FFT
        if rweights is None:
            sumk_fft=np.mean(kernels_fft,axis=0) # find combined kernel as a standard mean
            mov=convolve_fft(imgsq_fft,sumk_fft,s1,s2,fshape,fast_mode=fast_mode)-np.mean(rmeans**2,axis=0) # use the combined kernel to find MoV in one convolution
        else:
            sumk_fft=np.sum(kernels_fft*rweights[:,None,None],axis=0) # find combined kernel as a weighted mean
            mov=convolve_fft(imgsq_fft,sumk_fft,s1,s2,fshape,fast_mode=fast_mode)-np.sum(rmeans**2*rweights[:,None,None],axis=0) # use the combined kernel to find MoV in one convolution
        return vom/mov





def high_pass(img, size):
    """Perform Gaussian high-pass filter on the image"""
    img=img.astype(float)
    return img-scipy.ndimage.gaussian_filter(img,size)



def rvt(img, rmin, rmax, kind="basic", highpass_size=None, upsample=1, rweights=None, coarse_factor=1, coarse_mode="add", pad_mode="constant"):
    """
    Perform Radial Variance Transform (RVT) of an image.

    Args:
        img: source image (2D numpy array)
        rmin: minimal radius (inclusive)
        rmax: maximal radius (inclusive)
        kind: either ``"basic"`` (only VoM), or ``"normalized"`` (VoM/MoV);
            normalized version increases subpixel bias, but it works better at lower SNR
        highpass_size: size of the high-pass filter; ``None`` means no filter (effectively, infinite size)
        upsample: integer image upsampling factor;
            `rmin` and `rmax` are adjusted automatically (i.e., they refer to the non-upsampled image);
            if ``upsample>1``, the resulting image size is multiplied by ``upsample``
        rweights: relative weights of different radial kernels; must be a 1D array of the length ``(rmax-rmin+1)//coarse_factor``
        coarse_factor: the reduction factor for the number ring kernels; can be used to speed up calculations at the expense of precision
        coarse_mode: the reduction method; can be ``"add"`` (add ``coarse_factor`` rings in a row to get a thicker ring, which works better for smooth features),
            or ``"skip"`` (only keep on in ``coarse_factor`` rings, which works better for very fine features)
        pad_mode: edge padding mode for convolutions; can be either one of modes accepted by ``np.pad`` (such as ``"constant"``, ``"reflect"``, or ``"edge"``),
            or ``"fast"``, which means faster no-padding (a combination of ``"wrap"`` and ``"constant"`` padding depending on the image size);
            ``"fast"`` mode works faster for smaller images and larger ``rmax``, but the border pixels (within ``rmax`` from the edge) are less reliable;
            note that the image mean is subtracted before processing, so ``pad_mode="constant"`` (default) is equivalent to padding with a constant value equal to the image mean

    Returns:
        transformed image with a shape ``(img.shape[0]*upsample, img.shape[1]*upsample)``
    """
    upsample=int(upsample)
    _check_args(rmin,rmax,kind,coarse_mode,highpass_size,upsample)
    if highpass_size is not None:
        img=high_pass(img,highpass_size)
    if upsample>1:
        img=img.repeat(upsample,axis=-2).repeat(upsample,axis=-1) # nearest-neighbor upsampling on both axes
        if rweights is not None:
            rweights=np.asarray(rweights).repeat(upsample) # upsample radial weights as well
        rmin*=upsample # increase minimal and maximal radii
        rmax*=upsample
    return rvt_core(img,rmin,rmax,kind=kind,rweights=rweights,coarse_factor=coarse_factor,coarse_mode=coarse_mode,pad_mode=pad_mode)



def convert_upsampled_coordinate(coord, upsampling):
    """Convert the localized position in the upsampled image into the corresponding position in the original image"""
    return (coord+0.5)/upsampling-0.5