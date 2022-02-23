import deeptrack as dt
import numpy as np


def _propagate_field(field, Tz, z, x, y, K, C, k=2 * np.pi / 0.633):
    """
    Propagate field. Field =  a Complex array of row*col dimension
    """
    Field = np.fft.fft2(field)
    Field = C * Tz * Field
    Field = np.fft.ifft2(Field)

    return Field


def _precalc(field, px, k=2 * np.pi / 0.633):
    """
    Precalculate some constants for propagating field for faster computations.
    """
    yr, xr = field.real.shape

    x = 2 * np.pi / px * np.arange(-(xr / 2 - 1 / 2), (xr / 2 + 1 / 2), 1) / xr
    y = 2 * np.pi / px * np.arange(-(yr / 2 - 1 / 2), (yr / 2 + 1 / 2), 1) / yr
    KXk, KYk = np.meshgrid(x, y)

    K = np.real(
        np.sqrt(np.array(1 - (KXk / k) ** 2 - (KYk / k) ** 2, dtype=np.complex64))
    )
    # Create a circular disk here.
    C = np.fft.fftshift(((KXk / k) ** 2 + (KYk / k) ** 2 < 1) * 1.0)

    return x, y, K, C


def _precalc_Tz(k, zv, K, C):
    return [C * np.fft.fftshift(np.exp(k * 1j * z * (K - 1))) for z in zv]


def propagation_matrix(
    z,
    shape=(64, 64),
    padding=64,
    wavelength=633e-9, #525 nm!
    pixel_size=0.34e-6,
):

    field = np.zeros(np.array(shape) + padding * 2)
    k = 2 * np.pi / wavelength * 1e-6
    x, y, K, C = _precalc(field, pixel_size * 1e6, k)
    return _precalc_Tz(k, z, K, C)


class Rescale(dt.Feature):
    """Rescales an optical field.

    Parameters
    ----------
    i : int
        index of z-propagator matrix
    """

    def __init__(self, rescale=1, **kwargs):
        super().__init__(rescale=rescale, **kwargs)

    def get(self, image, rescale, **kwargs):
        image = np.array(image)
        
        #image[..., 0] = (image[..., 0] - 1) * rescale + 1
        #image[..., 1] *= rescale
        im=image[...,0]+1j*image[...,1]
        im=(rescale*(np.abs(im)-1)+1)*np.exp(1j*np.angle(im))
        image[..., 0] = im.real#(image[..., 0] - 1) * rescale + 1
        image[..., 1] = im.imag#rescale
        #image[...,1]=np.angle(np.exp(1j*image[...,1]))

        return image
        

    

class RotateField(dt.Feature):
    def __init__(self, angle=0, **kwargs):
        super().__init__(angle=angle, **kwargs)

    def get(self, image, angle, **kwargs):
        image = np.array(image)
        im2=np.copy(image)
        im2[...,0]-=1
        im2[...,0]=np.cos(angle)*image[...,0]+np.sin(angle)*image[...,1]
        im2[...,1]=np.cos(angle)*image[...,1]-np.sin(angle)*image[...,0]
        image=im2
        image[...,0]+=1

        return image

    
class PhaseShift(dt.Feature):
    def __init__(self, angle=0, **kwargs):
        super().__init__(angle=angle, **kwargs)

    def get(self, image, angle, **kwargs):
        image = np.array(image)
        im2=np.copy(image)
        #im2[...,0]-=1
        im2[...,0]=np.cos(angle)*image[...,0]-np.sin(angle)*image[...,1]
        im2[...,1]=np.cos(angle)*image[...,1]+np.sin(angle)*image[...,0]
        image=im2
        #image[...,0]+=1

        return image
class RescalePhase(dt.Feature):
    """Rescales the phase of an optical field.

    Parameters
    ----------
    i : int
        index of z-propagator matrix
    """

    def __init__(self, phasescale=1, **kwargs):
        super().__init__(phasescale=phasescale, **kwargs)

    def get(self, image, phasescale, **kwargs):
        image = np.array(image)
        im=image[...,0]+1j*image[...,1]
        im=np.abs(im)*np.exp(1j*phasescale*np.angle(im))
        image[..., 0] = im.real#(image[..., 0] - 1) * rescale + 1
        image[..., 1] = im.imag#rescale
        #image[...,1]=np.angle(np.exp(1j*image[...,1]))

        return image

class FourierTransform(dt.Feature):
    """Creates matrices for propagating an optical field.

    Parameters
    ----------
    i : int
        index of z-propagator matrix
    """

    def __init__(self, padding=(32,32), **kwargs):
        super().__init__(padding=padding, **kwargs)

    def get(self, image, padding, **kwargs):

        im = np.copy(image[..., 0] + 1j * image[..., 1])
        
        im = np.pad(im, (padding, padding), mode="symmetric")
        f1 = np.fft.fft2(im)
        return f1


class InverseFourierTransform(dt.Feature):
    """Creates matrices for propagating an optical field.

    Parameters
    ----------
    i : int
        index of z-propagator matrix
    """

    def __init__(self, padding=(32,32), **kwargs):
        super().__init__(padding=padding, **kwargs)

    def get(self, image, padding, **kwargs):
        im = np.fft.ifft2(image)
        #if len(padding)==2:
        imnew = np.zeros(
            (image.shape[0] - padding * 2, image.shape[1] - padding * 2, 2)
        )

        imnew[..., 0] = np.real(im[padding:-padding, padding:-padding])
        imnew[..., 1] = np.imag(im[padding:-padding, padding:-padding])
        #elif len(padding)==3:
        #    imnew = np.zeros(
        #        (image.shape[0] - padding[0] * 2, image.shape[1] - padding[1] * 2,image.shape[2] - padding[2] * 2, 2)
        #    )

        #    imnew[..., 0] = np.real(im[padding[0]:-padding[0], padding[1]:-padding[1],padding[2]:-padding[2]])
        #    imnew[..., 1] = np.imag(im[padding[0]:-padding[0], padding[1]:-padding[1],padding[2]:-padding[2]])
        return imnew


class FourierTransformTransformation(dt.Feature):
    def __init__(self, Tz=1, Tzinv=1, i: dt.PropertyLike[int] = 0, **kwargs):
        super().__init__(Tz=Tz, Tzinv=Tzinv, i=i, **kwargs)

    def get(self, image, Tz, Tzinv, i, **kwargs):
        #if len(image.shape)==4:
        #    Tz=np.expand_dims(Tz,axis=0)
        #    Tzinv=np.expand_dims(Tzinv,axis=0)
            
        if i < 0:
            propfac=1
            for j in range(int(np.abs(i))):
                propfac*=Tzinv
            image *= propfac#Tzinv ** np.abs(i)
        else:
            propfac=1
            for j in range(int(np.abs(i))):
                propfac*=Tz
            image *= Tz ** i
        return image
