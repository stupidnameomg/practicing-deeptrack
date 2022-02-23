import holography


propagate_up, propagate_down = holography.propagation_matrix([.05, -.05], \
    padding=32, wavelength=optics.wavelength(), pixel_size=optics.voxel_size()[0])

z=lambda: np.random.randint(0, 4)*1.0-np.random.randint(0,4)*1.0
sample = optics(particle)+holography.FourierTransform(padding=32)+holography.FourierTransformTransformation(propagate_down, propagate_up, z) + holography.InverseFourierTransform(padding=32)
