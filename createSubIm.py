"""
Functions to be used together with the DeepTrack-package and associated functions. The function 'cropParticles' 
takes an image with several particles in it and outputs a list of sub-images containing only one particle each. 

"""

from scipy.spatial import distance
import numpy as np

def get_particle_position(image_of_particles):

    position = []
    for property in image_of_particles.properties:
        if "position" in property:
            position.append(property["position"])
    
    positionArr = np.concatenate(position)
    positionArr = positionArr.reshape(-1, 2)
    return positionArr

def cropParticles(image_of_particles, outSize):

    # (Row, Col) position of particles
    positions = get_particle_position(image_of_particles)

    # We'll check if two particles are placed too closed to each other and not use sub-iamges of them
    safetyDist = np.sqrt(outSize[0]**2 + outSize[1]**2)/2
    distBetweenPts = distance.cdist(positions, positions).reshape(1,-1)
    distBetweenPts[distBetweenPts < safetyDist] = 0
    distBetweenPts = distBetweenPts.reshape(positions.shape[0], -1)
    
    accPosInd = []
    for i in range(distBetweenPts.shape[0]):
        if (distBetweenPts.shape[0] - np.count_nonzero(distBetweenPts[i, :])) == 1:
            accPosInd.append(i)

    accPos = positions[accPosInd, :]

    subIm = []
    for subCor in accPos:
        subRow = np.arange(round(subCor[0] - outSize[0]/2), round(subCor[0] + outSize[0]/2))
        subCol = np.arange(round(subCor[1] - outSize[1]/2), round(subCor[1] + outSize[1]/2))

        if subRow[-1] > image_of_particles.shape[0]:
            subRow[-1] = image_of_particles.shape[0]
        if subCol[-1] > image_of_particles.shape[1]:
            subCol[-1] = image_of_particles.shape[1]
        
        subIm.append(image_of_particles[subRow[0]:subRow[-1], subCol[0]:subCol[-1]])

    # We'll return a list of the sub-images
    return subIm