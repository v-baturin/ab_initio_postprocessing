import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as spe
from .hydrogen import H_real_p

def genDensity(length, width, height, nl, nw, nh, l0, w0, h0, inFunc):
    '''
    Generate density mtrx // n,l,m set

    INPUT
        length: Positive extent in x of box
        width: Positive extent in y of box
        height: Positive extent in z of box
        nl: Number of elements along length
        nw: Number of elements along width
        nh: Number of elements along height
        l0: Negative extent in x of box
        w0: Negative extent in y of box
        h0: Negative extent in z of box
        inFunc: Function to generate density data

    OUTPUT
        3D Density mtrx
    '''

    # Quantum numbers
    n = 5
    l = 3
    m = 1

    # Initialize mtrx
    outMat = np.zeros((nl,nw,nh))

    # Calculate function at each grid point
    for lIndex in range(nl):

        # X-coordinate
        xVal = l0 + 1.0*lIndex*(length-l0)/(nl-1)

        for wIndex in range(nw):

            # Y-coordinate
            yVal = w0 + 1.0*wIndex*(width-w0)/(nw-1)

            for hIndex in range(nh):

                # Z-coordinate
                zVal = h0 + 1.0*hIndex*(height-h0)/(nh-1)

                # Translate to spherical coordinates
                r = np.sqrt(xVal**2 + yVal**2 + zVal**2)

                if (r == 0):
                    theta = 0
                else:
                    theta = np.arccos(zVal/r)

                if (xVal == 0):
                    phi = np.pi/2
                else:
                    phi = np.arctan(yVal/xVal)

                funcEval = inFunc(r,theta,phi,n,l,m)

                outMat[lIndex,wIndex,hIndex] = np.real(funcEval*np.conj(funcEval))

    return outMat

def writeDens(inMat,length, width, height, l0, w0, h0):
    '''
    Write xsf 3D density file

    INPUT
        inMat: 3D input mtrx
        length: Length of box
        width: Width of box
        height: Height of box
        l0: Zero coordinate for length
        w0: Zero coordinate for width
        h0: Zero coordinate for height

    OUTPUT
        Void. Creates xsf file

    * Note that I define the zero coordinate at the negative extent of the box - This is an arbitrary choice.
    '''

    # Get mtrx shape
    nl = inMat.shape[0]
    nw = inMat.shape[1]
    nh = inMat.shape[2]

    # Open the file for writing
    with open('testDensity_H.xsf','w') as outFile:
        # Write header
        outFile.write('BEGIN_BLOCK_DATAGRID_3D\n')
        outFile.write('\tHydrogen Density\n')
        outFile.write('\tBEGIN_DATAGRID_3D\n')

        # Write cell information
        outFile.write('\t\t' + str(nl) + ' ' + str(nw) + ' ' + str(nh) + '\n')
        outFile.write('\t\t' + str(l0) + ' ' + str(w0) + ' ' + str(h0) + '\n')
        outFile.write('\t\t' + str(length-l0) + ' ' + str(0.0) + ' ' + str(0.0) + '\n')
        outFile.write('\t\t' + str(0.0) + ' ' + str(width-w0) + ' ' + str(0.0) + '\n')
        outFile.write('\t\t' + str(0.0) + ' ' + str(0.0) + ' ' + str(height-h0) + '\n')

        outStr = ''

        # Write datagrid
        for hIndex in range(nh):
            for wIndex in range(nw):
                for lIndex in range(nl):
                    outStr += str(inMat[lIndex,wIndex,hIndex]) + ' '

                outFile.write('\t\t\t' + outStr + '\n')
                outStr = ''

        # Write footer
        outFile.write('\tEND_DATAGRID_3D\n')
        outFile.write('END_BLOCK_DATAGRID_3D\n')