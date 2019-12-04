import cv2
import numpy as np
import math as m

# Creates a bank of Gabor filters with 'n' equally spaced orientations
def gaborBank(n, kSize, sigma, lambd, gamma, psi):

    # Initializes an array for the Gabor bank
    gaborBank = []
    
    # Gets the Gabor kernels at each orientation and adds them to the gaborBank array
    for theta in range(0,180,int(180/n)):

        # Converts degrees into radians since cv2.getGaborKernel requires radian input
        theta = m.radians(theta)

        # Retrieves the Gabor kernel with specified parameters
        gaborV1 = cv2.getGaborKernel(kSize, sigma, theta, lambd, gamma, psi, cv2.CV_32F)

        # Normalizes the kernal to sum to zero
        gaborV1 = gaborV1 - (np.sum(gaborV1)/(kSize[0]**2))

        # Flips the kernel vertically and horizontally for use in the cv2.filter2D function as a convolution
        gaborV1FlippedLeft = np.fliplr(gaborV1)
        gaborV1FlippedUp = np.flipud(gaborV1FlippedLeft)


        gaborBank.append(gaborV1FlippedUp)

    return gaborBank

# Filters and image using a bank of filters
def filterWithBank(img, bank):
    filteredImages = []

    for i in range(len(bank)):
        filteredImage = cv2.filter2D(img, -1, bank[i])
        filteredImages.append(filteredImage)

    return filteredImages


# Calculates the response of the Primary Visual Cortex (V1) to an input image
def v1Response(img, n, kSize, sigma, lambd, gamma, psi):
    # Creates the bank of Gabor filters
    gaborV1 = gaborBank(n, kSize, sigma, lambd, gamma, psi)

    # Creates the bank of Gabor filters phase shifted by pi/2
    gaborV1ShiftedPsi = gaborBank(n, kSize, sigma, lambd, gamma, psi + m.pi/2)

    # Filters the input image with the Gabor bank
    v1Response = filterWithBank(img, gaborV1)

    # Filters the input image with the phase shifted Gabor bank
    v1ResponseShiftedPsi = filterWithBank(img, gaborV1ShiftedPsi)

    quadratureV1 = []

    # Integrates the non-phase shifted and phase shifted responses using quadrature integration
    for i in range(len(v1Response)):
        quadratureV1Image = np.sqrt(np.square(v1Response[i]) + np.square(v1ResponseShiftedPsi[i]))
        quadratureV1Image = np.float32(quadratureV1Image)
        quadratureV1.append(quadratureV1Image)

    # Returns the integrated response
    return quadratureV1


# Calculates the response from MT integration
def mtIntegration(v1Response, kSize, sigma):
    mtResponse = []

    # Blurs each response from V1 with a Gaussian kernel as per the model of the MT area
    for i in range(len(v1Response)):
        mtResponseImage = cv2.GaussianBlur(v1Response[i], kSize, sigma) 
        mtResponse.append(mtResponseImage)

    # Returns the MT integrated resopnse
    return mtResponse


# Calculates the inhibition factor from the dorsal stream responses
def inhibitionFactor(mtIntegration, inhibitionType):

    # An intermediate matrix for accumulating the responses from the MT area
    tempMat = np.zeros((np.shape(mtIntegration)[1], np.shape(mtIntegration)[2]))

    # Calculates the sum of MT responses
    for i in range(len(mtIntegration)):
        tempMat = tempMat + mtIntegration[i]

    # Calculates the L1 norm of the sum of MT responses
    l1Norm = l1Normalize(tempMat)

    # Selects the isotropic inhibition scheme
    if (inhibitionType == "isotropic"):
        isoInhibition = []

        # Constructs the array of inhibition factors
        for i in range(len(mtIntegration)):
            isoInhibition.append(tempMat)

        # Normalizes each response
        for i in range(len(isoInhibition)):
            isoInhibition[i] = isoInhibition[i]/l1Norm

        return isoInhibition
    # Selects the anisotropic inhibition scheme
    elif (inhibitionType == "anisotropic"):

        # Normalizes each filter orientation according to the anisotropic scheme
        for i in range(len(mtIntegration)):
            l1NormAniso = l1Normalize(mtIntegration[i])
            mtIntegration[i] = mtIntegration[i]/l1NormAniso

        return mtIntegration


# Calcualtes a rectified difference of Gaussians function
def rectifiedDiffOfGaussian(kSize, sigma):

    # Calculates the difference of Guassians 
    oneSigma = cv2.getGaussianKernel(kSize, sigma)

    oneSigma = np.outer(oneSigma, oneSigma)

    fourSigma = cv2.getGaussianKernel(kSize, 4*sigma)

    fourSigma = np.outer(fourSigma, fourSigma)

    diffOfGaus = fourSigma - oneSigma

    # Rectifies the difference of Gaussians
    for i in range(kSize):
        for j in range(kSize):
            if (diffOfGaus[i][j] < 0):
                diffOfGaus[i][j] = 0
    
    return diffOfGaus


# Caclulates the L1 norm of a matrix
def l1Normalize(mat):
    l1NormColumns = []
    for k in range(np.shape(mat)[0]):
        sum = 0
        for j in range(np.shape(mat)[1]):
            sum = sum + abs(mat[j][k])
        l1NormColumns.append(sum)
        
    l1Norm = max(l1NormColumns)

    return l1Norm


# Calculates the centre-surround self-inhibition factor
def selfInhibitionFactor(v1Response, kSize, sigma):

    # Retrieves a difference of Gaussians
    DoG = rectifiedDiffOfGaussian(kSize, sigma)

    # Calculates the L1 norm of the difference of Gaussians
    l1Norm = l1Normalize(DoG)

    # Normalizes the difference of Gaussians
    DoG = DoG/l1Norm

    # Flips the difference of Gaussians vertically and horizontally for use in the cv2.filter2D function as a convolution
    DoGFlippedLeft = np.fliplr(DoG)
    DoGFlippedUp = np.flipud(DoGFlippedLeft)

    # Finds the dimensions of the V1 responses
    [array, row, col] = np.shape(v1Response)

    # Creates a matrix to hold the maximum pixel values across the different orientations of the V1 responses
    maxResp = np.zeros((row, col))
    
    # Finds the maximum pixel values for the maximum Gabor energy
    for i in range(row):
        for j in range(col):
            maxPx = 0
            for k in range(len(v1Response)):
                if (v1Response[k][i][j] > maxPx):
                    maxPx = v1Response[k][i][j]
            maxResp[i][j] = maxPx
                    
    
    # Convolves the normalized difference of Gaussians with the maximum Gabor energy
    selfInh = cv2.filter2D(maxResp, -1, DoGFlippedUp)

    return selfInh



def recurrentInhibitionEdgeDetection(img, n, kSizeVent, v1VentSigma, v1VentLambd, kSizeDors, v1DorsSigma, v1DrosLambd, v1Gamma, v1Psi, kSizeMT, mtSigma, inhibitionType, kSizeSelfInh, selfInhSigma, alpha):

    outputV1Dors = v1Response(img, n, kSizeDors, v1DorsSigma, v1DrosLambd, v1Gamma, v1Psi)

    outputV1Vent = v1Response(img, n, kSizeVent, v1VentSigma, v1VentLambd, v1Gamma, v1Psi)

    mtIntegrationDors = mtIntegration(outputV1Dors, kSizeMT, mtSigma)

    isoInh = inhibitionFactor(mtIntegrationDors, inhibitionType)

    selfInh = selfInhibitionFactor(outputV1Vent, kSizeSelfInh, selfInhSigma)

    output = []

    for i in range(len(isoInh)):
        temp = 5000*(outputV1Vent[i] * isoInh[i]) - (alpha*selfInh)

        output.append(temp)

    return output
