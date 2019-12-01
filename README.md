# EECS4422-Project

Code for the EECS 4422 - Computer Vision project. 

This project implements the edge detection model detailed in the paper by Xun Shi et al. <i>Early Recurrence Improves Edge Detection</i> [[1]](#references).

The paper proposes a new edge detection model inspired by research on biological vision in primates. The model is based on early recurrence and self-inhibition mechanisms found in the brains of priamtes. 


## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [References](#references)


### Prerequisites

Below is a list of dependencies and their installation instructions required for the project to run.

#### Python

This project uses Python v2.7. 

Please see (https://www.python.org/downloads/) for installation instructions.

#### OpenCV

```
$ pip install opencv-python
```
Please see (https://pypi.org/project/opencv-python/) for detailed installation instructions.

#### Numpy

```
$ python -m pip install --user numpy
```
Please see (https://numpy.org/devdocs/user/install.html) for detailed installation instructions.


### Installation

1. Download/clone the Python script to your local machine.
2. Import the script into your own python project.
```
import earlyRecurrenceEdgeDetection
```
3. Use the functions provided by the script. For an example usage see the [Usage](#usage) section.


### Usage

The project implements the core model from the paper <i>Early Recurrence Improves Edge Detection</i> (Note: the contour operator is not implemented yet). 

Each stage of the model's pipeline (see Figure 2 from [[1]](#references)) is implemented as a Python function. Each stage of the pipeline can be used indepenedently, or the final function which ties together the entire pipeline can be used.

Appropriate parameter values can be found from [[1]](#references).

```
import earlyRecurrenceEdgeDetection

img = cv2.imread("InputImage.jpg", 0)

n = 12
v1Psi = 0 
v1Gamma = 0.5 

kSizeVent = (7, 7) 
v1VentSigma = 2.8 
v1VentLambd = v1VentSigma/0.56 

kSizeDors = (33, 3) 
v1DorsSigma = 16.8
v1DrosLambd = v1DorsSigma/0.56

kSizeMT = (65, 65)
mtSigma = 32

inhibitionType = "anisotropic"

kSizeSelfInh = 129
selfInhSigma = 64

alpha = 0

outputEdgeMaps = recurrentInhibitionEdgeDetection(img, n, kSizeVent, v1VentSigma, v1VentLambd, kSizeDors, v1DorsSigma, v1DrosLambd, v1Gamma, v1Psi, kSizeMT, mtSigma, inhibitionType, kSizeSelfInh, selfInhSigma, alpha)
```

## Acknowledgments

* Calden Wloka - Provided guidance regarding the implementation of the model.

## References

[1] Shi, X., Wang, B., & Tsotsos, J. (2013). Early Recurrence Improves Edge Detection. Paper
presented at the British Machine Vision Conference, University of Bristol, England. 
