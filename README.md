# Kernel KNN and Kernel SPCA

This project implements two kernel-based algorithms: the **Kernel k-Nearest Neighbor (KNN) classifier** and **Kernel Principal Component Analysis (KSPCA)**, demonstrating their application in classification and dimensionality reduction.

---

## Part 1: Kernel k-Nearest Neighbor Classifier

The classic 1-Nearest Neighbor (1NN) classifier assigns the label of the closest training instance to a given test instance based on Euclidean distance. This approach can be kernelized by expressing distances in terms of kernel functions, allowing its application to structured data.

<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159655405-f5170007-0419-4fc2-bd41-9e22028309d0.png"
    alt="Kernelized 1NN formula"
  >
</p>

### Implementation Details:

- Implemented both the standard KNN classifier and kernelized KNN classifier with the following kernels:
  - Linear kernel  
  - Radial Basis Function (RBF) kernel (with \(\sigma\) tuned via cross-validation)  
  - Polynomial kernels of degrees \(d = 1, 2, 3\)  
- Datasets were split into 70% training and 30% testing sets.  
- Reported the mean classification accuracy over 10 independent runs for each classifier and dataset.  
- Measured and reported the running time (in seconds) for each method.

---

## Part 2: Kernel Principal Component Analysis (KSPCA)

KSPCA extends classical PCA by projecting data into a high-dimensional feature space defined implicitly by a kernel function, capturing nonlinear structures.

The following pseudo-code outlines the KSPCA algorithm:

<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159655927-99927a19-4790-49b5-bdf8-6416bb18728d.png"
    alt="KSPCA pseudo-code"
  >
</p>

### Key Components:

- Data matrix \(X \in \mathbb{R}^{p \times n}\) where \(p\) is the original feature dimension and \(n\) is the number of training samples.  
- Labels vector \(Y\).  
- Delta kernel matrix \(L\) computed over labels using:  

<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159665394-34978afe-f394-4d75-9ef6-7443e847d507.png"
    alt="Delta kernel L"
  >
</p>

- Kernel matrix \(K\) over training samples computed using the RBF kernel:  

<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159665676-f541212f-da0f-403b-9b49-e638212949cb.png"
    alt="RBF kernel K"
  >
</p>

- \(H = I - \frac{1}{n} ee^T\), where \(e\) is a vector of ones and \(I\) the identity matrix, used for centering.  
- Eigen-decomposition of generalized eigenvalue problem to obtain projection vectors.  
- Selected the first two eigenvectors for dimensionality reduction (\(d=2\)).  

### Procedure:

- Split dataset into 70% training and 30% testing.  
- Compute kernel matrices \(K\) and \(K_{\text{test}}\) accordingly.  
- Tune RBF kernel parameter \(\sigma\) over \(\{0.1, 0.2, ..., 1.0\}\) to optimize class separation.  
- Project both training and test data to 2D space using the eigenvectors.  
- Generated scatter plots for all four datasets, showing:
  - Original data (train and test combined)  
  - Projected data after KSPCA  
- Different colors and markers represent different classes and data splits. A total of 8 plots (4 datasets × 2 plots each) were produced.

<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159665553-5820ce0a-0d3f-458f-bc9b-c9402d7e6972.png"
    alt="Example delta kernel matrix"
  >
</p>

