# Kernel-KNN-and-KSPCA
Kernel k-nearest neighbor classifier and Kernel Principal Component Analysis

<b>Part 1:</b> You must to implement kernel k-nearest neighbor classifier. Recall that in 1NN 
classifier (𝑘 = 1), we just need to compute the Euclidean distance of a test instance to all the training 
instances, find the closest training instance, and find its label. The label of the nest instance will be 
identical to its nearest neighbor. This can be kernelized by observing that:
<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159655405-f5170007-0419-4fc2-bd41-9e22028309d0.png"
  >
</p>


This allows us to apply the nearest neighbor classifier to structured data objects.
Implement the KNN classifier and kernel KNN classifier with Linear, RBF (tune the 𝜎 parameter with 
cross-validation), Polynomial (𝑑 = 1), Polynomial (𝑑 = 2), and Polynomial (𝑑 = 3) kernels. Report the 
accuracy of classification for each data set with each classifier and compare the results. Split the data 
set into 70% and 30% for training and testing parts. You should report the mean of accuracies for 10 
individual runs. Report the running time of your code (in seconds) in the second table.

<b>Part 2:</b>
You should implement the Kernel Principal Component Analysis or KSPCA algorithm [1]. Implement the KSPCA algorithm according to the following pseudo-code.
<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159655927-99927a19-4790-49b5-bdf8-6416bb18728d.png"
  >
</p>

Consider data matrix ![image](https://user-images.githubusercontent.com/91370511/159656144-e08be6e1-5d99-4db8-b4de-e20f7f506aea.png) (𝑝 is the dimensionality of the data in original space and 𝑛 is the number of training samples) and labels 𝑌.
All you need is to compute delta kernel 𝐿, matrix 𝐻 (𝐻 = ![image](https://user-images.githubusercontent.com/91370511/159656252-2d7db591-3981-4b68-9bf7-fe65ec8ba404.png)
in which 𝑒 is a vector of all ones, ![image](https://user-images.githubusercontent.com/91370511/159664889-ecb2339b-8fb9-47f6-b3ed-8a126a8bf782.png), and kernel matrix 𝐾. 𝐼 is the identity matrix.
<br/>
▪ 𝐿 is a 𝑛 × 𝑛 delta kernel over 𝑌 (labels), compute 𝐿 with the following equation. 
<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159665394-34978afe-f394-4d75-9ef6-7443e847d507.png"
  >
</p>

For example, in a 2-class problem, if there are 5 data points such that the first 3 data points 
belong to class 1 and the fourth and the fifth data points are from class 2, 𝐿(𝑦, 𝑦′) can be
formed as follows:
<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159665553-5820ce0a-0d3f-458f-bc9b-c9402d7e6972.png"
  >
</p>

▪ 𝐾 is a 𝑛 × 𝑛 RBF kernel over samples in 𝑋𝑡𝑟𝑎𝑖𝑛, compute 𝐾 with the following equation.
<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/91370511/159665676-f541212f-da0f-403b-9b49-e638212949cb.png"
  >
</p>

▪ Split the data set into 70% and 30% for training and testing parts. 𝐾_𝑡𝑒𝑠𝑡 is a kernel over 𝑋_𝑡𝑟𝑎𝑖𝑛
and 𝑋_𝑡𝑒𝑠𝑡. For example, if we have 70 samples in 𝑋_𝑡𝑟𝑎𝑖𝑛 and 30 samples in 𝑋_𝑡𝑒𝑠𝑡, 𝐾_𝑡𝑒𝑠𝑡 is a 
matrix of 70 × 30, and 𝐾 is a matrix of 70 × 70 over 𝑋_𝑡𝑟𝑎𝑖𝑛. 
<br/>
▪ 𝛽 can be computed via eigs command in Matlab or eigh in Python (scipy.linalg).
<br/>
▪ Once you find the generalized eigenvectors of 𝑄 and 𝐾, select the first 2 columns (𝑑 = 2) and 
put them in 𝛽. Then you can find 𝑧 and 𝑍 with matrix multiplication. 𝑧 and 𝑍 are the test and 
train samples after projection. 
<br/>
▪ Provide scatter plots of all the datasets(in KSPCA folder) in original space and after projection.
In other words, you should plot 𝑋𝑡𝑟𝑎𝑖𝑛 and 𝑋𝑡𝑒𝑠𝑡 in one plot and 𝑧 and 𝑍 in another plot. Use 
different colors and symbols for different classes and different sets (train and test). You have 4 data sets, so you need to turn in 8 plots.
<br/>
▪ For 𝜎 in RBF kernel, you should select it from the {0.1, 0.2, 0.3, … ,1} set. Select the 𝜎 which 
has the best separation between classes (tune it empirically). 

