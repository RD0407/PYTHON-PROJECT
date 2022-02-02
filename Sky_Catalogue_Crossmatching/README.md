# Project 3 - Crossmatching astronomical catalogues
## Aim : To crossmatch astronomical catalogs.
We create an algorithm to automatically determine offsets between the exposures in x- and y-directions".
## Steps Involved:

### 1. Analysis Task
#### Task 1 : Implementing the algorithm

   * We write a Python program to implement the algorithm above and to estimate the $x-$ and $y-$offsets between the object patterns in catalogues [data/image013269.asc](data/image013269.asc) and [data/image013271.asc](data/image013271.asc).
   * The program should plot the two histograms of the x- and y- distances and print the estimated offsets.
#### Task 2: Estimate shifts between catalogues.
We have six catalogues and we want to estimate shifts between *all* of them. Having the algorithm for pairwise shifts (Task 2), one could estimate shifts between three catalogues `c_1.asc`, `c_2.asc` and `c_3.asc` in the following two ways:
   * Obtain the distances between `c_1.asc` and `c_3.asc`using Task 1. Similarily, the distance between `c_1.asc` and `c_2.asc` is calculated. Then the distance between `c_2.asc` and `c_3.asc` is just difference between the two.
   * Perform the same steps as above and at the same time estimate the distance between `c_1.asc` and `c_2.asc with Task 1 as well.
   * Perform this analysis for several combinations of the catalogues `image013269.asc .. image013274.asc` and investigate if we obtain different results for the distance between `c_2.asc` and `c_3.asc` with both methods.
#### Task 3: Improvement of Task 2:
Task 2 shows us that a pure pairwise analysis to obtain consistent offset values between more than two catalogues is not optimal. So, in this task, we want to estimate the shifts taking into account all available data points simultaneously.
   * Solve the offset problem for the six catalogues `image013269.asc .. image013274.asc`
