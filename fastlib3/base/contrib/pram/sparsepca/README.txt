Author: Parikshit Ram (pram@cc.gatech.edu)

The files are the following:
1. spca_main.cc - this is the main which creates an object of the class SparsePCA, initializes it, and then sparsifies the ordinary principal component loadings.
 - parameters input to the main:
  --data : this is the file containing the data on which the dimensionality reduction is to be performed.
  --spca/K : the number of principal components to be considered
  --output : the filename in which you want the results to be output

2. sparsepca.h - this is the file that contains the definition of the class SparsePCA.

3. sparsepca.cc - this contains definition of some of the functions that are only declared in sparsepca.h

4. build.py

5. The .csv file - the data file on which you can test run the program.

-> An example run would be:
fl-build spca_main
./spca_main --data=data_sparse_pca.csv --spca/K=2 

