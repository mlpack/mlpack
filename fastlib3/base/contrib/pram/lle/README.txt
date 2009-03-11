Author: Parikshit Ram (pram@cc.gatech.edu)

The files are the following:
1. lle_main.cc - this is the main which creates an object of the class LLE, initializes it, and then performs nonlinear dimensionality reduction on the dataset by locally linear embedding.
 - parameters input to the main:
  --data : this is the file containing the data on which the nonlinear dimensionality reduction is to be performed.
  --lle/knn : number of nearest neighbors required to obtain the weights for the locally linear embeddings
  --lle/d : this is the dimension to which the data is to be mapped
  --output : the filename in which you want the results to be output

2. lle.h - this is the file that contains the definition of the class LLE.

3. build.py

5. The .csv file - the data file on which you can test run the program.

-> An example run would be:
fl-build lle_main
./lle_main --data=data_1000_fastlib.csv --lle/knn=12 --lle/d=2 --output=output.csv 

