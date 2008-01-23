The files are the following:
1. mog_em_main.cc - this is the main which creates an object of the class MoG, initializes it, and then does parametric estimation on the data input using the L2 error criteria. The executable formed is called "mog_l2e_main".
 - parameters input to the main:
  --data : this is the file containing the data on which the mixture of gaussians is supposed to be fit.
  --mog_em/K : the number of gaussians you want to fit on the data

2. mog.h - this is the file that contains the definition of the class MoG.

3. mog.cc - this contains definition of some of the functions that are only declared in mog.h

4. phi.h - this contains the functions that calculates the value of the multivariate Gaussian PDF.

5. math_functions.h - this file contains functions that output the highest/lowest elements in an array and/or their indices.

6. build.py

7. The .arff and .csv file - the data files on which you can run the program.

-> An example run would be:
fl-build mog_em_main
./mog_em_main --data=data.arff --mog_em/K=3 

I have not written any test class for this because I could not think of any good way of doing it
But if you want to take a look at the program I have written to test individual functions, you can contact me personally :)
