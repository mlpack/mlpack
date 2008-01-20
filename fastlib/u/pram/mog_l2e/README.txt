The files are the following:
1. mog_l2e_main.cc - this is the main which creates an object of the class MoG, initializes it, and then does parametric estimation on the data input using the L2 error criteria. The executable formed is called "mog_l2e_main".
 - parameters input to the main:
  --data : this is the file containing the data on which the mixture of gaussians is supposed to be fit.
  --number_of_gaussians : the number of gaussians you want to fit on the data
  --optim_flag : this lets the user choose which optimizer he/she wants to use. It is '1' for the polytope method, otherwise it defaults to the quasi newton method.
  --output_filename : the file into which the estimated parameters of the gaussian mixture are written into, defaults to "output.csv" (the output is in the "pretty-print" format)

2. mog.h - this is the file that contains the definition of the class MoG.

3. mog.cc - this contains definition of some of the functions that are only declared in mog.h

4. phi.h - this contains the functions that calculates the value of the multivariate Gaussian PDF, and also the gradients of the gaussian PDF with respect to the mean and variance when d(sigma) is provided.

5. math_functions.h - this file contains functions that output the highest/lowest elements in an array and/or their indices.

6. l2_error.h - this file is used by a function in the class MoG to compute the value of the L2 error and/or its gradients, and also contains the implementation of the polytope and the quasi newton optimizer on the L2 error calculating function

7. build.py

8. The .arff and .csv file - the data files on which you can run the program.

-> An example run would be:
fl-build mog_l2e_main
./mog_l2e_main --data=data.arff --number_of_gaussians=3 --optim_flag=1

I have not written any test class for this because I could not think of any good way of doing it
But if you want to take a look at the program I have written to test individual functions, you can contact me personally :)
