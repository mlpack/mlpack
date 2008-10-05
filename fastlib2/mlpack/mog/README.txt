Author: Parikshit Ram (pram@cc.gatech.edu)

The files are the following:
1. mog_em_main.cc - this is the main which creates an object of the class MoGEM, initializes it, and then does parametric estimation on the data input using the L2 error criteria. The executable formed is called "mog_em_main".
 - parameters input to the main:
  --data : this is the file containing the data on which the mixture of gaussians is supposed to be fit.
  --mog_em/K : the number of gaussians you want to fit on the data

2. mog_em.h - this is the file that contains the definition of the class MoGEM and the function tht implements the EM algorithm.

3. mog_em.cc - this contains definition of some of the functions that are only declared in mog_em.h

4. mog_l2e_main.cc - this is the main which creates an object of the class MoGL2E, initializes it, and then does parametric estimation on the data input using the L2 error criteria. The executable formed is called "mog_l2e_main".
 - parameters input to the main:
  --data : this is the file containing the data on which the mixture of gaussians is supposed to be fit.
  --mog_l2e/K : the number of gaussians you want to fit on the data
  --opt/method : this lets the user choose which optimizer he/she wants to use. It is 'NelderMead' for the polytope method, otherwise it defaults to 'QuasiNewton' for the quasi newton method.
  --output : the file into which the estimated parameters of the gaussian mixture are written into, defaults to "output.csv" (the output is in the "pretty-print" format)

5. mog_l2e.h - this is the file that contains the definition of the class MoGL2E and computes the L2 error of the present model.

6. mog_l2e.cc - this contains definition of some of the functions that are only declared in mog.h

7. phi.h - this contains the functions that calculates the value of the multivariate Gaussian PDF, and also the gradients of the gaussian PDF with respect to the mean and variance when d(sigma) is provided.

8. math_functions.h - this file contains functions that output the highest/lowest elements in an array and/or their indices.

9. build.py

10. The .arff file - the data files on which you can run the program.

-> An example run would be:
fl-build mog_em_main
./mog_em_main --data=fake.arff --mog_em/K=3 

fl-build mog_l2e_main
./mog_l2e_main --data=fake.arff --mog_l2e/K=3 --opt/method=NelderMead
OR
./mog_l2e_main --data=fake.arff --mog_l2e/K=3 --opt/method=QuasiNewton
