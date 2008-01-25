Author : Parikshit Ram (pram@cc.gatech.edu)

The files are the following:
1. mog_l2e_main.cc - this is the main which creates an object of the class MoG, initializes it, and then does parametric estimation on the data input using the L2 error criteria. The executable formed is called "mog_l2e_main".
 - parameters input to the main:
  --data : this is the file containing the data on which the mixture of gaussians is supposed to be fit.
  --mog_l2e/K : the number of gaussians you want to fit on the data
  --opt/method : this lets the user choose which optimizer he/she wants to use. It is 'NelderMead' for the polytope method, otherwise it defaults to 'QuasiNewton' for the quasi newton method.
  --output : the file into which the estimated parameters of the gaussian mixture are written into, defaults to "output.csv" (the output is in the "pretty-print" format)

2. mog.h - this is the file that contains the definition of the class MoG.

3. mog.cc - this contains definition of some of the functions that are only declared in mog.h

4. phi.h - this contains the functions that calculates the value of the multivariate Gaussian PDF, and also the gradients of the gaussian PDF with respect to the mean and variance when d(sigma) is provided.

5. optimizers.h - this file contains the implementation of two optimizer
classes

6. build.py

7. The .arff and .csv file - the data files on which you can run the program.

-> An example run would be:
fl-build mog_l2e_main
./mog_l2e_main --data=fake.arff --mog_l2e/K=3 --opt/method=NelderMead
OR
./mog_l2e_main --data=fake.arff --mog_l2e/K=3 --opt/method=QuasiNewton

