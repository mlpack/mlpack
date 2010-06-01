#include "fastlib/fastlib.h"
#include "gaussian.h"
#include "mmk.h"


int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);

  MeanMapKernel mmk;
  mmk.Init(1);


  Vector mu_1, mu_2;

  Matrix Sigma_1, Sigma_2;

  int d = 1;
  
  mu_1.Init(d);
  mu_2.Init(d);

  Sigma_1.Init(d, d);
  Sigma_2.Init(d, d);

  Sigma_1.SetZero();
  Sigma_2.SetZero();
  
  for(int i = 0; i < d; i++) {
    mu_1[i] = drand48();
    mu_2[i] = mu_1[i];

    Sigma_1.set(i, i, 1);
    Sigma_2.set(i, i, 1);
  }
  
  mu_1.PrintDebug("mu_1");
  mu_2.PrintDebug("mu_2");

  Sigma_1.PrintDebug("Sigma_1");
  Sigma_2.PrintDebug("Sigma_2");

  

  Gaussian gaussian_1;
  gaussian_1.Init(mu_1, Sigma_1);
  
  Gaussian gaussian_2;
  gaussian_2.Init(mu_2, Sigma_2);
  
  printf("k(gaussian_1, gaussian_2) = %f\n",
	 mmk.Compute(gaussian_1, gaussian_2));

  printf("\n\n\n\n");

  fx_done(root);
  

  return SUCCESS_PASS;
}
