#include "fastlib/fastlib.h"
#include "distribution.h"
#include "gaussian.h"
#include "multinomial.h"



int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);

  Distribution* distributions[2];


  int d = 2;

  Vector mu_1;
  mu_1.Init(d);
  for(int i = 0; i < d; i++) {
    mu_1[i] = drand48();
  }

  Matrix Sigma_1;
  Sigma_1.Init(d, d);
  Sigma_1.SetZero();
  for(int i = 0; i < d; i++) {
    Sigma_1.set(i, i, 1);
  }

  Gaussian gaussian_1;
  gaussian_1.Init(mu_1, Sigma_1);

  distributions.Set(0, gaussian_1);

  int d_multinom;
  Vector p;
  p.Init(d_multinom);
  for(int i = 0; i < d_multinom; i++) {
    p[i] = drand48();
  }

  Multinomial* multinom_1 = (Multinomial*) malloc(sizeof(Multinomial));
  multinom_1 -> Init(&p);

  distributions[1] = multinom_1;

  free(gaussian_1);
  free(multinom_1);
  /*
  for(int i = 0; i < 2; i++) {
    free(distributions[i]);
  }
  */
  fx_done(root);

  return SUCCESS_PASS;
}
