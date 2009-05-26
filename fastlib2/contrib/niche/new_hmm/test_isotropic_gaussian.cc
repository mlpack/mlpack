#include "isotropic_gaussian.h"

int main(int argc, char* argv[]) {

  IsotropicGaussian g;
  g.Init(2, 0.001);
  Vector &mu = *(g.mu_);
  mu[0] = -1;
  mu[1] = 0.5;

  g.sigma_ = 0.5;
  
  g.ComputeNormConstant();
  g.PrintDebug("g");
  
  Vector x;
  x.Init(2);
  x[0] = 2;
  x[1] = 1;
  printf("%e\n", g.Pdf(x));
}
