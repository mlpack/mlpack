#include "fastlib/fastlib.h"
#include "chebyshev_fit.h"
#include "three_body_gaussian_kernel.h"

int main(int argc, char *argv[]) {

  srand(time(NULL));

  ThreeBodyGaussianKernel kernel;
  ChebyshevFit fit;
  DRange range, values;
  kernel.Init(0.8);
  range.lo = 0;
  range.hi = 5;
  values.lo = kernel.EvalUnnorm(range.lo);
  values.hi = kernel.EvalUnnorm(range.hi);
  
  printf("%g %g %g %g\n", range.lo, range.hi, values.lo, values.hi);

  fit.Init(range, 20, &kernel);

  for(double j = range.lo; j <= range.hi; j += 0.05) {
    printf("%g %g\n", fit.Evaluate(j, 3), kernel.EvalUnnorm(j));
  }

  Vector taylor;
  Vector original_taylor;
  fit.ConvertToTaylorExpansion(3, &taylor);
  fit.ConvertToTaylorExpansionOriginalVariable(3, &original_taylor);
  original_taylor.PrintDebug();
  
  return 0;
}
