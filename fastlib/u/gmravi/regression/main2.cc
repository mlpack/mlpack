#include "fastlib/fastlib_int.h"
#include "regression2.h"
#define  MAXDOUBLE 32768.0

int main (int argc, char *argv[]){

  fx_init (argc, argv);
  Regression2 <GaussianKernel> reg2;
  printf("going to initialization function...\n");
  reg2.Init();
  printf("Initializations done..\n");
  //  reg2.Compute(fx_param_double (NULL, "tau", 0.1));
  reg2.Compute(0.1);
  ArrayList<Matrix> wfkde_results;
  wfkde_results.Copy(reg2.get_results());

  //NaiveRegression2 <GaussianKernel> naive_reg2;
  // naive_reg2.Init();
  //naive_reg2.Compute();
  //naive_reg2.ComputeMaximumRelativeError(wfkde_results);
  //printf("done..\n");
  fx_done();
}

