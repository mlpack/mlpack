#include "multibody.h"

int main(int argc, char *argv[])
{
  int leaflen;
  bool do_naive;

  fx_init(argc, argv);

  // PARSE INPUTS
  leaflen = fx_param_int(NULL, "leaflen", 20);
  do_naive = fx_param_exists(NULL, "do_naive");

  // Multibody computation
  fx_timer_start(NULL,"multibody");
  MultitreeMultibody<GaussianKernel, GaussianKernelDerivative> mtmb;
  mtmb.Init(0.1);
  mtmb.Compute(0.1);
  fx_timer_stop(NULL, "multibody");

  // NAIVE
  if (do_naive) {
  }

  fx_done();
}
