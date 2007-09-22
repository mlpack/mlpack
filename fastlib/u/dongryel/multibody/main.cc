#include "multibody.h"

int main(int argc, char *argv[])
{
  bool do_naive;
  double bandwidth;
  
  fx_init(argc, argv);

  // PARSE INPUTS
  do_naive = fx_param_exists(NULL, "do_naive");
  bandwidth = fx_param_double(NULL, "bandwidth", 0.5);

  // Multibody computation
  printf("Starting multitree multibody...\n");
  fx_timer_start(NULL, "multibody");
  MultitreeMultibody<GaussianKernel, GaussianKernelDerivative> mtmb;
  mtmb.Init(bandwidth);
  mtmb.Compute(0.1);
  fx_timer_stop(NULL, "multibody");
  printf("multitree multibody completed...\n");

  // NAIVE
  if (do_naive) {
    printf("Starting naive multibody...\n");
    fx_timer_start(NULL, "naive_multibody");
    NaiveMultibody<GaussianKernel> nmb;
    nmb.Init(mtmb.get_data(), bandwidth);
    nmb.Compute();
    fx_timer_stop(NULL, "naive_multibody");
    printf("finished naive multibody...\n");
  }

  fx_done();
}
