/** @file multibody_main.cc
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include "multibody.h"
#include "multibody_kernel.h"
#include "naive_multibody.h"

int main(int argc, char *argv[])
{
  bool do_naive;
  double tau;
  double bandwidth;
  const char *kernel;
  
  fx_init(argc, argv, NULL);

  // PARSE INPUTS
  do_naive = fx_param_exists(NULL, "do_naive");
  bandwidth = fx_param_double(NULL, "bandwidth", 0.1);
  tau = fx_param_double(NULL, "tau", 0.1);
  kernel = fx_param_str(NULL, "kernel", "gaussianthreebody");
  
  // Multibody computation
  printf("Starting multitree multibody...\n");

  if(!strcmp(kernel, "gaussianthreebody")) {
    fx_timer_start(NULL, "multibody");
    MultitreeMultibody<GaussianThreeBodyKernel, GaussianKernelAux> mtmb;
    mtmb.Init(bandwidth);
    mtmb.Compute(tau);
    fx_timer_stop(NULL, "multibody");
    printf("multitree multibody completed...\n");
    
    // NAIVE
    if (do_naive) {
      printf("Starting naive multibody...\n");
      fx_timer_start(NULL, "naive_multibody");
      NaiveMultibody<GaussianThreeBodyKernel> nmb;
      nmb.Init(mtmb.get_data(), bandwidth);
      nmb.Compute();
      fx_timer_stop(NULL, "naive_multibody");
      printf("finished naive multibody...\n");
    }
  }
  else if(!strcmp(kernel, "axilrodteller")) {
    fx_timer_start(NULL, "multibody");
    MultitreeMultibody<AxilrodTellerKernel, GaussianKernelAux> mtmb;
    mtmb.Init(bandwidth);
    mtmb.Compute(tau);
    fx_timer_stop(NULL, "multibody");
    printf("multitree multibody completed...\n");
    
    // NAIVE
    if (do_naive) {
      printf("Starting naive multibody...\n");
      fx_timer_start(NULL, "naive_multibody");
      NaiveMultibody<AxilrodTellerKernel> nmb;
      nmb.Init(mtmb.get_data(), bandwidth);
      nmb.Compute();
      fx_timer_stop(NULL, "naive_multibody");
      printf("finished naive multibody...\n");
    }
  }

  fx_done(NULL);
}
