/** @file multibody_main.cc
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include "multibody.h"
#include "multibody_kernel.h"

typedef BinarySpaceTree<DHrectBound<2>, Matrix, MultibodyStat > Tree;

int main(int argc, char *argv[])
{
  bool do_naive;
  double relative_error;
  double bandwidth;
  const char *kernel;
  
  fx_init(argc, argv, NULL);

  // PARSE INPUTS
  do_naive = fx_param_exists(fx_root, "do_naive");
  bandwidth = fx_param_double(fx_root, "bandwidth", 0.1);
  relative_error = fx_param_double(fx_root, "relative_error", 0.1);
  kernel = fx_param_str(fx_root, "kernel", "axilrodteller");
  
  // Multibody computation
  printf("Starting multitree multibody...\n");

  if(!strcmp(kernel, "axilrodteller")) {
    fx_timer_start(fx_root, "multitree multibody");
    MultitreeMultibody<AxilrodTellerForceKernel<Tree, DHrectBound<2> >, Tree > mtmb;
    mtmb.Init(bandwidth);
    mtmb.Compute(relative_error);
    fx_timer_stop(fx_root, "multitree multibody");
    printf("Multitree multibody completed...\n");
    mtmb.PrintDebug(false);
    
    // Do naive algorithm if requested.
    if (do_naive) {

      // Get the approximated vectors from the multitree algorithm.
      Matrix approximated, exact;
      mtmb.get_force_vectors(&approximated);

      printf("Starting naive multibody...\n");
      fx_timer_start(fx_root, "naive_multibody");
      mtmb.NaiveCompute();
      fx_timer_stop(fx_root, "naive_multibody");
      printf("Finished naive multibody...\n");
      mtmb.PrintDebug(true);

      // Get the exact vectors from the naive algorithm.
      mtmb.get_force_vectors(&exact);
            
      double max_relative_l1_norm_error =
	MultitreeMultibody<AxilrodTellerForceKernel<Tree, DHrectBound<2> >, Tree >::MaxRelativeL1NormError(approximated, exact);

      // Compute the maximum L1 norm error and output the result.
      fx_format_result(fx_root, "maximum L1 norm error", "%g", 
		       max_relative_l1_norm_error);
    }
  }

  fx_done(NULL);
}
