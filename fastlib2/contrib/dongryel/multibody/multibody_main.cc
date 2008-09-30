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
  double relative_error, threshold, probability, coverage_percentile;
  double bandwidth;
  const char *kernel;
  
  fx_init(argc, argv, NULL);

  // PARSE INPUTS
  do_naive = fx_param_exists(fx_root, "do_naive");
  bandwidth = fx_param_double(fx_root, "bandwidth", 0.1);
  relative_error = fx_param_double(fx_root, "relative_error", 0.1);
  threshold = fx_param_double(fx_root, "threshold", 0);
  coverage_percentile = 
    fx_param_double(fx_root, "coverage_percentile", 100);
  probability = fx_param_double(fx_root, "probability", 1);
  kernel = fx_param_str(fx_root, "kernel", "axilrodteller");
  
  // Multibody computation
  printf("Starting multitree multibody...\n");

  if(!strcmp(kernel, "axilrodteller")) {
    fx_timer_start(fx_root, "multitree multibody");
    MultitreeMultibody<AxilrodTellerForceKernel<Tree, DHrectBound<2> >, Tree> mtmb;
    mtmb.Init(bandwidth);
    mtmb.Compute(relative_error, threshold, coverage_percentile, probability);
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
      
      // Compute the net force from the naively computed vectors.
      Vector naive_net_force;
      naive_net_force.Init(exact.n_rows());
      naive_net_force.SetZero();
      for(index_t i = 0; i < exact.n_cols(); i++) {
	for(index_t d = 0; d < exact.n_rows(); d++) {
	  naive_net_force[d] += exact.get(d, i);
	}
      }
      printf("Naive net force:\n");
      for(index_t i = 0; i < naive_net_force.length(); i++) {
	printf("%g ", naive_net_force[i]);
      }
      printf("\n");
            
      double max_relative_l1_norm_error;
      int relative_error_under_threshold;
      double max_absolute_l1_norm_error;
      int absolute_error_under_threshold;

      MultitreeMultibody<AxilrodTellerForceKernel<Tree, DHrectBound<2> >, Tree >::
	MaxL1NormError(approximated, exact, &max_relative_l1_norm_error,
		       &relative_error_under_threshold,
		       &max_absolute_l1_norm_error,
		       &absolute_error_under_threshold, relative_error, 
		       threshold);

      // Compute the maximum L1 norm error and output the result.
      fx_format_result(fx_root, "maximum relative L1 norm error", "%g", 
		       max_relative_l1_norm_error);
      fx_format_result(fx_root, "relative error under threshold", "%d",
		       relative_error_under_threshold);
      fx_format_result(fx_root, "maximum absolute L1 norm error", "%g",
		       max_absolute_l1_norm_error);
      fx_format_result(fx_root, "absolute error under threshold", "%d",
		       absolute_error_under_threshold);
    }
  }

  fx_done(NULL);
}
