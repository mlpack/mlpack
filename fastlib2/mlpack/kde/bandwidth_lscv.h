/** @file bandwidth_lscv.h
 *
 *  @author Dongryeol Lee (dongryel)
 *
 *  TO-DO: To make this cross-validation a general-purpose tool.
 */

#ifndef BANDWIDTH_LSCV_H
#define BANDWIDTH_LSCV_H

#include "dualtree_kde.h"

class BandwidthLSCV {
  
 private:

  static double plugin_bandwidth_(const Matrix &references) {
    
    double avg_sdev = 0;
    int num_dims = references.n_rows();
    int num_data = references.n_cols();
    Vector mean_vector;
    mean_vector.Init(references.n_rows());
    mean_vector.SetZero();
    
    // First compute the mean vector.
    for(index_t i = 0; i < references.n_cols(); i++) {
      for(index_t j = 0; j < references.n_rows(); j++) {
	mean_vector[j] += references.get(j, i);
      }
    }
    la::Scale(1.0 / ((double) num_data), &mean_vector);
    
    // Loop over the dataset again and compute variance along each
    // dimension.
    for(index_t i = 0; i < num_data; i++) {
      for(index_t j = 0; j < num_dims; j++) {
	avg_sdev += math::Sqr(references.get(j, i) - mean_vector[j]);
      }
    }
    avg_sdev /= ((double) (num_data - 1) * num_dims);

    double plugin_bw = 
      pow((4.0 / (num_dims + 2.0)), 1.0 / (num_dims + 4.0)) * avg_sdev * 
      pow(num_data, -1.0 / (num_dims + 4.0));

    return plugin_bw;
  }

  template<typename TKernel>
  static double lscv_score_
  (const Vector &density_estimates_on_bandwidth,
   const Vector &density_estimates_on_two_times_bandwidth,
   const TKernel &kernel, int num_dims) {
    
    double lscv_score = 0;
    double tmp_lscv_score = 0;

    // M_1(h) score on page 50 of Density Estimation for Statistics
    // and Data Analysis by B. W. Silverman.
    for(index_t i = 0; i < density_estimates_on_bandwidth.length(); i++) {
      lscv_score += density_estimates_on_two_times_bandwidth[i];
      tmp_lscv_score += density_estimates_on_bandwidth[i];
    }
    lscv_score -= 2 * tmp_lscv_score;
    lscv_score += 2 * kernel.EvalUnnormOnSq(0) /
      kernel.CalcNormConstant(num_dims);

    // Normalize score by dividing the number of points.
    lscv_score /= ((double) density_estimates_on_bandwidth.length());

    return lscv_score;
  }
  
 public:

  template<typename TKernelAux>
  static void ComputeLSCVScore(const Matrix &references,
			       const Matrix &reference_weights,
			       double bandwidth) {

    // Get the parameters.
    struct datanode *kde_module = fx_submodule(fx_root, "kde");

    // Kernel object.
    typename TKernelAux::TKernel kernel;
    
    // Set the bandwidth of the kernel.
    kernel.Init(bandwidth);
    
    printf("Trying the bandwidth value of %g...\n", bandwidth);
    
    // Need to run density estimates twice: on $h$ and $sqrt(2)
    // h$. Free memory after each run to minimize memory usage.
    fx_set_param_double(kde_module, "bandwidth", bandwidth);
    DualtreeKde<TKernelAux> *fast_kde_on_bandwidth = 
      new DualtreeKde<TKernelAux>();
    Vector results_on_bandwidth;
    fast_kde_on_bandwidth->Init(references, references, reference_weights,
				true, kde_module);
    fast_kde_on_bandwidth->Compute(&results_on_bandwidth);
    delete fast_kde_on_bandwidth;
    
    fx_set_param_double(kde_module, "bandwidth", bandwidth * sqrt(2));
    DualtreeKde<TKernelAux> *fast_kde_on_two_times_bandwidth =
      new DualtreeKde<TKernelAux>();
    Vector results_on_two_times_bandwidth;
    fast_kde_on_two_times_bandwidth->Init(references, references, 
					  reference_weights, true, 
					  kde_module);
    fast_kde_on_two_times_bandwidth->Compute
      (&results_on_two_times_bandwidth);
    delete fast_kde_on_two_times_bandwidth;
    
    // Compute LSCV score.
    double lscv_score = lscv_score_(results_on_bandwidth,
				    results_on_two_times_bandwidth,
				    kernel, references.n_rows());
    
    printf("Least squares cross-validation score is %g...\n\n", lscv_score);
 
  }

  template<typename TKernelAux>
  static void Optimize(const Matrix &references,
		       const Matrix &reference_weights) {
    
    // Get the parameters.
    struct datanode *kde_module = fx_submodule(fx_root, "kde");

    // Minimum LSCV score so far.
    double min_lscv_score = DBL_MAX;

    // Kernel object.
    typename TKernelAux::TKernel kernel;

    // The current lower and upper search limit.
    double plugin_bandwidth = plugin_bandwidth_(references);
    double current_lower_search_limit = plugin_bandwidth * 0.001;
    double current_upper_search_limit = plugin_bandwidth * 1000;
    double min_bandwidth = DBL_MAX;

    printf("Searching the optimal bandwidth in [%g %g]...\n",
	   current_lower_search_limit, current_upper_search_limit);

    do {
      
      // Set bandwidth to the middle of the lower and the upper limit
      // and initialize the kernel.
      double bandwidth = current_upper_search_limit;
      kernel.Init(bandwidth);

      printf("Trying the bandwidth value of %g...\n", bandwidth);

      // Need to run density estimates twice: on $h$ and $sqrt(2)
      // h$. Free memory after each run to minimize memory usage.
      fx_set_param_double(kde_module, "bandwidth", bandwidth);
      DualtreeKde<TKernelAux> *fast_kde_on_bandwidth = 
	new DualtreeKde<TKernelAux>();
      Vector results_on_bandwidth;
      fast_kde_on_bandwidth->Init(references, references, reference_weights,
				  true, kde_module);
      fast_kde_on_bandwidth->Compute(&results_on_bandwidth);
      delete fast_kde_on_bandwidth;

      fx_set_param_double(kde_module, "bandwidth", bandwidth * sqrt(2));
      DualtreeKde<TKernelAux> *fast_kde_on_two_times_bandwidth =
	new DualtreeKde<TKernelAux>();
      Vector results_on_two_times_bandwidth;
      fast_kde_on_two_times_bandwidth->Init(references, references, 
					    reference_weights, true, 
					    kde_module);
      fast_kde_on_two_times_bandwidth->Compute
	(&results_on_two_times_bandwidth);
      delete fast_kde_on_two_times_bandwidth;

      // Compute LSCV score.
      double lscv_score = lscv_score_(results_on_bandwidth,
				      results_on_two_times_bandwidth,
				      kernel, references.n_rows());

      printf("Least squares cross-validation score is %g...\n\n", lscv_score);

      if(lscv_score < min_lscv_score) {
	min_lscv_score = lscv_score;
	min_bandwidth = bandwidth;
      }
      current_upper_search_limit /= 2.0;
      
    } while(current_upper_search_limit > current_lower_search_limit);


    // Output the final density estimates that minimize the least
    // squares cross-validation to the file.
    printf("Minimum score was %g and achieved at the bandwidth value of %g\n", 
	   min_lscv_score, min_bandwidth);

  }

};

#endif
