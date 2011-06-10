/** @file bandwidth_lscv.h
 *
 *  @author Dongryeol Lee (dongryel)
 *
 *  TO-DO: To make this cross-validation a general-purpose tool.
 */

#ifndef BANDWIDTH_LSCV_H
#define BANDWIDTH_LSCV_H

#include "dualtree_kde.h"
#include "dualtree_kde_cv.h"

class BandwidthLSCV {
  
 private:

  static double plugin_bandwidth_(const arma::mat& references) {
    
    double avg_sdev = 0;
    index_t num_dims = references.n_rows;
    index_t num_data = references.n_cols;
    arma::vec mean_vector(references.n_rows);
    mean_vector.zeros();
    
    // First compute the mean vector.
    for(index_t i = 0; i < references.n_cols; i++) {
      for(index_t j = 0; j < references.n_rows; j++) {
	mean_vector[j] += references(j, i);
      }
    }
    mean_vector /= (double) num_data;
    
    // Loop over the dataset again and compute variance along each
    // dimension.
    for(index_t j = 0; j < num_dims; j++) {
      double sdev = 0;
      for(index_t i = 0; i < num_data; i++) {
	sdev += pow(references(j, i) - mean_vector[j], 2.0);
      }
      sdev /= ((double) num_data - 1);
      sdev = sqrt(sdev);
      avg_sdev += sdev;
    }
    avg_sdev /= ((double) num_dims);

    double plugin_bw = 
        pow((4.0 / (num_dims + 2.0)), 1.0 / (num_dims + 4.0)) * avg_sdev * 
        pow(num_data, -1.0 / (num_dims + 4.0));

    return plugin_bw;
  }
  
 public:

  template<typename TKernelAux>
  static void ComputeLSCVScore(const arma::mat& references,
			       const arma::mat& reference_weights,
			       double bandwidth) {
    // Kernel object.
    typename TKernelAux::TKernel kernel;
    
    // LSCV score.
    double lscv_score;

    // Set the bandwidth of the kernel.
    kernel.Init(bandwidth);
    
    mlpack::IO::Info << "Trying the bandwidth value of " << bandwidth << "..." << std::endl;
    
    // Need to run density estimates twice: on $h$ and $sqrt(2)
    // h$. Free memory after each run to minimize memory usage.
    mlpack::IO::GetParam<double>("kde/bandwidth") = bandwidth;

    DualtreeKdeCV<TKernelAux> *fast_kde_on_bandwidth = 
      new DualtreeKdeCV<TKernelAux>();
    fast_kde_on_bandwidth->Init(references, reference_weights);
    lscv_score = fast_kde_on_bandwidth->Compute();
    delete fast_kde_on_bandwidth;
    
    mlpack::IO::Info << "Least squares cross-validation score is " 
      << lscv_score << "..." << std::endl << std::endl;
  }

  template<typename TKernelAux>
  static void Optimize(const arma::mat& references,
		       const arma::mat& reference_weights) {
    // Minimum LSCV score so far.
    double min_lscv_score = DBL_MAX;

    // Kernel object.
    typename TKernelAux::TKernel kernel;

    // The current lower and upper search limit.
    double plugin_bandwidth = plugin_bandwidth_(references);
    double current_lower_search_limit = plugin_bandwidth * 0.00001;
    double current_upper_search_limit = plugin_bandwidth;
    double min_bandwidth = DBL_MAX;

    mlpack::IO::Info << "Searching the optimal bandwidth in [" 
      << current_lower_search_limit 
      << " " << current_upper_search_limit <<"]..." << std::endl;

    do {
      
      // Set bandwidth to the middle of the lower and the upper limit
      // and initialize the kernel.
      double bandwidth = current_upper_search_limit;
      kernel.Init(bandwidth);

      mlpack::IO::Info << "Trying the bandwidth value of " << bandwidth << "..." << std::endl;

      // Need to run density estimates twice: on $h$ and $sqrt(2)
      // h$. Free memory after each run to minimize memory usage.
      mlpack::IO::GetParam<double>("kde/bandwidth") = bandwidth;
      DualtreeKdeCV<TKernelAux> *fast_kde_on_bandwidth =
	new DualtreeKdeCV<TKernelAux>();
      fast_kde_on_bandwidth->Init(references, reference_weights);
      double lscv_score = fast_kde_on_bandwidth->Compute();
      delete fast_kde_on_bandwidth;

      mlpack::IO::Info << "Least squares cross-validation score is " 
        << lscv_score << "..." << std::endl << std::endl;

      if(lscv_score < min_lscv_score) {
	min_lscv_score = lscv_score;
	min_bandwidth = bandwidth;
      }
      current_upper_search_limit /= 2.0;
      
    } while(current_upper_search_limit > current_lower_search_limit);


    // Output the final density estimates that minimize the least
    // squares cross-validation to the file.
    mlpack::IO::Info << "Minimum score was " << min_lscv_score <<
       " and achieved at the bandwidth value of " << min_bandwidth << std::endl; 

  }

};

#endif
