/* MLPACK 0.1
 *
 * Copyright (c) 2008 Alexander Gray,
 *                    Garry Boyer,
 *                    Ryan Riegel,
 *                    Nikolaos Vasiloglou,
 *                    Dongryeol Lee,
 *                    Chip Mappus, 
 *                    Nishant Mehta,
 *                    Hua Ouyang,
 *                    Parikshit Ram,
 *                    Long Tran,
 *                    Wee Chin Wong
 *
 * Copyright (c) 2008 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
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
    for(index_t j = 0; j < num_dims; j++) {
      double sdev = 0;
      for(index_t i = 0; i < num_data; i++) {
	sdev += math::Sqr(references.get(j, i) - mean_vector[j]);
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
  static void ComputeLSCVScore(const Matrix &references,
			       const Matrix &reference_weights,
			       double bandwidth) {

    // Get the parameters.
    struct datanode *kde_module = fx_submodule(fx_root, "kde");

    // Kernel object.
    typename TKernelAux::TKernel kernel;
    
    // LSCV score.
    double lscv_score;

    // Set the bandwidth of the kernel.
    kernel.Init(bandwidth);
    
    printf("Trying the bandwidth value of %g...\n", bandwidth);
    
    // Need to run density estimates twice: on $h$ and $sqrt(2)
    // h$. Free memory after each run to minimize memory usage.
    fx_set_param_double(kde_module, "bandwidth", bandwidth);
    DualtreeKdeCV<TKernelAux> *fast_kde_on_bandwidth = 
      new DualtreeKdeCV<TKernelAux>();
    fast_kde_on_bandwidth->Init(references, reference_weights, kde_module);
    lscv_score = fast_kde_on_bandwidth->Compute();
    delete fast_kde_on_bandwidth;
    
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
    double current_lower_search_limit = plugin_bandwidth * 0.00001;
    double current_upper_search_limit = plugin_bandwidth;
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
      DualtreeKdeCV<TKernelAux> *fast_kde_on_bandwidth =
	new DualtreeKdeCV<TKernelAux>();
      fast_kde_on_bandwidth->Init(references, reference_weights, kde_module);
      double lscv_score = fast_kde_on_bandwidth->Compute();
      delete fast_kde_on_bandwidth;

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
