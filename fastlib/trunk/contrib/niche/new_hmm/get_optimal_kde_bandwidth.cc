/** @file kde_bandwidth_cv_main.cc
 *
 *  Driver file for bandwidth optimizer for KDE.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include "fastlib/fastlib.h"
#include "mlpack/kde/bandwidth_lscv.h"
#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/kde/dualtree_kde.h"
#include "mlpack/kde/naive_kde.h"

/**
 * Main function which reads parameters and determines which
 * algorithms to run.
 *
 * In order to compile this driver, do:
 * fl-build dualtree_kde_bin --mode=fast
 *
 * In order to run this driver for the fast KDE algorithms, type the
 * following (which consists of both required and optional arguments)
 * in a single command line:
 *
 * ./dualtree_kde_bin --data=name_of_the_reference_dataset
 *                    --query=name_of_the_query_dataset 
 *                    --kde/kernel=gaussian
 *                    --kde/bandwidth=0.0130619
 *                    --kde/scaling=range 
 *                    --kde/multiplicative_expansion
 *                    --kde/fast_kde_output=fast_kde_output.txt
 *                    --kde/naive_kde_output=naive_kde_output.txt 
 *                    --kde/do_naive
 *                    --kde/relative_error=0.01
 *
 * Explanations for the arguments listed with possible values: 
 *
 * 1. data (required): the name of the reference dataset
 *
 * 2. query (optional): the name of the query dataset (if missing, the
 * query dataset is assumed to be the same as the reference dataset)
 *
 * 3. kde/kernel (optional): kernel function to use
 * - gaussian: Gaussian kernel (default) 
 * - epan: Epanechnikov kernel
 *
 * 4. kde/bandwidth (required): smoothing parameter used for KDE; this
 * has to be positive.
 *
 * 5. kde/scaling (optional): whether to prescale the dataset 
 *
 * - range: scales both the query and the reference sets to be within
 * the unit hypercube [0, 1]^D where D is the dimensionality.
 * - standardize: scales both the query and the reference set to have
 * zero mean and unit variance.
 * - none: default value; no scaling
 *
 * 6. kde/multiplicative_expansion (optional): If this flag is
 * present, the series expansion for the Gaussian kernel uses O(p^D)
 * expansion. Otherwise, the Gaussian kernel uses O(D^p)
 * expansion. See kde.h for details.
 *
 * 7. kde/do_naive (optional): run the naive algorithm after the fast
 * algorithm.
 *
 * 8. kde/fast_kde_output (optional): if this flag is present, the
 * approximated density estimates are output to the filename provided
 * after it.
 *
 * 9. kde/naive_kde_output (optional): if this flag is present, the
 * exact density estimates computed by the naive algorithm are output
 * to the filename provided after it. This flag is not ignored if
 * --kde/do_naive flag is not present.
 *
 * 10. kde/relative_error (optional): relative error criterion for the
 * fast algorithm; default value is 0.1 (10 percent relative error for
 * all query density estimates).
 */

/*
void ScaleSamplingsToCube(ArrayList<Matrix> *p_samplings) {
  ArrayList<Matrix> &samplings = *p_samplings;

  DHrectBound<2> total_bound;
  total_bound.Init(samplings[0].n_rows());

  int n_samplings = samplings.size();
  int n_dims = samplings[0].n_rows();

  for(int k = 0; k < n_samplings; k++) {
    const Matrix &dataset = samplings[k];
    int n_points = dataset.n_cols();
    for(int i = 0; i < n_points; i++) {
      Vector point;
      dataset.MakeColumnVector(i, &point);
      total_bound |= point;
    }
  }

  Vector mins;
  mins.Init(n_dims);
  
  Vector ranges;
  ranges.Init(n_dims);
  
  for(int i = 0; i < n_dims; i++) {
    mins[i] = total_bound.get(i).lo;
    ranges[i] = total_bound.get(i).hi - mins[i];
  }
  
  for(int k = 0; k < n_samplings; k++) {
    Matrix &dataset = samplings[k];
    int n_points = dataset.n_cols();
    for(int i = 0; i < n_points; i++) {
      for(int j = 0; j < n_dims; j++) {
	dataset.set(j, i,
		    (dataset.get(j, i) - mins[i]) / ranges[i]);
      }
    }
  }
}

// scales the data permanently!
void KDEGenerativeMMKBatch(double lambda,
			   ArrayList<Matrix> *p_samplings,
			   Matrix *p_kernel_matrix) {
  ArrayList<Matrix> &samplings = *p_samplings;
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_samplings = samplings.size();
  
  ScaleSamplingsToCube(&samplings);

  Vector optimal_bandwidths;
  optimal_bandwidths.Init(n_samplings);
  for(int k = 0; k < n_samplings; k++) {
    // Query and reference datasets, reference weight dataset.
    const Matrix &references = samplings[k];
    Matrix reference_weights;
    Matrix queries;

    // data::Load inits a matrix with the contents of a .csv or .arff.
    queries.Alias(references);
  
    // initialize to uniform weights.
    reference_weights.Init(1, queries.n_cols());
    reference_weights.SetAll(1);

    optimal_bandwidths[j] =
      BandwidthLSCV::OptimizeReturn<GaussianKernelAux>(references, 
						       reference_weights);
  }

  kernel_matrix.Init(n_samplings, n_samplings);
  for(int i = 0; i < n_samplings; i++) {
    printf("%f%%\n", ((double)(i + 1)) / ((double)n_samplings));
    for(int j = i; j < n_samplings; j++) {
      double gmmk = 
	GenerativeMMK(lambda,
		      samplings[i], samplings[j],
		      optimal_bandwidths[i], optimal_bandwidths[j]);
      
      kernel_matrix.set(j, i, gmmk);
      if(i != j) {
	kernel_matrix.set(i, j, gmmk);
      }
    }
  }
}
*/



int main(int argc, char *argv[]) {

  fx_init(argc, argv, &kde_main_doc);

  const char* references_file_name = fx_param_str_req(fx_root, "data");
  
  // Query and reference datasets, reference weight dataset.
  Matrix references;
  Matrix reference_weights;
  Matrix queries;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);  
  queries.Alias(references);
  
  // initialize to uniform weights.
  reference_weights.Init(1, queries.n_cols());
  reference_weights.SetAll(1);

  BandwidthLSCV::OptimizeReturn<GaussianKernelAux>(references, 
						   reference_weights);
  
  fx_done(fx_root);
  return 0;
}
