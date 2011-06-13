/** @file kde_bandwidth_cv_main.cc
 *
 *  Driver file for bandwidth optimizer for KDE.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>
#include <armadillo>

#include "bandwidth_lscv.h"
#include "dataset_scaler.h"
#include "dualtree_kde.h"
#include "naive_kde.h"

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

using namespace mlpack;

int main(int argc, char *argv[]) {

  // initialize FastExec (parameter handling stuff)
  IO::ParseCommandLine(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".

  // The reference data file is a required parameter.
  const char* references_file_name = IO::GetParam<std::string>("kde/data").c_str();
  
  // The query data file defaults to the references.
  const char* queries_file_name;
  if(!IO::HasParam("kde/query"))
    queries_file_name = references_file_name;
  else
    queries_file_name = IO::GetParam<std::string>("kde/queries").c_str();

  // Query and reference datasets, reference weight dataset.
  arma::mat references;
  arma::mat reference_weights;
  arma::mat* queries_ptr;
  arma::mat queries;

  // Flag for telling whether references are equal to queries
  bool queries_equal_references = 
    !strcmp(queries_file_name, references_file_name);

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, references);
  if(queries_equal_references) {
    queries_ptr = &references;
  }
  else {
    data::Load(queries_file_name, queries);
    queries_ptr = &queries;
  }
  
  // If the reference weight file name is specified, then read in,
  // otherwise, initialize to uniform weights.
  if(IO::HasParam("kde/dwgts")) {
    data::Load(fx_param_str(fx_root, "dwgts", NULL), reference_weights);
  }
  else {
    reference_weights.ones(1, queries.n_cols);
  }

  // Confirm whether the user asked for scaling of the dataset.
  if(!strcmp(IO::GetParam<std::string>("kde/scaling").c_str(), "range")) {
    DatasetScaler::ScaleDataByMinMax(*queries_ptr, references,
                                     queries_equal_references);
  }
  else if(!strcmp(IO::GetParam<std::string>("kde/scaling").c_str(), 
		  "standardize")) {
    DatasetScaler::StandardizeData(*queries_ptr, references, 
				   queries_equal_references);
  }

  // There are two options: 1) do bandwidth optimization 2) output a
  // goodness score of a given bandwidth.
  if(IO::GetParam<std::string>("kde/task").compare("") == 0)
    IO::GetParam<std::string>("kde/task") = "optimize";

  if(!strcmp(IO::GetParam<std::string>("kde/task").c_str(), "optimize")) { //Default value optimize

    // Optimize bandwidth using least squares cross-validation.
    //kde/kernel will always default to gaussian.
    if(IO::GetParam<std::string>("kde/kernel").compare("") == 0)
      IO::GetParam<std::string>("kde/kernel") = "gaussian";

    if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "gaussian")) { //Default value gaussian
      BandwidthLSCV::Optimize<GaussianKernelAux>(references, 
						 reference_weights);
    }
    else if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "epan")) { //Default value epan
      
      // Currently, I have not implemented the direct way to
      // cross-validate for the optimal bandwidth using the Epanechnikov
      // kernel, so I will use cross-validation using the Gaussian
      // kernel with the equivalent kernel scaling.
      BandwidthLSCV::Optimize<GaussianKernelAux>(references, 
						 reference_weights);
    }
  }
  else if(!strcmp(IO::GetParam<std::string>("kde/task").c_str(), //Default value lscvscore
		  "lscvscore")) {
    
    // Get the bandwidth.
    double bandwidth = IO::GetParam<double>("kde/bandwidth"); //Default value .1

    // Optimize bandwidth using least squares cross-validation.
    if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "gaussian")) { //Default value gaussian
      BandwidthLSCV::ComputeLSCVScore<GaussianKernelAux>
	(references, reference_weights, bandwidth);
    }
    else if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "epan")) { //Default value epan
      
      // Currently, I have not implemented the direct way to
      // cross-validate for the optimal bandwidth using the Epanechnikov
      // kernel, so I will use cross-validation using the Gaussian
      // kernel with the equivalent kernel scaling.
      BandwidthLSCV::ComputeLSCVScore<GaussianKernelAux>
	(references, reference_weights, bandwidth);
    }    
  }

  return 0;
}
