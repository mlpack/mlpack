/** @file dualtree_kde_main.cc
 *
 *  Driver file for dualtree KDE algorithm.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include "mlpack/core.h"
#include "mlpack/core/kernels/gaussian_kernel.hpp"
#include "dataset_scaler.h"
#include "dualtree_kde.h"
#include "dualtree_vkde.h"
#include "naive_kde.h"

using namespace mlpack;
using namespace mlpack::kernel;

void VariableBandwidthKde(arma::mat &queries, arma::mat &references, 
			  arma::mat &reference_weights, 
			  bool queries_equal_references,
			  struct datanode *kde_module) {

  // flag for determining whether to compute naively
  bool do_naive = CLI::HasParam("do_naive");

  if(!strcmp(CLI::GetParam<std::string>("kernel").c_str(), "gaussian")) {

    arma::vec fast_kde_results;

    // for O(p^D) expansion
    if(CLI::HasParam("multiplicative_expansion")) {

      printf("O(p^D) expansion KDE\n");
      DualtreeVKde<GaussianKernel> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references, kde_module);
      fast_kde.Compute(&fast_kde_results);

      if(CLI::HasParam("fast_kde_output")) {
	fast_kde.PrintDebug();
      }
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      printf("O(D^p) expansion KDE\n");
      DualtreeVKde<GaussianKernel> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references, kde_module);
      fast_kde.Compute(&fast_kde_results);
      
      if(true || CLI::HasParam("fast_kde_output")) {
	fast_kde.PrintDebug();
      }
    }

    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights, kde_module);
      naive_kde.Compute();
      
      if(true || CLI::HasParam("naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
  }
//  else if(!strcmp(CLI::GetParam<std::string>("kernel").c_str(), "epan")) {
//    DualtreeVKde<EpanKernel> fast_kde;
//    arma::vec fast_kde_results;
//
//    fast_kde.Init(queries, references, reference_weights,
//		  queries_equal_references, kde_module);
//    fast_kde.Compute(&fast_kde_results);
//    
//    if(CLI::HasParam("fast_kde_output")) {
//      fast_kde.PrintDebug();
//    }
//
//    if(do_naive) {
//      NaiveKde<EpanKernel> naive_kde;
//      naive_kde.Init(queries, references, reference_weights, kde_module);
//      naive_kde.Compute();
//      
//      if(CLI::HasParam("naive_kde_output")) {
//	naive_kde.PrintDebug();
//      }
//      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
//    }
//  }
}

void FixedBandwidthKde(arma::mat &queries, arma::mat &references, 
		       arma::mat &reference_weights, 
		       bool queries_equal_references,
		       struct datanode *kde_module) {

  // flag for determining whether to compute naively
  bool do_naive = CLI::HasParam("do_naive");

  if(!strcmp(CLI::GetParam<std::string>("kernel").c_str(), "gaussian")) {
    
    arma::vec fast_kde_results;
    
    // for O(p^D) expansion
    if(CLI::HasParam("multiplicative_expansion")) {
      
      printf("O(p^D) expansion KDE\n");
      DualtreeKde<kernel::GaussianKernel> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references, kde_module);
      fast_kde.Compute(&fast_kde_results);
      
      if(CLI::HasParam("fast_kde_output")) {
	fast_kde.PrintDebug();
      }
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      printf("O(D^p) expansion KDE\n");
      DualtreeKde<kernel::GaussianKernel> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references, kde_module);
      fast_kde.Compute(&fast_kde_results);
      
      if(true || CLI::HasParam("fast_kde_output")) {
	fast_kde.PrintDebug();
      }
    }
    
    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights, kde_module);
      naive_kde.Compute();
      
      if(true || CLI::HasParam("naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }

  }
//  else if(!strcmp(CLI::GetParam<std::string>("kernel").c_str(), "epan")) {
//    DualtreeKde<EpanKernelAux> fast_kde;
//    arma::vec fast_kde_results;
//
//    fast_kde.Init(queries, references, reference_weights,
//		  queries_equal_references, kde_module);
//    fast_kde.Compute(&fast_kde_results);
//
//    if(CLI::HasParam("fast_kde_output")) {
//      fast_kde.PrintDebug();
//    }
//    
//    if(do_naive) {
//      NaiveKde<EpanKernel> naive_kde;
//      naive_kde.Init(queries, references, reference_weights, kde_module);
//      naive_kde.Compute();
//      
//      if(CLI::HasParam("naive_kde_output")) {
//	naive_kde.PrintDebug();
//      }
//      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
//    }
//  }
}

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
 *                    --dwgts=name_of_the_reference_weight_dataset
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
 * 2. dwgts (optional): the name of the reference weight dataset. If
 * missing, then assumes a uniformly weighted kernel density
 * estimation.
 *
 * 3. query (optional): the name of the query dataset (if missing, the
 * query dataset is assumed to be the same as the reference dataset)
 *
 * 4. kde/kernel (optional): kernel function to use
 * - gaussian: Gaussian kernel (default) 
 * - epan: Epanechnikov kernel
 *
 * 5. kde/bandwidth (required): smoothing parameter used for KDE; this
 * has to be positive.
 *
 * 6. kde/scaling (optional): whether to prescale the dataset 
 *
 * - range: scales both the query and the reference sets to be within
 * the unit hypercube [0, 1]^D where D is the dimensionality.
 * - standardize: scales both the query and the reference set to have
 * zero mean and unit variance.
 * - none: default value; no scaling
 *
 * 7. kde/multiplicative_expansion (optional): If this flag is
 * present, the series expansion for the Gaussian kernel uses O(p^D)
 * expansion. Otherwise, the Gaussian kernel uses O(D^p)
 * expansion. See kde.h for details.
 *
 * 8. kde/do_naive (optional): run the naive algorithm after the fast
 * algorithm.
 *
 * 9. kde/fast_kde_output (optional): if this flag is present, the
 * approximated density estimates are output to the filename provided
 * after it.
 *
 * 10. kde/naive_kde_output (optional): if this flag is present, the
 * exact density estimates computed by the naive algorithm are output
 * to the filename provided after it. This flag is not ignored if
 * --kde/do_naive flag is not present.
 *
 * 11. kde/relative_error (optional): relative error criterion for the
 * fast algorithm; default value is 0.1 (10 percent relative error for
 * all query density estimates).
 */
int main(int argc, char *argv[]) {

  // initialize FastExec (parameter handling stuff)
  CLI::ParseCommandLine(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  struct datanode* kde_module = NULL;

  // The reference data file is a required parameter.
  std::string references_file_name = CLI::GetParam<std::string>("data");

  // The query data file defaults to the references.
  std::string queries_file_name =
    CLI::GetParam<std::string>("query");
  
  // Query and reference datasets, reference weight dataset.
  arma::mat references;
  arma::mat reference_weights;
  arma::mat queries;

  // Flag for telling whether references are equal to queries
  bool queries_equal_references = 
    !strcmp(queries_file_name.c_str(), references_file_name.c_str());

  // data::Load inits a arma::mat with the contents of a .csv or .arff.
  references.load(references_file_name.c_str());
  if(queries_equal_references) {
    queries = references;
  }
  else {
    queries.load(queries_file_name.c_str());
  }

  // If the reference weight file name is specified, then read in,
  // otherwise, initialize to uniform weights.
  if(CLI::HasParam("dwgts")) {
    reference_weights.load(CLI::GetParam<std::string>("dwgts").c_str());
  }
  else {
    reference_weights = arma::mat(1, references.n_cols);
    reference_weights.fill(1);
  }

  // Confirm whether the user asked for scaling of the dataset
  if(!strcmp(CLI::GetParam<std::string>("scaling").c_str(), "range")) {
    DatasetScaler::ScaleDataByMinMax(queries, references,
                                     queries_equal_references);
  }
  else if(!strcmp(CLI::GetParam<std::string>("scaling").c_str(),
		  "standardize")) {
    DatasetScaler::StandardizeData(queries, references,
				   queries_equal_references);
  }

  // By default, we want to run the fixed-bandwidth KDE.
  if(!strcmp(CLI::GetParam<std::string>("mode").c_str(), "variablebw")) {
    VariableBandwidthKde(queries, references, reference_weights, 
			 queries_equal_references, kde_module);
  }
  else {
    FixedBandwidthKde(queries, references, reference_weights, 
		      queries_equal_references, kde_module);
  }

  return 0;
}
