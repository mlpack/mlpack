/** @file dualtree_kde_main.cc
 *
 *  Driver file for dualtree KDE algorithm.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>
#include <armadillo>

#include "dataset_scaler.h"
#include "dualtree_kde.h"
#include "dualtree_vkde.h"
#include "naive_kde.h"

using namespace mlpack;

void VariableBandwidthKde(arma::mat& queries, arma::mat& references, 
			  arma::mat& reference_weights, 
			  bool queries_equal_references) {

  // flag for determining whether to compute naively
  bool do_naive = IO::HasParam("kde/do_naive");
  
  //Despite the various default values specified for kde/kernel
  //kde/kernel can only have one, and that is 'gaussian'
  if(IO::GetParam<std::string>("kde/kernel").compare("") == 0)
    IO::GetParam<std::string>("kde/kernel") = "gaussian";  

  if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "gaussian")) {
    
    arma::vec fast_kde_results;
    
    // for O(p^D) expansion
    if(IO::HasParam("kde/multiplicative_expansion")) {
      
      IO::Info << "O(p^D) expansion KDE" << std::endl;

      DualtreeVKde<GaussianKernel> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references);
      fast_kde.Compute(fast_kde_results);
      
      if(IO::HasParam("kde/fast_kde_output")) {
        fast_kde.PrintDebug();
      }
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      IO::Info << "O(D^p) expansion KDE" << std::endl;
      DualtreeVKde<GaussianKernel> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references);
      fast_kde.Compute(fast_kde_results);
      
      if(true || IO::HasParam("kde/fast_kde_output")) {
        fast_kde.PrintDebug();
      }
    }

    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights);
      naive_kde.Compute();
      
      if(true || IO::HasParam("kde/naive_kde_output")) {
        naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
  }
  else if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "epan")) {
    DualtreeVKde<EpanKernel> fast_kde;
    arma::vec fast_kde_results;

    fast_kde.Init(queries, references, reference_weights,
		  queries_equal_references);
    fast_kde.Compute(fast_kde_results);
    
    if(IO::HasParam("kde/fast_kde_output")) {
      fast_kde.PrintDebug();
    }

    if(do_naive) {
      NaiveKde<EpanKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights);
      naive_kde.Compute();
      
      if(IO::HasParam("kde/naive_kde_output")) {
        naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
  }
}

void FixedBandwidthKde(arma::mat& queries, arma::mat& references, 
		       arma::mat& reference_weights,
		       bool queries_equal_references) {
  
  // flag for determining whether to compute naively
  bool do_naive = IO::HasParam("kde/do_naive");

  if(IO::GetParam<std::string>("kde/kernel").compare("") == 0)
    IO::GetParam<std::string>("kde/kernel") = std::string("epan");

  if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "gaussian")) {
    
    arma::vec fast_kde_results;
    
    // for O(p^D) expansion
    if(IO::HasParam("kde/multiplicative_expansion")) {
      
      IO::Info << "O(p^D) expansion KDE" << std::endl;
      DualtreeKde<GaussianKernelMultAux> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references);
      fast_kde.Compute(fast_kde_results);
      
      if(IO::HasParam("kde/fast_kde_output")) {
        fast_kde.PrintDebug();
      }
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      IO::Info << "O(D^p) expansion KDE" << std::endl;

      DualtreeKde<GaussianKernelAux> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references);
      fast_kde.Compute(fast_kde_results);
      
      if(true || IO::HasParam("kde/fast_kde_output")) {
        fast_kde.PrintDebug();
      }
    }
    
    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights);
      naive_kde.Compute();
      
      if(true || IO::HasParam("kde/naive_kde_output")) {
        naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
    
  }
  else if(!strcmp(IO::GetParam<std::string>("kde/kernel").c_str(), "epan")) {
    DualtreeKde<EpanKernelAux> fast_kde;
    arma::vec fast_kde_results;

    fast_kde.Init(queries, references, reference_weights,
		  queries_equal_references);
    fast_kde.Compute(fast_kde_results);
    
    if(IO::HasParam("kde/fast_kde_output")) {
      fast_kde.PrintDebug();
    }
    
    if(do_naive) {
      NaiveKde<EpanKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights);
      naive_kde.Compute();
      
      if(IO::HasParam("kde/naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
  }
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
  IO::ParseCommandLine(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".

  // The reference data file is a required parameter.
  const char* references_file_name = 
    IO::GetParam<std::string>("kde/data").c_str();

  // The query data file defaults to the references.
  if(!IO::HasParam("kde/query"))
    IO::GetParam<std::string>("kde/query") = references_file_name;
  const char* queries_file_name =
    IO::GetParam<std::string>("kde/query").c_str(); //Default value = references_file_name
  
  // Query and reference datasets, reference weight dataset.
  arma::mat references;
  arma::mat reference_weights;
  arma::mat queries;

  // Flag for telling whether references are equal to queries
  bool queries_equal_references = 
    !strcmp(queries_file_name, references_file_name);

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, references);
  if(queries_equal_references) {
    // make an alias; I don't like to do it this way, this code needs to be
    // redesigned so that this is not a problem
    queries = arma::mat(references.memptr(), references.n_rows, references.n_cols, false, true);
  }
  else {
    data::Load(queries_file_name, queries);
  }

  // If the reference weight file name is specified, then read in,
  // otherwise, initialize to uniform weights.
  if(IO::HasParam("kde/dwgts")) {
    data::Load(IO::GetParam<std::string>("kde/dwgts").c_str(), reference_weights);
  }
  else {
    reference_weights.set_size(1, references.n_cols);
    reference_weights.fill(1.0);
  }
  
  // Confirm whether the user asked for scaling of the dataset
  if(!strcmp(IO::GetParam<std::string>("kde/scaling").c_str(), "range")) {
    DatasetScaler::ScaleDataByMinMax(queries, references,
                                     queries_equal_references);
  }
  else if(!strcmp(IO::GetParam<std::string>("kde/scaling").c_str(), 
		  "standardize")) {
    DatasetScaler::StandardizeData(queries, references, 
				   queries_equal_references);
  }

  // By default, we want to run the fixed-bandwidth KDE.
  //Default kde/mode = fixedbw
  if(IO::GetParam<std::string>("kde/mode").compare("") == 0)
    IO::GetParam<std::string>("kde/mode") = "variablebw";

  if(!strcmp(IO::GetParam<std::string>("kde/mode").c_str(), "variablebw")) {
    VariableBandwidthKde(queries, references, reference_weights, 
			 queries_equal_references);
  }
  else {
    FixedBandwidthKde(queries, references, reference_weights, 
		      queries_equal_references);
  }

  return 0;
}
