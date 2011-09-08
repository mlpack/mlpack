/*
 * ============================================================================
 *
 *       Filename:  ridge_main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/16/2009 10:55:21 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  
 *
 * ============================================================================
 */
#include <string>
#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>
#include "ridge_regression.h"
#include "ridge_regression_util.h"

#include <armadillo>

using namespace mlpack;


  ////////// Documentation stuffs //////////
PARAM_STRING("inversion_method", "The method chosen for inverting the design matrix: normalsvd\
 (SVD on normal equation: default), svd (SVD), quicsvd (QUIC-SVD).\n", "ridge", "");
PARAM_STRING_REQ("predictions", "A file containing the observed predictions.", "ridge");
PARAM_STRING("predictor_indices", "The file containing the indces of the\
 dimensions that act as the predictors for the input dataset.", "ridge", "");
PARAM_STRING_REQ("predictors", "A file containing the predictors.", "ridge");
PARAM_STRING("predictor_indices", "The file containing the indecs of the\
 dimensions that act as the predictors for the input database.", "ridge", "");
PARAM_STRING("prune_predictor_indices","The file containing the indeces of the\
 dimensions that must be considered for pruning for the input dataset.", "ridge", "");
PARAM_STRING("mode", "Undocumented module", "ridge", "regress");
PARAM_STRING("inversion_method", "Undocumented parameter", "ridge", "normal_svd");
PARAM_STRING("factors", "Undocumented parameter", "ridge", "factors.csv");

PARAM(double, "lambda_min", "The minimum lambda value used for CV (set to zero\
 by default).", "ridge", 0.0, false); 
PARAM(double, "lambda_max", "The maximum lambda value used for CV (set to zero\
 by default).", "ridge", 0.0, false);

PARAM_INT("num_lambdas", "The number of lamdas to try for CV (set to 1 by default).",
 "ridge", 1);

int main(int argc, char** argv) {
  IO::ParseCommandLine(argc, argv);

  double lambda_min = IO::GetParam<double>("ridge/lambda_min");
  double lambda_max = IO::GetParam<double>("ridge/lambda_max");
  int num_lambdas_to_cv = IO::GetParam<int>("ridge/num_lambdas");

  const char *mode = IO::GetParam<std::string>("ridge/mode").c_str();  
  if(lambda_min == lambda_max) {
    num_lambdas_to_cv = 1;
    if(!strcmp(mode, "crossvalidate")) {
      IO::GetParam<std::string>("ridge/mode") = "regress";
      mode = IO::GetParam<std::string>("ridge/mode").c_str();
    }
  }
  else {
    IO::GetParam<std::string>("ridge/mode") = "cvregress";
    mode = IO::GetParam<std::string>("ridge/mode").c_str();
  }

  // Read the dataset and its labels.
  std::string predictors_file = IO::GetParam<std::string>("ridge/predictors"); 
  std::string predictions_file = IO::GetParam<std::string>("ridge/predictions");

  arma::mat predictors;
  if (data::Load(predictors_file.c_str(), predictors) == false) {
    IO::Fatal << "Unable to open file " << predictors_file.c_str() << std::endl;
  }

  arma::mat predictions;
  if (data::Load(predictions_file.c_str(), predictions) == false) {
    IO::Fatal << "Unable to open file " << predictions_file.c_str() << std::endl;
  }

  RidgeRegression engine;
  IO::Info << "Computing Regression..." << std::endl;
  

  if(!strcmp(mode, "regress")) {

    engine.Init(predictors, predictions, 
		!strcmp(IO::GetParam<std::string>("ridge/inversion_method").c_str(), "normalsvd"));
    engine.QRRegress(lambda_min);
  }
  else if(!strcmp(mode, "cvregress")) {
  //    IO::Info << "Crossvalidating for the optimal lamda in [ " 
//	<< lambda_min << " " << lamda_max << " ] by trying "
//	<< num_lamdas_to_cv << std::endl;

     IO::Info << "Crossvalidating for the optimal lambda in [" 
	<<  lambda_min << " " << lambda_max << " ] " 
     	<< "by trying " << num_lambdas_to_cv << " values..." << std::endl;
   
    engine.Init(predictors, predictions);
    engine.CrossValidatedRegression(lambda_min, lambda_max, num_lambdas_to_cv);
  }
  else if(!strcmp(mode, "fsregress")) {

    IO::Info << "Feature selection based regression." << std::endl;

    arma::mat predictor_indices_intermediate;
    arma::mat prune_predictor_indices_intermediate;
    std::string predictor_indices_file = 
      IO::GetParam<std::string>("ridge/predictor_indices");
    std::string prune_predictor_indices_file = 
      IO::GetParam<std::string>("ridge/prune_predictor_indices");
    if(data::Load(predictor_indices_file.c_str(), 
	  predictor_indices_intermediate) == false) {
      IO::Fatal << "Unable to open file " << prune_predictor_indices_file.c_str() << std::endl;
    }
    if(data::Load(prune_predictor_indices_file.c_str(),
	  prune_predictor_indices_intermediate) == false) {
      IO::Fatal << "Unable to open file " << prune_predictor_indices_file.c_str() << std::endl;
    }

    arma::Col<size_t> predictor_indices;
    arma::Col<size_t> prune_predictor_indices;
    predictor_indices.zeros(predictor_indices_intermediate.n_cols);
    prune_predictor_indices.zeros
      (prune_predictor_indices_intermediate.n_cols);
    
    // This is a pretty retarded way of copying from a double-matrix
    // to an int vector. This can be simplified only if there were a
    // way to read integer-based dataset directly without typecasting.
    for(size_t i = 0; i < predictor_indices_intermediate.n_cols; i++) {
      predictor_indices[i] =
	(size_t) predictor_indices_intermediate(0, i);
    }
    for(size_t i = 0; i < prune_predictor_indices_intermediate.n_cols;
	i++) {
      prune_predictor_indices[i] = (size_t)
	prune_predictor_indices_intermediate(0, i);
    }
    
    // Run the feature selection.
    arma::Col<size_t> output_predictor_indices;
    engine.Init(predictors, predictor_indices, predictions,
		!strcmp(IO::GetParam<std::string>("ridge/inversion_method").c_str(), "normalsvd"));
    engine.FeatureSelectedRegression(predictor_indices,
				     prune_predictor_indices,
				     predictions,
				     &output_predictor_indices);
  }

  IO::Info << "Ridge Regression Model Training Complete!" << std::endl;
  double square_error = engine.ComputeSquareError();
  IO::Info << "Square Error: " << square_error << std::endl;
  IO::GetParam<double>("ridge/square error") = square_error;
  arma::mat factors;
  engine.factors(&factors);
  std::string factors_file = IO::GetParam<std::string>("ridge/factors");
  IO::Info << "Saving factors..." << std::endl;
  data::Save(factors_file.c_str(), factors);

  return 0;
}
