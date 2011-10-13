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
#include <mlpack/core.h>
#include "ridge_regression.h"
#include "ridge_regression_util.h"

using namespace mlpack;
using namespace regression;

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

  std::string mode = IO::GetParam<std::string>("ridge/mode");
  if(lambda_min == lambda_max) {
    num_lambdas_to_cv = 1;
    if(mode == "crossvalidate") {
      IO::GetParam<std::string>("ridge/mode") = "regress";
      mode = IO::GetParam<std::string>("ridge/mode");
    }
  }
  else {
    IO::GetParam<std::string>("ridge/mode") = "cvregress";
    mode = IO::GetParam<std::string>("ridge/mode");
  }

  // Read the dataset and its labels.
  std::string predictors_file = IO::GetParam<std::string>("ridge/predictors");
  std::string predictions_file = IO::GetParam<std::string>("ridge/predictions");

  arma::mat predictors;
  if (data::Load(predictors_file.c_str(), predictors) == false) {
    Log::Fatal << "Unable to open file " << predictors_file << std::endl;
  }

  arma::mat predictions;
  if (data::Load(predictions_file.c_str(), predictions) == false) {
    Log::Fatal << "Unable to open file " << predictions_file << std::endl;
  }

  RidgeRegression engine;
  Log::Info << "Computing Regression..." << std::endl;


  if(mode == "regress") {

    engine = RidgeRegression(predictors, predictions,
		IO::GetParam<std::string>("ridge/inversion_method") == "normalsvd");
    engine.QRRegress(lambda_min);
  }
  else if(mode == "cvregress") {
     Log::Info << "Crossvalidating for the optimal lambda in ["
	<<  lambda_min << " " << lambda_max << " ] "
     	<< "by trying " << num_lambdas_to_cv << " values..." << std::endl;

    engine = RidgeRegression(predictors, predictions);
    engine.CrossValidatedRegression(lambda_min, lambda_max, num_lambdas_to_cv);
  }
  else if(mode == "fsregress") {

    Log::Info << "Feature selection based regression." << std::endl;

    arma::mat predictor_indices_intermediate;
    arma::mat prune_predictor_indices_intermediate;
    std::string predictor_indices_file =
      IO::GetParam<std::string>("ridge/predictor_indices");
    std::string prune_predictor_indices_file =
      IO::GetParam<std::string>("ridge/prune_predictor_indices");
    if(data::Load(predictor_indices_file.c_str(),
	  predictor_indices_intermediate) == false) {
      Log::Fatal << "Unable to open file " << prune_predictor_indices_file << std::endl;
    }
    if(data::Load(prune_predictor_indices_file.c_str(),
	  prune_predictor_indices_intermediate) == false) {
      Log::Fatal << "Unable to open file " << prune_predictor_indices_file << std::endl;
    }

    arma::Col<size_t> predictor_indices;
    arma::Col<size_t> prune_predictor_indices;

    { // Convert from double rowvec -> size_t colvec
      typedef arma::Col<size_t> size_t_vec;
      predictor_indices = arma::conv_to<size_t_vec>::
                          from(predictor_indices_intermediate.row(0));
      prune_predictor_indices = arma::conv_to<size_t_vec>::
	                        from(prune_predictor_indices_intermediate.row(0));
    }
  }

  Log::Info << "Ridge Regression Model Training Complete!" << std::endl;
  double square_error = engine.ComputeSquareError();
  Log::Info << "Square Error: " << square_error << std::endl;
  IO::GetParam<double>("ridge/square error") = square_error;
  arma::mat factors;
  engine.factors(&factors);
  std::string factors_file = IO::GetParam<std::string>("ridge/factors");
  Log::Info << "Saving factors..." << std::endl;
  data::Save(factors_file.c_str(), factors);

  return 0;
}
