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
#include "ridge_regression.h"
#include "ridge_regression_util.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

int main(int argc, char *argv[]) {

  ////////// Documentation stuffs //////////
  const fx_entry_doc ridge_main_entries[] = {
    {"inversion_method", FX_PARAM, FX_STR, NULL,
     "  The method chosen for inverting the design matrix: normalsvd\
 (SVD on normal equation: default), svd (SVD), quicsvd (QUIC-SVD).\n"},
    {"lambda_min", FX_PARAM, FX_DOUBLE, NULL,
     "  The minimum lambda value used for CV (set to zero by default).\n"},
    {"lambda_max", FX_PARAM, FX_DOUBLE, NULL,
     "  The maximum lambda value used for CV (set to zero by default).\n"},
    {"mode", FX_PARAM, FX_STR, NULL,
     "  The computation mode: regress, cvregress (cross-validated regression),\
 fsregress (feature selection then regress).\n"},
    {"num_lambdas", FX_PARAM, FX_INT, NULL,
     "  The number of lambdas to try for CV (set to 1 by default).\n"},
    {"predictions", FX_REQUIRED, FX_STR, NULL,
     "  A file containing the observed predictions.\n"},
    {"predictor_indices", FX_PARAM, FX_STR, NULL,
     "  The file containing the indices of the dimensions that act as the \
predictors for the input dataset.\n"},
    {"predictors", FX_REQUIRED, FX_STR, NULL,
     "  A file containing the predictors.\n"},
    {"prune_predictor_indices", FX_PARAM, FX_STR, NULL,
     "  The file containing the indices of the dimensions that must be \
considered for pruning for the input dataset.\n"},
    FX_ENTRY_DOC_DONE
  };
  
  const fx_submodule_doc ridge_main_submodules[] = {
    FX_SUBMODULE_DOC_DONE
  };
  
  const fx_module_doc ridge_main_doc = {
    ridge_main_entries, ridge_main_submodules,
    "This is the driver for the ridge regression.\n"
  };

  fx_module *module = fx_init(argc, argv, &ridge_main_doc);
  double lambda_min = fx_param_double(module, "lambda_min", 0.0);
  double lambda_max = fx_param_double(module, "lambda_max", 0.0);
  int num_lambdas_to_cv = fx_param_int(module, "num_lambdas", 1);
  const char *mode = fx_param_str(module, "mode", "regress");  
  if(lambda_min == lambda_max) {
    num_lambdas_to_cv = 1;
    if(!strcmp(mode, "crossvalidate")) {
      fx_set_param_str(module, "mode", "regress");
      mode = fx_param_str(module, "mode", "regress");
    }
  }
  else {
    fx_set_param_str(module, "mode", "cvregress");
    mode = fx_param_str(module, "mode", "cvregress");
  }

  // Read the dataset and its labels.
  std::string predictors_file = fx_param_str_req(module, "predictors"); 
  std::string predictions_file = fx_param_str_req(module, "predictions");

  arma::mat predictors;
  if (data::Load(predictors_file.c_str(), predictors) == SUCCESS_FAIL) {
    FATAL("Unable to open file %s", predictors_file.c_str());
  }

  arma::mat predictions;
  if (data::Load(predictions_file.c_str(), predictions) == SUCCESS_FAIL) {
    FATAL("Unable to open file %s", predictions_file.c_str());
  }

  RidgeRegression engine;
  NOTIFY("Computing Regression...");

  if(!strcmp(mode, "regress")) {

    engine.Init(module, predictors, predictions, 
		!strcmp(fx_param_str(fx_root, "inversion_method",
                                     "normalsvd"), "normalsvd"));
    engine.QRRegress(lambda_min);
  }
  else if(!strcmp(mode, "cvregress")) {
    NOTIFY("Crossvalidating for the optimal lambda in [ %g %g ] by trying \
%d values...", lambda_min, lambda_max, num_lambdas_to_cv);
    engine.Init(module, predictors, predictions);
    engine.CrossValidatedRegression(lambda_min, lambda_max, num_lambdas_to_cv);
  }
  else if(!strcmp(mode, "fsregress")) {

    NOTIFY("Feature selection based regression.\n");

    arma::mat predictor_indices_intermediate;
    arma::mat prune_predictor_indices_intermediate;
    std::string predictor_indices_file = fx_param_str_req(module, 
							  "predictor_indices");
    std::string prune_predictor_indices_file = 
      fx_param_str_req(module, "prune_predictor_indices");
    if(data::Load(predictor_indices_file.c_str(), 
	  predictor_indices_intermediate) == SUCCESS_FAIL) {
      FATAL("Unable to open file %s", predictor_indices_file.c_str());
    }
    if(data::Load(prune_predictor_indices_file.c_str(),
	  prune_predictor_indices_intermediate) == SUCCESS_FAIL) {
      FATAL("Unable to open file %s", prune_predictor_indices_file.c_str());
    }

    arma::Col<index_t> predictor_indices;
    arma::Col<index_t> prune_predictor_indices;
    predictor_indices.zeros(predictor_indices_intermediate.n_cols);
    prune_predictor_indices.zeros
      (prune_predictor_indices_intermediate.n_cols);
    
    // This is a pretty retarded way of copying from a double-matrix
    // to an int vector. This can be simplified only if there were a
    // way to read integer-based dataset directly without typecasting.
    for(index_t i = 0; i < predictor_indices_intermediate.n_cols; i++) {
      predictor_indices[i] =
	(index_t) predictor_indices_intermediate(0, i);
    }
    for(index_t i = 0; i < prune_predictor_indices_intermediate.n_cols;
	i++) {
      prune_predictor_indices[i] = (index_t)
	prune_predictor_indices_intermediate(0, i);
    }
    
    // Run the feature selection.
    arma::Col<index_t> output_predictor_indices;
    engine.Init(module, predictors, predictor_indices, predictions,
		!strcmp(fx_param_str(fx_root, "inversion_method", 
				     "normalsvd"), "normalsvd"));
    engine.FeatureSelectedRegression(predictor_indices,
				     prune_predictor_indices,
				     predictions,
				     &output_predictor_indices);
  }

  NOTIFY("Ridge Regression Model Training Complete!");
  double square_error = engine.ComputeSquareError();
  NOTIFY("Square Error:%g", square_error);
  fx_result_double(module, "square error", square_error);
  arma::mat factors;
  engine.factors(&factors);
  std::string factors_file = fx_param_str(module, "factors", "factors.csv");
  NOTIFY("Saving factors...");
  data::Save(factors_file.c_str(), factors);

  fx_done(module);
  return 0;
}
