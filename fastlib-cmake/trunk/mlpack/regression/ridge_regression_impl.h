/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
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
/*
 * ============================================================================
 * 
 *       Filename:  ridge_regression_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/15/2009 12:19:22 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 * 
 * ============================================================================
*/ 
#ifndef RIDGE_REGRESSION_H_
//COMPILER_PRINTF("%s", "!!!!!You have accidently included ridge_regression_impl.h "
//    "Fix it otherwise the program will behave unexpectedly");                
#else

void RidgeRegression::BuildCovariance_
(const Matrix &input_data, const GenVector<index_t> *predictor_indices,
 const double *predictions_in) {
  
  NOTIFY("RidgeRegression::BuildCovariance_: starting.");

  // Initialize the covariance matrix to the zero matrix.
  covariance_.Init(input_data.n_rows() + 1, input_data.n_rows() + 1);
  covariance_.SetZero();
  
  // Set the alias to the input data.
  predictors_.Alias(input_data);

  // Initialiez the target training value matrix.
  predictions_.Init(input_data.n_cols(), 1);
    
  int limit = -1;
  if(predictor_indices == NULL) {
    limit = input_data.n_rows();
  }
  else {
    limit = predictor_indices->length();
  }
  
  // Loop over each column point.
  for(index_t i = 0; i < input_data.n_cols(); i++) {
    
    // The pointer to the point in consideration.
    const double *point = input_data.GetColumnPtr(i);

    // Copy over the training target values.
    predictions_.set(i, 0, predictions_in[i]);

    // Loop over each predictor index.
    for(index_t j = -1; j < limit; j++) {
      
      // The current predictor index.
      index_t outer_predictor_index = -1;
      double outer_value = 1.0;
      
      if(j >= 0) {
	if(predictor_indices == NULL) {
	  outer_predictor_index = j;
	}
	else {
	  outer_predictor_index = (*predictor_indices)[j];
	}
	outer_value = point[outer_predictor_index];
      }
      
      // Accumulate the covariance matrix.
      for(index_t k = -1; k < limit; k++) {
	
	// The inner predictor index.
	index_t inner_predictor_index = -1;
	double inner_value = 1.0;
	
	if(k >= 0) {
	  if(predictor_indices == NULL) {
	    inner_predictor_index = k;
	  }
	  else {
	    inner_predictor_index = (*predictor_indices)[k];
	  }
	  inner_value = point[inner_predictor_index];
	}
	
	covariance_.set
	  (outer_predictor_index + 1, inner_predictor_index + 1,
	   covariance_.get(outer_predictor_index + 1,
			   inner_predictor_index + 1) +
	   outer_value * inner_value);
      }
    }
  }
  NOTIFY("RidgeRegression::BuildCovariance_: complete.");
}

void RidgeRegression::ExtractCovarianceSubset_
(const Matrix &precomputed_covariance,
 const GenVector<index_t> *loo_current_predictor_indices,
 Matrix *precomputed_covariance_subset) {

  // If no indices are specified, then copy over the entire thing.
  if(loo_current_predictor_indices == NULL) {    
    precomputed_covariance_subset->Copy(covariance_);
  }

  else {
    precomputed_covariance_subset->Init
      (loo_current_predictor_indices->length() + 1,
       loo_current_predictor_indices->length() + 1);
    
    for(index_t i = -1; i < loo_current_predictor_indices->length(); i++) {
      index_t column_position = (i == -1) ? 
	0:(*loo_current_predictor_indices)[i] + 1;
      
      for(index_t j = -1; j < loo_current_predictor_indices->length(); j++) {
	index_t row_position = (j == -1) ? 
	  0:(*loo_current_predictor_indices)[j] + 1;
	
	precomputed_covariance_subset->set
	  (j + 1, i + 1, 
	   precomputed_covariance.get(row_position, column_position));
      }
    }
  }
}

void RidgeRegression::Init(fx_module *module, const Matrix &predictors, 
                           const Matrix &predictions,
			   bool use_normal_equation_method) {

  module_ = module;
  DEBUG_ERROR_MSG_IF(predictors.n_cols()<predictors.n_rows(),
     "The number of the columns %"LI"d must be less or equal to the number of "
     " the rows %"LI"d ", predictors.n_cols(), predictors.n_rows());
  DEBUG_ERROR_MSG_IF(predictions.n_rows() >1, 
      "The current implementation supports only one dimensional predictions");
  DEBUG_ERROR_MSG_IF(predictors.n_cols()!=predictions.n_cols(), 
      "Predictors and predictions must have the same same number "
      "of rows %"LI"d != %"LI"d ", predictors.n_cols(), predictions.n_cols());

  if(use_normal_equation_method) {

    // Build the covariance matrix.
    BuildCovariance_(predictors, NULL, predictions.ptr());
  }
  else {
    
    // The covariance is empty.
    BuildDesignMatrixFromIndexSet_(predictors, predictions.ptr(), NULL);
    covariance_.Init(0, 0);
  }

  // Initialize the factor to be empty.
  factors_.Init(0, 0);
}

void RidgeRegression::Init(fx_module *module, 
                           const Matrix &input_data, 
                           const GenVector<index_t> &predictor_indices,
                           index_t &prediction_index,
			   bool use_normal_equation_method) {
  
  module_ = module;
 
  if(use_normal_equation_method) {

    // Build the covariance matrix.
    BuildCovariance_(input_data, &predictor_indices,
		     input_data.GetColumnPtr(prediction_index));
  }
  else {
    
    // Set the covariance stuff to be empty.
    BuildDesignMatrixFromIndexSet_(input_data, 
				   input_data.GetColumnPtr(prediction_index),
				   &predictor_indices);
    covariance_.Init(0, 0);
  }

  // Initialize the factor to be empty.
  factors_.Init(0, 0);
}

void RidgeRegression::Init(fx_module *module, 
                           const Matrix &input_data, 
                           const GenVector<index_t> &predictor_indices,
                           const Matrix &predictions,
			   bool use_normal_equation_method) {

  module_ = module;

  if(use_normal_equation_method) {

    // Build the covariance matrix.
    BuildCovariance_(input_data, &predictor_indices, predictions.ptr());
  }
  else {
    
    // Set the covariance stuff to be empty.
    BuildDesignMatrixFromIndexSet_(input_data, predictions.ptr(),
				   &predictor_indices);
    covariance_.Init(0, 0);
  }

  // Initialize the factor to be empty.
  factors_.Init(0, 0);
}

void RidgeRegression::ReInitTargetValues(const Matrix &input_data,
					 index_t target_value_index) {

  for(index_t i = 0; i < predictions_.n_rows(); i++) {
    predictions_.set(i, 0, input_data.get(target_value_index, i));
  }  
}

void RidgeRegression::ReInitTargetValues(const Matrix &target_values_in) {

  for(index_t i = 0; i < predictions_.n_rows(); i++) {
    predictions_.set(i, 0, target_values_in.get(0, i));
  }
}

void RidgeRegression::BuildDesignMatrixFromIndexSet_
(const Matrix &input_data, const double *predictions,
 const GenVector<index_t> *predictor_indices) {
  
  // Extract the rows that are relevant, and construct the appropriate
  // design matrix from it. We add one more parameter to the model for
  // the intercept (the constant term). Here we are transposing the
  // dataset.
  int num_features = 0;
  if(predictor_indices == NULL) {
    num_features = input_data.n_rows();
  }
  else {
    num_features = predictor_indices->length();
  }  
  predictors_.Init(input_data.n_cols(), num_features + 1);
  predictions_.Init(input_data.n_cols(), 1);
  
  for(index_t i = 0; i < input_data.n_cols(); i++) {
    
    // Initialize the first column of the design matrix to be 1.
    predictors_.set(i, 0, 1.0);

    for(index_t j = 0; j < num_features; j++) {
      int get_feature_index = -1;
      if(predictor_indices == NULL) {
	get_feature_index = j;
      }
      else {
	get_feature_index = (*predictor_indices)[j];
      }
      predictors_.set(i, j + 1, input_data.get(get_feature_index, i));
    }

    // Copy over the predictions.
    predictions_.set(i, 0, predictions[i]);
  } 
}

void RidgeRegression::ComputeLinearModel_
(double lambda_sq, const Vector &singular_values, 
 const Matrix &u, const Matrix &v_t, int num_features) {

  // Factors should have $D + 1$ parameters.
  factors_.Destruct();
  factors_.Init(num_features + 1, 1);
  factors_.SetZero();

  for(index_t i = 0; i < singular_values.length(); i++) {
    double s_sq = math::Sqr(singular_values[i]);
    double alpha = singular_values[i] / (lambda_sq + s_sq) * 
      la::Dot(u.n_rows(), u.GetColumnPtr(i), predictions_.ptr());

    // Scale each row vector of V^T and add to the factor.
    for(index_t j = 0; j < v_t.n_cols(); j++) {
      factors_.set(j, 0, factors_.get(j, 0) + alpha * v_t.get(i, j));
    }
  }
}

void RidgeRegression::SVDRegress
(double lambda, const GenVector<index_t> *predictor_indices) {

  NOTIFY("SVDRegress: starting.");

  Vector singular_values;
  Matrix v_t, u;
  ExtractSubspace_(&u, &singular_values, &v_t, predictor_indices);
  
  double lambda_sq = lambda * lambda;

  ComputeLinearModel_(lambda_sq, singular_values, u, v_t,
		      (predictor_indices == NULL) ?
		      predictors_.n_rows():
		      (predictor_indices->length()));

  NOTIFY("SVDRegress: complete.");
}

void RidgeRegression::ExtractSubspace_
(Matrix *u, Vector *singular_values, Matrix *v_t,
 const GenVector<index_t> *predictor_indices) {

  if(covariance_.n_rows() > 0) {

    Matrix eigen_v;
    Matrix precomputed_covariance;
    ExtractCovarianceSubset_(covariance_, predictor_indices, 
			     &precomputed_covariance);
    la::SVDInit(precomputed_covariance, singular_values, &eigen_v, v_t);
    
    // Take the square root of each eigenvalue to get the singular
    // values.
    for(index_t i = 0; i < singular_values->length(); i++) {
      (*singular_values)[i] = sqrt((*singular_values)[i]);
    }
    
    u->Init(predictors_.n_cols(), eigen_v.n_cols());
    
    int limit = -1;
    if(predictor_indices == NULL) {
      limit = predictors_.n_rows();
    }
    else {
      limit = predictor_indices->length();
    }
    
    for(index_t i = 0; i < predictors_.n_cols(); i++) {
      
      const double *point = predictors_.GetColumnPtr(i);
      for(index_t j = 0; j < eigen_v.n_cols(); j++) {
	
	const double *eigen_v_column = eigen_v.GetColumnPtr(j);
	double dot_product = eigen_v_column[0];
	for(index_t k = 1; k <= limit; k++) {
	  
	  if(predictor_indices == NULL) {
	    dot_product += point[k - 1] * eigen_v_column[k];
	  }
	  else {
	    dot_product += point[(*predictor_indices)[k - 1]] *
	      eigen_v_column[k];
	  }
	}      
	u->set(i, j, dot_product);
      }
    }
    
    for(index_t i = 0; i < u->n_cols(); i++) {
      double *u_column = u->GetColumnPtr(i);
      if((*singular_values)[i] > 0) {
	la::Scale(u->n_rows(), 1.0 / (*singular_values)[i], u_column);
      }
    }
  }
  else {
    la::SVDInit(predictors_, singular_values, u, v_t);
  }
}

void RidgeRegression::CrossValidatedRegression(double lambda_min, 
					       double lambda_max,
					       index_t num) {
  DEBUG_ERROR_MSG_IF(lambda_min > lambda_max, 
		     "lambda_max %lg must be larger than lambda_min %lg",
		     lambda_max, lambda_min );
  double step = (lambda_max - lambda_min) / num;
  Vector singular_values;
  Matrix u, v_t;

  // Compute the SVD and extract the left/right singular vectors and
  // singular values.
  ExtractSubspace_(&u, &singular_values, &v_t, NULL);

  // Square the singular values and store it.
  Vector singular_values_sq;
  singular_values_sq.Copy(singular_values);
  for(index_t i = 0; i < singular_values.length(); i++) {
    singular_values_sq[i] = math::Sqr(singular_values[i]);
  }

  // u_x_b will be a vector of length s such that each entry is a dot
  // product between the $i$-th left singular vector and the
  // predictions_ values.
  Matrix u_x_b;
  la::MulTransAInit(u, predictions_, &u_x_b);
  double min_score = DBL_MAX;
  index_t min_index = -1;

  Matrix error;
  error.Init(1, predictors_.n_cols());

  // Try different values of lambda and choose the best one that
  // minimizes the loss function.
  for(index_t i = 0; i < num; i++) {
    double lambda = lambda_min + i * step;
    double lambda_sq = math::Sqr(lambda);

    // compute residual error
    error.SetZero();

    // tau starts from the number of columns of predictors_ minus one
    // because we append a column of 1's at the start to the
    // dimensionality of the problem.
    double tau = predictors_.n_cols() - 1;
    for(index_t j = 0; j < singular_values_sq.length(); j++) {
      double alpha = lambda_sq / (singular_values_sq[j] + lambda_sq);
      la::AddExpert(error.n_cols(), alpha * u_x_b.get(j, 0), 
                    u.GetColumnPtr(j), error.ptr());
      // compute tau
      tau -= singular_values_sq[j] / (singular_values_sq[j] + lambda_sq);
    }
    double rss = la::Dot(error, error);

    // Here we need to add to residual squared error the squared error
    // of the predictions.
    for(index_t j = 0; j < predictions_.n_rows(); j++) {
      double accumulant = predictions_.get(j, 0);
      
      for(index_t k = 0; k < singular_values_sq.length(); k++) {
	accumulant -= u_x_b.get(k, 0) * u.get(j, k);
      }
      rss += math::Sqr(accumulant);
    }

    double score = rss / math::Sqr(tau);

    if(score < min_score) {
      min_score = score;
      min_index = i;
    }
  }
  fx_result_double(module_, "cross_validation_score", min_score);

  NOTIFY("The optimal lambda: %g...\n", lambda_min + min_index * step);

  // Using the best lambda, compute the linear model.
  double lambda_sq = math::Sqr(lambda_min + min_index * step);
  ComputeLinearModel_(lambda_sq, singular_values, u, v_t,
		      predictors_.n_rows());
}

void RidgeRegression::FeatureSelectedRegression
(const GenVector<index_t> &predictor_indices, 
 const GenVector<index_t> &prune_predictor_indices,
 const Matrix &original_target_training_values,
 GenVector<index_t> *output_predictor_indices) {
  
  NOTIFY("Starting VIF-based feature selection.");
  
  double lambda = fx_param_double(module_, "lambda", 0.0);
  double variance_inflation_factor_threshold = 
    fx_param_double(module_, "vif_threshold", 8.0);
  bool done_flag = false;
  GenVector<index_t> *current_predictor_indices = new GenVector<index_t>();
  GenVector<index_t> *current_prune_predictor_indices = new 
    GenVector<index_t>();
  current_predictor_indices->Copy(predictor_indices);
  current_prune_predictor_indices->Copy(prune_predictor_indices);
  
  do {
    
    // The maximum variance inflation factor and the index that
    // achieved it.
    double max_variance_inflation_factor = 0.0;
    index_t index_of_max_variance_inflation_factor = -1;
    
    // Reset the flag to be true.
    done_flag = true;
    
    // For each of the features in the current list, regress the
    // i-th feature versus the rest of the features and compute its
    // variance inflation factor.
    for(index_t i = 0; i < current_prune_predictor_indices->length(); i++) {
      
      // Take out the current dimension being regressed against from
      // the predictor list to form the leave-one-out predictor
	// list.
      GenVector<index_t> loo_current_predictor_indices;
      RidgeRegressionUtil::CopyVectorExceptOneIndex_
	(*current_predictor_indices, (*current_prune_predictor_indices)[i],
	 &loo_current_predictor_indices);
      
      // Initialize the ridge regression model using the
      // leave-one-out predictor indices with the appropriate
      // prediction index.
      ReInitTargetValues
	(predictors_, (*current_prune_predictor_indices)[i]);
      
      // Do the regression.
      SVDRegress(lambda, &loo_current_predictor_indices);

      Vector loo_predictions;
      Predict(predictors_, loo_current_predictor_indices, &loo_predictions);
      
      // Extract the dimension that is being regressed against and
      // compute the variance inflation factor.
      Vector loo_feature;
      loo_feature.Init(predictors_.n_cols());
      for(index_t j = 0; j < predictors_.n_cols(); j++) {
	loo_feature[j] = predictors_.get
	  ((*current_prune_predictor_indices)[i], j);
      }
      double variance_inflation_factor = 
	RidgeRegressionUtil::VarianceInflationFactor(loo_feature,
						     loo_predictions);
      
      NOTIFY("The %d-th dimension has a variance inflation factor of %g.\n",
	     (*current_prune_predictor_indices)[i], 
	     variance_inflation_factor);
      
      if(variance_inflation_factor > max_variance_inflation_factor) {
	max_variance_inflation_factor = variance_inflation_factor;
	index_of_max_variance_inflation_factor = 
	  (*current_prune_predictor_indices)[i];
      }
      
    } // end of iterating over each feature that is being considered
    // for pruning...
    
    // If the maximum variance inflation factor exceeded the
    // threshold, then eliminate it from the current list of
    // features, and the do-while loop repeats.
    if(max_variance_inflation_factor > variance_inflation_factor_threshold) {
      
      GenVector<index_t> *new_predictor_indices = new GenVector<index_t>();
      RidgeRegressionUtil::CopyVectorExceptOneIndex_
	(*current_predictor_indices, index_of_max_variance_inflation_factor,
	 new_predictor_indices);
      delete current_predictor_indices;
      current_predictor_indices = new_predictor_indices;
      
      GenVector<index_t> *new_prune_indices = new GenVector<index_t>();
      RidgeRegressionUtil::CopyVectorExceptOneIndex_
	(*current_prune_predictor_indices, 
	 index_of_max_variance_inflation_factor, new_prune_indices);
      delete current_prune_predictor_indices;
      current_prune_predictor_indices = new_prune_indices;
      done_flag = false;
    }
    
  } while(!done_flag && current_prune_predictor_indices->length() > 1);
  
  // Copy the output indices and free the temporary index vectors
  // afterwards.
  output_predictor_indices->Copy(*current_predictor_indices);
  delete current_predictor_indices;
  delete current_prune_predictor_indices;

  // Change the target training values to the original prediction values.
  ReInitTargetValues(original_target_training_values);
  SVDRegress(lambda, output_predictor_indices);
  
  NOTIFY("VIF feature selection complete.");
}

double RidgeRegression::ComputeSquareError() {
  Matrix error;
  error.Init(predictors_.n_cols(), 1);
  for(index_t i = 0; i < predictors_.n_cols(); i++) {
    const double *point = predictors_.GetColumnPtr(i);
    error.set(i, 0, factors_.get(0, 0));

    for(index_t j = 0; j < predictors_.n_rows(); j++) {
      error.set(i, 0, error.get(i, 0) + factors_.get(j + 1, 0) * point[j]);
    }
  }
  la::SubFrom(predictions_, &error);
  double square_error = la::Dot(error.n_rows(), error.ptr(), error.ptr());
  return square_error;
}

void RidgeRegression::Predict(const Matrix &dataset, 
			      const GenVector<index_t> &predictor_indices,
			      Vector *new_predictions) {

  // (Roughly) take each column of the dataset and compute the
  // dot-product between it and the linear model coefficients, but
  // also need to take care of the intercept part of the coefficients.
  new_predictions->Init(dataset.n_cols());

  if(predictor_indices.length() + 1 != factors_.n_rows()) {
    printf("The number of selected indices is not equal to the ");
    printf("non-constant coefficients!\n");
    return;
  }

  for(index_t i = 0; i < dataset.n_cols(); i++) {
    (*new_predictions)[i] = factors_.get(0, 0);

    for(index_t j = 0; j < predictor_indices.length(); j++) {
      (*new_predictions)[i] += factors_.get(j + 1, 0) * 
	dataset.get(predictor_indices[j], i);
    }
  }
}

void RidgeRegression::Predict(const Matrix &dataset, Vector *new_predictions) {
  
  // (Roughly) take each column of the dataset and compute the
  // dot-product between it and the linear model coefficients, but
  // also need to take care of the intercept part of the coefficients.
  new_predictions->Init(dataset.n_cols());

  for(index_t i = 0; i < dataset.n_cols(); i++) {
    (*new_predictions)[i] = factors_.get(0, 0);

    for(index_t j = 0; j < dataset.n_rows(); j++) {
      (*new_predictions)[i] += factors_.get(j + 1, 0) * dataset.get(j, i);
    }
  }
}

void RidgeRegression::factors(Matrix *factors) {
  factors->Copy(factors_);
}

#endif
