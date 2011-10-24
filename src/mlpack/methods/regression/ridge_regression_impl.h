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
#ifndef __RIDGE_REGRESSCLIN_IMPL_H
#define __RIDGE_REGRESSCLIN_IMPL_H

namespace mlpack {
namespace regression {

void RidgeRegression::BuildCovariance_
(const arma::mat &input_data, const arma::Col<size_t> *predictor_indices,
 const arma::mat& predictions_in) {

  //mlpack::Log::Info << "RidgeRegression::BuildCovariance_:starting." << std::endl;

  // Initialize the covariance matrix to the zero matrix.
  covariance_.zeros(input_data.n_rows + 1, input_data.n_rows + 1);

  // Set the alias to the input data.
  predictors_ = &input_data;

  // Initialiez the target training value matrix.
  predictions_.ones(input_data.n_cols, 1);

  int limit = -1;
  if(predictor_indices == NULL) {
    limit = input_data.n_rows;
  }
  else {
    limit = predictor_indices->n_elem;
  }

  // Loop over each column point.
  for(size_t i = 0; i < input_data.n_cols; i++) {

    // Copy over the training target values.
    predictions_(i, 0)= predictions_in[i];

    // Loop over each predictor index.
    for(int j = -1; j < limit; j++) {

      // The current predictor index.
      int outer_predictor_index = -1;
      double outer_value = 1.0;

      if(j >= 0) {

	if(predictor_indices == NULL) {
	  outer_predictor_index = j;
	}
	else {
	  outer_predictor_index = (*predictor_indices)[j];
	}
	outer_value = input_data(outer_predictor_index,i);
      }

      // Accumulate the covariance matrix.
      for(int k = -1; k < limit; k++) {

	// The inner predictor index.
	int inner_predictor_index = -1;
	double inner_value = 1.0;

	if(k >= 0) {
	  if(predictor_indices == NULL) {
	    inner_predictor_index = k;
	  }
	  else {
	    inner_predictor_index = (*predictor_indices)[k];
	  }
	  inner_value = input_data(inner_predictor_index,i);
	}

	covariance_(outer_predictor_index + 1, inner_predictor_index + 1 ) =
	   covariance_(outer_predictor_index + 1,
			   inner_predictor_index + 1) +
	   outer_value * inner_value;

      }
    }
  }
  //mlpack::Log::Info << "RidgeRegression::BuildCovariance_:complete." << std::endl;
}

void RidgeRegression::ExtractDesignMatrixSubset_
(const arma::Col<size_t> *loo_current_predictor_indices,
 arma::mat *extracted_design_matrix_subset) {

  size_t num_features = 0;

  if(loo_current_predictor_indices == NULL) {
    num_features = predictors_->n_rows;
  }
  else {
    num_features = loo_current_predictor_indices->n_elem;
  }

  // You have to add 1 to take the constant term into account.
  extracted_design_matrix_subset->zeros(predictors_->n_cols, num_features + 1);

  for(size_t i = 0; i < predictors_->n_cols; i++) {

    // Set the zero-th column of every row to 1.
    (*extracted_design_matrix_subset)(i, 0) = 1.0;

    for(size_t j = 0; j < num_features; j++) {

      if(loo_current_predictor_indices != NULL) {
	(*extracted_design_matrix_subset)(i, j + 1) =
	   (*predictors_)((*loo_current_predictor_indices)[j], i);
      }
      else {
	// ???
	(*extracted_design_matrix_subset)(i, j + 1) = (*predictors_)(j, i);
      }
    }
  }
}

void RidgeRegression::ExtractCovarianceSubset_
(const arma::mat &precomputed_covariance,
 const arma::Col<size_t> *loo_current_predictor_indices,
 arma::mat *precomputed_covariance_subset) {

  // If no indices are specified, then copy over the entire thing.
  if(loo_current_predictor_indices == NULL) {
    (*precomputed_covariance_subset) = covariance_;
  }

  else {
    precomputed_covariance_subset->zeros
      (loo_current_predictor_indices->n_elem + 1,
       loo_current_predictor_indices->n_elem + 1);

    for(size_t i = 0; i-1 < loo_current_predictor_indices->n_elem; i++) {
      size_t column_position = (i == 0) ?
	0:(*loo_current_predictor_indices)[i-1] + 1;

      for(size_t j = 0; j-1 < loo_current_predictor_indices->n_elem; j++) {
	size_t row_position = (j == 0) ?
	  0:(*loo_current_predictor_indices)[j-1] + 1;

	// ???
	(*precomputed_covariance_subset)(j-1 + 1, i-1 + 1) =
	   precomputed_covariance(row_position, column_position);
      }
    }
  }
}

RidgeRegression::RidgeRegression(const arma::mat &predictors,
                           const arma::mat &predictions,
			   bool use_normal_equation_method) {
  if (predictors.n_cols < predictors.n_rows)
    mlpack::Log::Fatal << "The number of columns (" << predictors.n_cols <<
        ") must be less than or equal to the number of the rows (" <<
        predictors.n_rows << ")." << std::endl;
  if (predictions.n_rows > 1)
    mlpack::Log::Fatal << "The current implementation only supports "
        "one-dimensional predictions." << std::endl;
  if (predictors.n_cols != predictions.n_cols)
    mlpack::Log::Fatal << "Predictors and predictions must have the same number "
        "of columns (" << predictors.n_cols << " != " << predictions.n_cols <<
        ")." << std::endl;

  if(use_normal_equation_method) {

    // Build the covariance matrix.
    BuildCovariance_(predictors, NULL, predictions);
  }
  else {

    // The covariance is empty.
    BuildDesignMatrixFromIndexSet_(predictors, predictions, NULL);
    covariance_.reset();
  }

  // Initialize the factor to be empty.
  factors_.reset();
}

RidgeRegression::RidgeRegression(const arma::mat &input_data,
                           const arma::Col<size_t> &predictor_indices,
                           size_t &prediction_index,
			   bool use_normal_equation_method) {

  if(use_normal_equation_method) {

    // Build the covariance matrix.
    BuildCovariance_(input_data, &predictor_indices,
		     input_data.col(prediction_index));
  }
  else {

    // Set the covariance stuff to be empty.
    BuildDesignMatrixFromIndexSet_(input_data,
				   input_data.col(prediction_index),
				   &predictor_indices);
    covariance_.reset();
  }

  // Initialize the factor to be empty.
  factors_.reset();
}

RidgeRegression::RidgeRegression(const arma::mat &input_data,
                           const arma::Col<size_t> &predictor_indices,
                           const arma::mat &predictions,
			   bool use_normal_equation_method) {

  if(use_normal_equation_method) {

    // Build the covariance matrix.
    BuildCovariance_(input_data, &predictor_indices, predictions);
  }
  else {

    // Set the covariance stuff to be empty.
    BuildDesignMatrixFromIndexSet_(input_data, predictions,
				   &predictor_indices);
    covariance_.reset();
  }

  // Initialize the factor to be empty.
  factors_.reset();
}

inline void RidgeRegression::ReInitTargetValues(const arma::mat &input_data,
					 const size_t target_value_index) {

  predictions_.col(0) = trans(input_data.row(target_value_index));
}

inline void RidgeRegression::ReInitTargetValues(const arma::mat &input_data) {
  ReInitTargetValues(input_data, 0);
}

void RidgeRegression::BuildDesignMatrixFromIndexSet_
(const arma::mat &input_data, const arma::mat& predictions,
 const arma::Col<size_t> *predictor_indices) {

  // We just have an alias to the input data, and copy the training
  // values.
  predictors_ = &input_data;
//  predictions_.Init(input_data.n_cols, 1);
  predictions_.zeros(input_data.n_cols, 1);

  for(size_t i = 0; i < input_data.n_cols; i++) {
    predictions_(i, 0) = predictions[i];
  }
}

void RidgeRegression::ComputeLinearModel_
(double lambda_sq, const arma::vec &singular_values,
 const arma::mat &u, const arma::mat &v_t, size_t num_features) {

  // Factors should have $D + 1$ parameters.
//  factors_.reset();
  //factors_.zeros(num_features + 1, 1);
  factors_.zeros(v_t.n_cols, 1);

  for(size_t i = 0; i < singular_values.n_elem; i++) {
    double s_sq = pow(singular_values[i], 2);
    double alpha = singular_values[i] / (lambda_sq + s_sq) *
      //la::Dot(u.n_rows, u.GetColumnPtr(i), predictions_.ptr());
      arma::dot(u.col(i),predictions_.rows(0,u.n_rows-1));

    // Scale each row vector of V^T and add to the factor.
    for(size_t j = 0; j < v_t.n_cols && j < factors_.n_rows; j++) {
      factors_(j, 0) = factors_(j, 0) + alpha * v_t(i, j);
    }
  }
}

void RidgeRegression::QRRegress
(double lambda, const arma::Col<size_t> *predictor_indices) {

  // THIS FUNCTCLIN DOES NOT TAKE lambda into ACCOUNT YET! FIX ME!
  //mlpack::Log::Info << "QRRegress: starting." << std::endl;

  // At this point, QR should not be used when the covariance based
  // method is used! Only do QR on the design matrix.
  arma::mat extracted_design_matrix_subset;
  ExtractDesignMatrixSubset_(predictor_indices,
			     &extracted_design_matrix_subset);
  arma::mat q, r;
  arma::qr(q,r,extracted_design_matrix_subset); // Sets q and r to have 0 elements on failure

 // if(q.n_elem == 0) {
    //mlpack::Log::Info << "QRRegress: QR decomposition encountered problems!" << std::endl;
  //}

  // Multiply the target training values by the Q^T and solve the
  // resulting triangular system.
  arma::mat q_transpose_y;
  q_transpose_y = trans(q) * predictions_;

  factors_.reset();
  factors_ = arma::solve(r,q_transpose_y);

  //mlpack::Log::Info << "QRRegress: complete." << std::endl;
}

void RidgeRegression::SVDRegress
(double lambda, const arma::Col<size_t> *predictor_indices) {

  //mlpack::Log::Info << "SVDRegress: starting." << std::endl;

  arma::vec singular_values;
  arma::mat v_t, u;
  ExtractSubspace_(u, singular_values, v_t, predictor_indices);

  double lambda_sq = lambda * lambda;

  ComputeLinearModel_(lambda_sq, singular_values, u, v_t,
		      (predictor_indices == NULL) ?
		      predictors_->n_rows:
		      (predictor_indices->n_elem));

  //mlpack::Log::Info << "SVDRegress: complete." << std::endl;
}

void RidgeRegression::ExtractSubspace_
(arma::mat& u, arma::vec& singular_values, arma::mat& v_t,
 const arma::Col<size_t> *predictor_indices) {

  if(covariance_.n_rows > 0) {

    arma::mat eigen_v;
    arma::mat precomputed_covariance;
    ExtractCovarianceSubset_(covariance_, predictor_indices,
			     &precomputed_covariance);
    arma::svd( eigen_v, singular_values, v_t, precomputed_covariance);

    // Take the square root of each eigenvalue to get the singular
    // values.
    for(size_t i = 0; i < singular_values.n_elem; i++) {
      singular_values(i) = sqrt(singular_values(i));
    }

    u.zeros(predictors_->n_cols, eigen_v.n_cols);

    int limit = -1;
    if(predictor_indices == NULL) {
      limit = predictors_->n_rows;
    }
    else {
      limit = predictor_indices->n_elem;
    }

    for(size_t i = 0; i < predictors_->n_cols; i++) {

      const arma::vec& point = predictors_->col(i);
      for(size_t j = 0; j < eigen_v.n_cols; j++) {

	const arma::vec& eigen_v_column = eigen_v.col(j);
	double dot_product = eigen_v_column(0);
	for(int k = 1; k <= limit; k++) {

	  if(predictor_indices == NULL) {
	    dot_product += point(k - 1) * eigen_v_column(k);
	  }
	  else {
	    dot_product += point((*predictor_indices)(k - 1)) *
	      eigen_v_column(k);
	  }
	}
	u(i, j) = dot_product;
      }
    }

    for(size_t i = 0; i < u.n_cols; i++) {
      //arma::vec& u_column = u->col(i);
      if(singular_values(i) > 0) {
	u.col(i) *= 1.0/singular_values(i);
      }
    }
  }
  else {
    arma::svd( u, singular_values, v_t, *predictors_);
  }
}

void RidgeRegression::CrossValidatedRegression(double lambda_min,
					       double lambda_max,
					       size_t num) {
  if (lambda_min > lambda_max)
    mlpack::Log::Fatal << "lambda_max (" << lambda_max << ") must be larger than"
        " lambda_min (" << lambda_min << ")." << std::endl;

  double step = (lambda_max - lambda_min) / num;
  arma::vec singular_values;
  arma::mat u, v_t;

  // Compute the SVD and extract the left/right singular vectors and
  // singular values.
  ExtractSubspace_(u, singular_values, v_t, NULL);

  // Square the singular values and store it.
  arma::vec singular_values_sq;
  singular_values_sq = singular_values;
  for(size_t i = 0; i < singular_values.n_elem; i++) {
    singular_values_sq[i] = pow(singular_values[i], 2);
  }

  // u_x_b will be a vector of length s such that each entry is a dot
  // product between the $i$-th left singular vector and the
  // predictions_ values.
  arma::mat u_x_b;
  u_x_b = arma::trans(u) * predictions_;
  double min_score = DBL_MAX;
  size_t min_index = -1;

  arma::mat error;
  error.zeros(1, predictors_->n_cols);

  // Try different values of lambda and choose the best one that
  // minimizes the loss function.
  for(size_t i = 0; i < num; i++) {
    double lambda = lambda_min + i * step;
    double lambda_sq = pow(lambda, 2);

    // compute residual error
    error.zeros();

    // tau starts from the number of columns of predictors_ minus one
    // because we append a column of 1's at the start to the
    // dimensionality of the problem.
    double tau = predictors_->n_cols - 1;
    for(size_t j = 0; j < singular_values_sq.n_elem; j++) {
      double alpha = lambda_sq / (singular_values_sq[j] + lambda_sq);
//      la::AddExpert(error.n_cols, alpha * u_x_b(j, 0),
//                    u.GetColumnPtr(j), error.ptr());
      error += (alpha * u_x_b(j,0)) * u.col(j);
      // compute tau
      tau -= singular_values_sq[j] / (singular_values_sq[j] + lambda_sq);
    }
    double rss = dot(error, error);

    // Here we need to add to residual squared error the squared error
    // of the predictions.
    for(size_t j = 0; j < predictions_.n_rows; j++) {
      double accumulant = predictions_(j, 0);

      for(size_t k = 0; k < singular_values_sq.n_elem; k++) {
	accumulant -= u_x_b(k, 0) * u(j, k);
      }
      rss += pow(accumulant, 2);
    }

    double score = rss / pow(tau, 2);

    if(score < min_score) {
      min_score = score;
      min_index = i;
    }
  }
  mlpack::CLI::GetParam<double>("reg/cross_validation_score") =  min_score;

  //mlpack::Log::Info << "The optimal lamda: " <<  lambda_min + min_index * step << std::endl;


  // Using the best lambda, compute the linear model.
  double lambda_sq = pow(lambda_min + min_index * step, 2);
  ComputeLinearModel_(lambda_sq, singular_values, u, v_t,
		      predictors_->n_rows);
}

void RidgeRegression::FeatureSelectedRegression
(const arma::Col<size_t> &predictor_indices,
 const arma::Col<size_t> &prune_predictor_indices,
 const arma::mat &original_target_training_values,
 arma::Col<size_t> *output_predictor_indices) {

  //mlpack::Log::Info << "Starting VIF-based feature selection." << std::endl;

  double lambda = mlpack::CLI::GetParam<double>("ridge/lambda"); //Default value, 0.0
  double variance_inflation_factor_threshold =
    mlpack::CLI::GetParam<double>("ridge/vif_threshold"); //Default value, 8.0;
  bool done_flag = false;
  arma::Col<size_t> *current_predictor_indices = new arma::Col<size_t>();
  arma::Col<size_t> *current_prune_predictor_indices = new
    arma::Col<size_t>();
  (*current_predictor_indices) = predictor_indices;
  (*current_prune_predictor_indices) = prune_predictor_indices;

  do {

    // The maximum variance inflation factor and the index that
    // achieved it.
    double max_variance_inflation_factor = 0.0;
    size_t index_of_max_variance_inflation_factor = -1;

    // Reset the flag to be true.
    done_flag = true;

    // For each of the features in the current list, regress the
    // i-th feature versus the rest of the features and compute its
    // variance inflation factor.
    for(size_t i = 0; i < current_prune_predictor_indices->n_elem; i++) {

      // Take out the current dimension being regressed against from
      // the predictor list to form the leave-one-out predictor
	// list.
      arma::Col<size_t> loo_current_predictor_indices;
      RidgeRegressionUtil::CopyVectorExceptOneIndex_
	(*current_predictor_indices, (*current_prune_predictor_indices)[i],
	 &loo_current_predictor_indices);

      // Initialize the ridge regression model using the
      // leave-one-out predictor indices with the appropriate
      // prediction index.
      ReInitTargetValues
	(*predictors_, (*current_prune_predictor_indices)[i]);

      std::cout << "Current leave one out index: " <<
	     (*current_prune_predictor_indices)[i] << '\n';

      // Do the regression.
      // SVDRegress(lambda, &loo_current_predictor_indices);
      QRRegress(lambda, &loo_current_predictor_indices);

      arma::vec loo_predictions;
      Predict(*predictors_, loo_current_predictor_indices, &loo_predictions);

      // Extract the dimension that is being regressed against and
      // compute the variance inflation factor.
      arma::vec loo_feature;
      loo_feature.zeros(predictors_->n_cols);
      for(size_t j = 0; j < predictors_->n_cols; j++) {
	loo_feature[j] = (*predictors_)
	  ((*current_prune_predictor_indices)[i], j);
      }
      double variance_inflation_factor =
	RidgeRegressionUtil::VarianceInflationFactor(loo_feature,
						     loo_predictions);
     // NOTIFY("The %zu"-th dimension has a variance inflation factor of %g.\n",
//	     (*current_prune_predictor_indices)[i],
//	     variance_inflation_factor);

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

      arma::Col<size_t> *new_predictor_indices = new arma::Col<size_t>();
      RidgeRegressionUtil::CopyVectorExceptOneIndex_
	(*current_predictor_indices, index_of_max_variance_inflation_factor,
	 new_predictor_indices);
      delete current_predictor_indices;
      current_predictor_indices = new_predictor_indices;

      arma::Col<size_t> *new_prune_indices = new arma::Col<size_t>();
      RidgeRegressionUtil::CopyVectorExceptOneIndex_
	(*current_prune_predictor_indices,
	 index_of_max_variance_inflation_factor, new_prune_indices);
      delete current_prune_predictor_indices;
      current_prune_predictor_indices = new_prune_indices;
      done_flag = false;
    }

  } while(!done_flag && current_prune_predictor_indices->n_elem > 1);

  // Copy the output indices and free the temporary index vectors
  // afterwards.
  (*output_predictor_indices) = *current_predictor_indices;
  delete current_predictor_indices;
  delete current_prune_predictor_indices;

  // Change the target training values to the original prediction values.
  ReInitTargetValues(original_target_training_values);
  // SVDRegress(lambda, output_predictor_indices);
  QRRegress(lambda, output_predictor_indices);

  //mlpack::Log::Info << "VIF feature selection complete." << std::endl;
}

double RidgeRegression::ComputeSquareError() {

  arma::mat error;
  error.zeros(predictors_->n_cols, 1);
  for(size_t i = 0; i < predictors_->n_cols; i++) {
    const arma::vec& point = predictors_->col(i);
    error(i, 0) = factors_(0, 0);

    for(size_t j = 0; j < predictors_->n_rows; j++) {
      error(i, 0) =  error(i, 0) + factors_(j + 1, 0) * point[j];
    }
  }
  error -= predictions_;
  double square_error = dot(error, error);
  return square_error;
}

void RidgeRegression::Predict(const arma::mat &dataset,
			      const arma::Col<size_t> &predictor_indices,
			      arma::vec *new_predictions) {

  // (Roughly) take each column of the dataset and compute the
  // dot-product between it and the linear model coefficients, but
  // also need to take care of the intercept part of the coefficients.
  new_predictions->zeros(dataset.n_cols);

  if(predictor_indices.n_elem + 1 != factors_.n_rows) {
    std::cout << "The number of selected indices is not equal to the \
    non-constant coefficients!";
    return;
  }

  for(size_t i = 0; i < dataset.n_cols; i++) {
    (*new_predictions)[i] = factors_(0, 0);

    for(size_t j = 0; j < predictor_indices.n_elem; j++) {
      (*new_predictions)[i] += factors_(j + 1, 0) *
	dataset(predictor_indices[j], i);
    }
  }
}

void RidgeRegression::Predict(const arma::mat &dataset, arma::vec *new_predictions) {

  // (Roughly) take each column of the dataset and compute the
  // dot-product between it and the linear model coefficients, but
  // also need to take care of the intercept part of the coefficients.
  new_predictions->zeros(dataset.n_cols);

  for(size_t i = 0; i < dataset.n_cols; i++) {
    (*new_predictions)[i] = factors_(0, 0);

    for(size_t j = 0; j < dataset.n_rows; j++) {
      (*new_predictions)[i] += factors_(j + 1, 0) * dataset(j, i);
    }
  }
}

void RidgeRegression::factors(arma::mat *factors) {
  (*factors) = factors_;
}

}; // namespace regression
}; // namespace mlpack

#endif
