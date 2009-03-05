#ifndef RIDGE_REGRESSION_UTIL_H
#define RIDGE_REGRESSION_UTIL_H

#include "ridge_regression.h"

class RidgeRegressionUtil {

 private:

  template<typename T>
  static void CopyVectorExceptOneIndex_(const GenVector<T> &source,
					index_t exclude_index,
					GenVector<T> *destination) {
    destination->Init(source.length() - 1);
    index_t current_index = 0;

    for(index_t j = 0; j < source.length(); j++) {
      if(source[j] != exclude_index) {
	(*destination)[current_index] = source[j];
	current_index++;
      }
    }
  }

 public:

  static double SquaredCorrelationCoefficient(const Vector &observations,
					      const Vector &predictions) {
    
    // Compute the average of the observed values.
    double avg_observed_value = 0;
    
    for(index_t i = 0; i < observations.length(); i++) {
      avg_observed_value += observations[i];
    }
    avg_observed_value /= ((double) observations.length());

    // Compute something proportional to the variance of the observed
    // values, and the sum of squared residuals of the predictions
    // against the observations.
    double variance = 0;
    double residual = 0;
    for(index_t i = 0; i < observations.length(); i++) {
      variance += math::Sqr(observations[i] - avg_observed_value);
      residual += math::Sqr(observations[i] - predictions[i]);
    }
    return (variance - residual) / variance;
  }

  static double VarianceInflationFactor(const Vector &observations,
					const Vector &predictions) {
    
    return 1.0 / 
      (1.0 - SquaredCorrelationCoefficient(observations, predictions));
  }

  static void ComputeRoughCovariance_
  (const Matrix &input_data, const GenVector<index_t> &predictor_indices, 
   Matrix &rough_covariance) {
    
    // Initialize the covariance matrix to the zero matrix.
    rough_covariance.SetZero();

    // Loop over each column point.
    for(index_t i = 0; i < input_data.n_cols(); i++) {

      // The pointer to the point in consideration.
      const double *point = input_data.GetColumnPtr(i);

      // Loop over each predictor index.
      for(index_t j = -1; j < predictor_indices.length(); j++) {

	// The current predictor index.
	index_t outer_predictor_index = -1;
	double outer_value = 1.0;
	
	if(j >= 0) {
	  outer_predictor_index = predictor_indices[j];
	  outer_value = point[outer_predictor_index];
	}

	for(index_t k = -1; k < predictor_indices.length(); k++) {

	  // The inner predictor index.
	  index_t inner_predictor_index = -1;
	  double inner_value = 1.0;

	  if(k >= 0) {
	    inner_predictor_index = predictor_indices[k];
	    inner_value = point[inner_predictor_index];
	  }

	  rough_covariance.set
	    (outer_predictor_index + 1, inner_predictor_index + 1,
	     rough_covariance.get(outer_predictor_index + 1,
				  inner_predictor_index + 1) +
	     outer_value * inner_value);
	}
      }
    }
  }
  
  static void ExtractCovarianceMatrixSubset_
  (const Matrix &precomputed_covariance,
   const GenVector<index_t> &loo_current_predictor_indices,
   Matrix &precomputed_covariance_subset) {
    
    for(index_t i = -1; i < loo_current_predictor_indices.length(); i++) {
      index_t column_position = (i == -1) ? 
	0:loo_current_predictor_indices[i] + 1;

      for(index_t j = -1; j < loo_current_predictor_indices.length(); j++) {
	index_t row_position = (j == -1) ? 
	  0:loo_current_predictor_indices[j] + 1;

	precomputed_covariance_subset.set
	  (j + 1, i + 1, 
	   precomputed_covariance.get(row_position, column_position));
      }
    }    
  }

  /** @brief Performs the feature selection using the variance
   *         inflation factor. The invariant assumed in this function
   *         is that predictor_indices includes the
   *         prune_predictor_indices.
   *
   *  @param input_data The column-oriented dataset.
   *
   *  
   */
  static void FeatureSelection
  (fx_module *module, const Matrix &input_data,
   const GenVector<index_t> &predictor_indices, 
   const GenVector<index_t> &prune_predictor_indices, 
   GenVector<index_t> *output_predictor_indices) {

    NOTIFY("Starting VIF-based feature selection.");

    double lambda = fx_param_double(module, "lambda", 0.0);
    double variance_inflation_factor_threshold = 
      fx_param_double(module, "vif_threshold", 8.0);
    bool done_flag = false;
    GenVector<index_t> *current_predictor_indices = new GenVector<index_t>();
    GenVector<index_t> *current_prune_predictor_indices = new 
      GenVector<index_t>();
    const char *method = fx_param_str(module, "inversion_method", "quicsvd");
    current_predictor_indices->Copy(predictor_indices);
    current_prune_predictor_indices->Copy(prune_predictor_indices);

    // Precompute the covariance matrix to be used in the SVD if the
    // covariance-based method is chosen.
    Matrix *precomputed_covariance = NULL;
    if(!strcmp(method, "normalsvd")) {
      precomputed_covariance = new Matrix();
      precomputed_covariance->Init(input_data.n_rows() + 1,
				   input_data.n_rows() + 1);
      ComputeRoughCovariance_(input_data, predictor_indices, 
			      *precomputed_covariance);      
    }

    do {

      // The maximum variance inflation factor and the index that
      // achieved it.
      double max_variance_inflation_factor = 0.0;
      index_t index_of_max_variance_inflation_factor = -1;

      // Reset the flag to be true.
      done_flag = true;

      Matrix precomputed_covariance_subset;
      precomputed_covariance_subset.Init(current_predictor_indices->length(),
					 current_predictor_indices->length());

      // For each of the features in the current list, regress the
      // i-th feature versus the rest of the features and compute its
      // variance inflation factor.
      for(index_t i = 0; i < current_prune_predictor_indices->length(); i++) {
	
	// Take out the current dimension being regressed against from
	// the predictor list to form the leave-one-out predictor
	// list.
	GenVector<index_t> loo_current_predictor_indices;
	CopyVectorExceptOneIndex_(*current_predictor_indices, 
				  (*current_prune_predictor_indices)[i],
				  &loo_current_predictor_indices);

	// Initialize the ridge regression model using the
	// leave-one-out predictor indices with the appropriate
	// prediction index.
	RidgeRegression ridge_regression;
	ridge_regression.Init(module, input_data, 
			      loo_current_predictor_indices, 
			      (*current_prune_predictor_indices)[i]);

	// Do the regression.
	if(!strcmp(method, "normalsvd")) {

	  // Extract the covariance matrix and feed it to the SVD so
	  // that it is not computed again.
	  ExtractCovarianceMatrixSubset_(*precomputed_covariance,
					 loo_current_predictor_indices,
					 precomputed_covariance_subset);
	  
	  ridge_regression.SVDNormalEquationRegress
	    (lambda, &precomputed_covariance_subset);
	}
	else if(!strcmp(method, "quicsvd")) {
	  ridge_regression.QuicSVDRegress(lambda, 0.1);
	}
	else {
	  ridge_regression.SVDRegress(lambda);
	}
	Vector loo_predictions;
	ridge_regression.Predict(input_data, loo_current_predictor_indices, 
				 &loo_predictions);

	// Extract the dimension that is being regressed against and
	// compute the variance inflation factor.
	Vector loo_feature;
	loo_feature.Init(input_data.n_cols());
	for(index_t j = 0; j < input_data.n_cols(); j++) {
	  loo_feature[j] = input_data.get
	    ((*current_prune_predictor_indices)[i], j);
	}
	double variance_inflation_factor = 
	  VarianceInflationFactor(loo_feature, loo_predictions);

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
	CopyVectorExceptOneIndex_(*current_predictor_indices, 
				  index_of_max_variance_inflation_factor,
				  new_predictor_indices);
	delete current_predictor_indices;
	current_predictor_indices = new_predictor_indices;

	GenVector<index_t> *new_prune_indices = new GenVector<index_t>();
	CopyVectorExceptOneIndex_(*current_prune_predictor_indices, 
				  index_of_max_variance_inflation_factor,
				  new_prune_indices);
	delete current_prune_predictor_indices;
	current_prune_predictor_indices = new_prune_indices;
	done_flag = false;
      }

    } while(!done_flag && current_prune_predictor_indices->length() > 1);

    if(precomputed_covariance) {
      delete precomputed_covariance;
    }

    // Copy the output indices and free the temporary index vectors
    // afterwards.
    output_predictor_indices->Copy(*current_predictor_indices);
    delete current_predictor_indices;
    delete current_prune_predictor_indices;
    
    NOTIFY("VIF feature selection complete.");
  }

};

#endif
