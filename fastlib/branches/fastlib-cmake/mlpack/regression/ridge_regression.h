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
#ifndef RIDGE_REGRESSION_H_
#define RIDGE_REGRESSION_H_

#include "fastlib/fastlib.h"
#include "mlpack/quicsvd/quicsvd.h"
#include "ridge_regression_util.h"

class RidgeRegression {
 public:

  RidgeRegression() {
  }

  void Init(fx_module *module, const Matrix &predictors, 
	    const Matrix &predictions, 
	    bool use_normal_equation_method = true);

  void Init(fx_module *module, const Matrix &input_data, index_t selector,
	    bool use_normal_equation_method = true);

  /** @brief From a column-oriented dataset, initialize the design
   *         matrix using the row features whose indices belong to the
   *         given index set, and the predicted values from the given
   *         prediction index.
   *
   *  @param module The module from which the parameters are passed
   *  from.
   *
   *  @param input_data The entire dataset containing the features and
   *  the prediction values.
   *
   *  @param predictor_indices The row indices of the input data that
   *  should be part of the design matrix.
   *
   *  @param prediction_index The row index of the input data that
   *  should be used as the predictions (training target).
   */
  void Init(fx_module *module, 
            const Matrix &input_data, 
            const GenVector<index_t> &predictor_indices,
            index_t &prediction_index, bool use_normal_equation_method = true);

  void Init(fx_module *module, 
            const Matrix &input_data, 
            const GenVector<index_t> &predictor_indices,
            const Matrix &prediction, bool use_normal_equation_method = true);

  void ReInitTargetValues(const Matrix &input_data, 
			  index_t target_value_index);

  void ReInitTargetValues(const Matrix &target_values_in);

  void Destruct();

  void SVDRegress(double lambda,
		  const GenVector<index_t> *predictor_indices = NULL);

  void CrossValidatedRegression(double lambda_min, double lambda_max,
				index_t num);

  void FeatureSelectedRegression
  (const GenVector<index_t> &predictor_indices, 
   const GenVector<index_t> &prune_predictor_indices, 
   const Matrix &original_target_training_values,
   GenVector<index_t> *output_predictor_indices);

  double ComputeSquareError();

  /** @brief Predict on another dataset given the trained linear
   *         model.
   */
  void Predict(const Matrix &dataset, Vector *new_predictions);

  void Predict(const Matrix &dataset, 
	       const GenVector<index_t> &predictor_indices,
	       Vector *new_predictions);

  void factors(Matrix *factors);

 private:

  fx_module *module_;

  /** @brief The design matrix.
   */
  Matrix predictors_;

  /** @brief The training target values for each instance. This is
   *         generalizable to multi-target case by using a matrix.
   */
  Matrix predictions_;

  /** @brief The covariance matrix: roughly A A^T, modulo the
   *         mandatory 1 vector added to the features for the
   *         intercept.
   */
  Matrix covariance_;

  /** @brief The trained linear regression model output.
   */
  Matrix factors_;

  void ComputeLinearModel_(double lambda_sq, const Vector &singular_values, 
			   const Matrix &u, const Matrix &v_t,
			   int num_features);
  
  void BuildDesignMatrixFromIndexSet_
  (const Matrix &input_data, const double *predictions,
   const GenVector<index_t> *predictor_indices);
  
  void BuildCovariance_(const Matrix &input_data, 
			const GenVector<index_t> *predictor_indices,
			const double *predictions_in);

  void ExtractCovarianceSubset_
  (const Matrix &precomputed_covariance,
   const GenVector<index_t> *loo_current_predictor_indices,
   Matrix *precomputed_covariance_subset);

  void ExtractSubspace_(Matrix *u, Vector *singular_values, Matrix *v_t,
			const GenVector<index_t> *predictor_indices);
};

#include "ridge_regression_impl.h"
#endif
