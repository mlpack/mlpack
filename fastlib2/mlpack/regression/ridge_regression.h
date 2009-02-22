#ifndef RIDGE_REGRESSION_H_
#define RIDGE_REGRESSION_H_
#include "fastlib/fastlib.h"
#include "mlpack/quicsvd/quicsvd.h"

class RidgeRegression {
 public:

  RidgeRegression() {
  }

  void Init(fx_module *module, Matrix &predictors, Matrix &predictions);

  void Init(fx_module *module, Matrix &input_data, index_t selector);

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
            Matrix &input_data, 
            GenVector<index_t> &predictor_indices,
            index_t prediction_index);

  void Init(fx_module *module, 
            Matrix &input_data, 
            GenVector<index_t> &predictor_indices,
            Matrix &prediction);

  void Destruct();

  void Regress(double lambda);

  void QuicSVDRegress(double lambda, double relative_error);

  void SVDRegress(double lambda);

  void CrossValidatedRegression(double lambda_min, double lambda_max,
				index_t num);

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

  Matrix predictors_;

  Matrix predictions_;

  Matrix factors_;

  void ComputeLinearModel_(double lambda_sq, const Vector &singular_values, 
			   const Matrix &u, const Matrix v_t);

  void BuildDesignMatrixFromIndexSet_
  (const Matrix &input_data, const GenVector<index_t> &predictor_indices);

};

#include "ridge_regression_impl.h"
#endif
