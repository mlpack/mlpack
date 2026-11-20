/**
 * @file methods/lars/lars.hpp
 * @author Nishant Mehta (niche)
 *
 * Definition of the LARS class, which performs Least Angle Regression and the
 * LASSO.
 *
 * Only minor modifications of LARS are necessary to handle the constrained
 * version of the problem:
 *
 * \f[
 * \min_{\beta} 0.5 || X \beta - y ||_2^2 + 0.5 \lambda_2 || \beta ||_2^2
 * \f]
 * subject to \f$ ||\beta||_1 <= \tau \f$
 *
 * Although this option currently is not implemented, it will be implemented
 * very soon.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LARS_LARS_HPP
#define MLPACK_METHODS_LARS_LARS_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * An implementation of LARS, a stage-wise homotopy-based algorithm for
 * l1-regularized linear regression (LASSO) and l1+l2 regularized linear
 * regression (Elastic Net).
 *
 * Let \f$ X \f$ be a matrix where each row is a point and each column is a
 * dimension and let \f$ y \f$ be a vector of responses.
 *
 * The Elastic Net problem is to solve
 *
 * \f[ \min_{\beta} 0.5 || X \beta - y ||_2^2 + \lambda_1 || \beta ||_1 +
 *     0.5 \lambda_2 || \beta ||_2^2 \f]
 *
 * where \f$ \beta \f$ is the vector of regression coefficients.
 *
 * If \f$ \lambda_1 > 0 \f$ and \f$ \lambda_2 = 0 \f$, the problem is the LASSO.
 * If \f$ \lambda_1 > 0 \f$ and \f$ \lambda_2 > 0 \f$, the problem is the
 *   elastic net.
 * If \f$ \lambda_1 = 0 \f$ and \f$ \lambda_2 > 0 \f$, the problem is ridge
 *   regression.
 * If \f$ \lambda_1 = 0 \f$ and \f$ \lambda_2 = 0 \f$, the problem is
 *   unregularized linear regression.
 *
 * Note: This algorithm is not recommended for use (in terms of efficiency)
 * when \f$ \lambda_1 \f$ = 0.
 *
 * For more details, see the following papers:
 *
 * @code
 * @article{efron2004least,
 *   title={Least angle regression},
 *   author={Efron, B. and Hastie, T. and Johnstone, I. and Tibshirani, R.},
 *   journal={The Annals of statistics},
 *   volume={32},
 *   number={2},
 *   pages={407--499},
 *   year={2004},
 *   publisher={Institute of Mathematical Statistics}
 * }
 * @endcode
 *
 * @code
 * @article{zou2005regularization,
 *   title={Regularization and variable selection via the elastic net},
 *   author={Zou, H. and Hastie, T.},
 *   journal={Journal of the Royal Statistical Society Series B},
 *   volume={67},
 *   number={2},
 *   pages={301--320},
 *   year={2005},
 *   publisher={Royal Statistical Society}
 * }
 * @endcode
 */
template<typename ModelMatType = arma::mat>
class LARS
{
 public:
  using ModelColType = typename GetColType<ModelMatType>::type;
  using DenseMatType = typename GetDenseMatType<ModelMatType>::type;
  using ElemType = typename ModelMatType::elem_type;

  /**
   * Set the parameters to LARS.  Both lambda1 and lambda2 default to 0.
   *
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *    solving linear system (as opposed to using the full Gram matrix).
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param tolerance Run until the maximum correlation of elements in (X^T y)
   *     is less than this.
   * @param intercept If true, fit an intercept in the model.
   * @param normalize If true, normalize all features to have unit variance for
   *     training.
   * @param fitIntercept If true, fit an intercept in the model.
   * @param normalizeData If true, normalize all features to have unit variance
   * for training.
   */
  LARS(const bool useCholesky = false,
       const ElemType lambda1 = 0.0,
       const ElemType lambda2 = 0.0,
       const ElemType tolerance = 1e-16,
       const bool fitIntercept = true,
       const bool normalizeData = true);

  /**
   * Set the parameters to LARS, and pass in a precalculated Gram matrix.  Both
   * lambda1 and lambda2 default to 0.
   *
   * Note that the precalculated Gram matrix must match the settings of
   * `fitIntercept` and `normalizeData` (which both default to `true`): so, this
   * means that by default, the Gram matrix should be computed on mean-centered
   * data whose features are normalized to have unit variance.
   *
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *    solving linear system (as opposed to using the full Gram matrix).
   * @param gramMatrix Gram matrix.
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param tolerance Run until the maximum correlation of elements in (X^T y)
   *     is less than this.
   * @param intercept If true, fit an intercept in the model.
   * @param normalize If true, normalize all features to have unit variance for
   *     training.
   * @param fitIntercept If true, fit an intercept in the model.
   * @param normalizeData If true, normalize all features to have unit variance
   * for training.
   */
  [[deprecated("Use other constructors")]]
  LARS(const bool useCholesky,
       const arma::mat& gramMatrix,
       const double lambda1 = 0.0,
       const double lambda2 = 0.0,
       const double tolerance = 1e-16,
       const bool fitIntercept = true,
       const bool normalizeData = true);

  /**
   * Set the parameters to LARS and run training. Both lambda1 and lambda2
   * are set by default to 0.
   *
   * @param data Input data.
   * @param responses A vector of targets.
   * @param colMajor Should be true if the input data is column-major.  Passing
   *     row-major data can avoid a transpose operation.
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *     solving linear system (as opposed to using the full Gram matrix).
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param tolerance Run until the maximum correlation of elements in (X^T y)
   *     is less than this.
   * @param fitIntercept If true, fit an intercept in the model.
   * @param normalizeData If true, normalize all features to have unit variance
   * for training.
   */
  template<typename MatType,
           typename ResponsesType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  LARS(const MatType& data,
       const ResponsesType& responses,
       bool colMajor = true,
       const bool useCholesky = false,
       const ElemType lambda1 = 0.0,
       const ElemType lambda2 = 0.0,
       const ElemType tolerance = 1e-16,
       const bool fitIntercept = true,
       const bool normalizeData = true);

  /**
   * Set the parameters to LARS, pass in a precalculated Gram matrix, and run
   * training. Both lambda1 and lambda2 are set by default to 0.
   *
   * Note that the precalculated Gram matrix must match the settings of
   * `fitIntercept` and `normalizeData` (which both default to `true`): so, this
   * means that by default, the Gram matrix should be computed on mean-centered
   * data whose features are normalized to have unit variance.
   *
   * @param data Input data.
   * @param responses A vector of targets.
   * @param colMajor Should be true if the input data is column-major.  Passing
   *     row-major data can avoid a transpose operation.
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *     solving linear system (as opposed to using the full Gram matrix).
   * @param gramMatrix Gram matrix.
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param tolerance Run until the maximum correlation of elements in (X^T y)
   *     is less than this.
   * @param fitIntercept If true, fit an intercept in the model.
   * @param normalizeData If true, normalize all features to have unit variance
   * for training.
   */
  template<typename MatType,
           typename ResponsesType,
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  LARS(const MatType& data,
       const ResponsesType& responses,
       const bool colMajor,
       const bool useCholesky,
       const DenseMatType& gramMatrix,
       const ElemType lambda1 = 0.0,
       const ElemType lambda2 = 0.0,
       const ElemType tolerance = 1e-16,
       const bool fitIntercept = true,
       const bool normalizeData = true);

  /**
   * Construct the LARS object by copying the given LARS object.
   *
   * @param other LARS object to copy.
   */
  LARS(const LARS& other);

  /**
   * Construct the LARS object by taking ownership of the given LARS object.
   *
   * @param other LARS object to take ownership of.
   */
  LARS(LARS&& other);

  /**
   * Copy the given LARS object.
   *
   * @param other LARS object to copy.
   */
  LARS& operator=(const LARS& other);

  /**
   * Take ownership of the given LARS object.
   *
   * @param other LARS object to take ownership of.
   */
  LARS& operator=(LARS&& other);

  /**
   * Run LARS.  The input matrix (like all mlpack matrices) should be
   * column-major -- each column is an observation and each row is a dimension.
   * However, because LARS is more efficient on a row-major matrix, this method
   * will (internally) transpose the matrix.  If this transposition is not
   * necessary (i.e., you want to pass in a row-major matrix), pass `false` for
   * the `colMajor` parameter.
   *
   * @param data Column-major input data (or row-major input data if colMajor =
   *     false).
   * @param responses A vector of targets.
   * @param beta Vector to store the solution (the coefficients) in.
   * @param colMajor Should be true if the input data is column-major.  Passing
   *     row-major data can avoid a transpose operation.
   * @return minimum cost error(||y-beta*X||2 is used to calculate error).
   */
  [[deprecated("Use other constructors")]]
  double Train(const arma::mat& data,
               const arma::rowvec& responses,
               arma::vec& beta,
               const bool colMajor = true);

  /**
   * Run LARS.  The input matrix (like all mlpack matrices) should be
   * column-major -- each column is an observation and each row is a dimension.
   * However, because LARS is more efficient on a row-major matrix, this method
   * will (internally) transpose the matrix.  If this transposition is not
   * necessary (i.e., you want to pass in a row-major matrix), pass `false` for
   * the `colMajor` parameter.
   *
   * All of the different overloads below are needed until C++17 is the minimum
   * required standard (then std::optional could be used).
   *
   * @param data Input data.
   * @param responses A vector of targets.
   * @param colMajor Should be true if the input data is column-major.  Passing
   *     row-major data can avoid a transpose operation.
   * @return minimum cost error(||y-beta*X||2 is used to calculate error).
   */

  // Dummy overload so MetaInfoExtractor can properly detect that LARS is a
  // regression method.
  template<typename MatType>
  ElemType Train(const MatType& data,
                 const arma::rowvec& responses,
                 const bool colMajor = true);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>,
           typename = std::enable_if_t<
               !std::is_same_v<ResponsesType, arma::rowvec>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor = true);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const ElemType lambda1);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const ElemType lambda1,
                 const ElemType lambda2);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const ElemType lambda1,
                 const ElemType lambda2,
                 const ElemType tolerance);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const ElemType lambda1,
                 const ElemType lambda2,
                 const ElemType tolerance,
                 const bool fitIntercept);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const ElemType lambda1,
                 const ElemType lambda2,
                 const ElemType tolerance,
                 const bool fitIntercept,
                 const bool normalizeData);

  /**
   * Run LARS with a precomputed Gram matrix.  The input matrix (like all mlpack
   * matrices) should be column-major -- each column is an observation and each
   * row is a dimension.  However, because LARS is more efficient on a row-major
   * matrix, this method will (internally) transpose the matrix.  If this
   * transposition is not necessary (i.e., you want to pass in a row-major
   * matrix), pass `false` for the `colMajor` parameter.
   *
   * All of the different overloads below are needed until C++17 is the minimum
   * required standard (then std::optional could be used).
   *
   * @param data Input data.
   * @param responses A vector of targets.
   * @param colMajor Should be true if the input data is column-major.  Passing
   *     row-major data can avoid a transpose operation.
   * @return minimum cost error(||y-beta*X||2 is used to calculate error).
   */
  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const DenseMatType& gramMatrix);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const DenseMatType& gramMatrix,
                 const ElemType lambda1);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const DenseMatType& gramMatrix,
                 const ElemType lambda1,
                 const ElemType lambda2);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const DenseMatType& gramMatrix,
                 const ElemType lambda1,
                 const ElemType lambda2,
                 const ElemType tolerance);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const DenseMatType& gramMatrix,
                 const ElemType lambda1,
                 const ElemType lambda2,
                 const ElemType tolerance,
                 const bool fitIntercept);

  template<typename MatType,
           typename ResponsesType,
           typename = void, /* so MetaInfoExtractor does not get confused */
           typename = std::enable_if_t<
               std::is_same_v<typename ResponsesType::elem_type, ElemType>>>
  ElemType Train(const MatType& data,
                 const ResponsesType& responses,
                 const bool colMajor,
                 const bool useCholesky,
                 const DenseMatType& gramMatrix,
                 const ElemType lambda1,
                 const ElemType lambda2,
                 const ElemType tolerance,
                 const bool fitIntercept,
                 const bool normalizeData);

  /**
   * Predict y_i for the given data point.
   *
   * @param point The data point to regress on.
   * @return Predicted value for y_i for `point`.
   */
  template<typename VecType>
  ElemType Predict(const VecType& point) const;

  /**
   * Predict y_i for each data point in the given data matrix using the
   * currently-trained LARS model.
   *
   * @param points The data points to regress on.
   * @param predictions y, which will contained calculated values on completion.
   * @param colMajor Should be true if the input data is column-major.  Passing
   *     row-major data can avoid a transpose operation.
   */
  template<typename MatType, typename ResponsesType>
  void Predict(const MatType& points,
               ResponsesType& predictions,
               const bool colMajor = true) const;

  //! Get the L1 regularization coefficient.
  ElemType Lambda1() const { return lambda1; }
  //! Modify the L1 regularization coefficient.
  ElemType& Lambda1() { return lambda1; }

  //! Get the L2 regularization coefficient.
  ElemType Lambda2() const { return lambda2; }
  //! Modify the L2 regularization coefficient.
  ElemType& Lambda2() { return lambda2; }

  //! Get whether to use the Cholesky decomposition.
  bool UseCholesky() const { return useCholesky; }
  //! Modify whether to use the Cholesky decomposition.
  bool& UseCholesky() { return useCholesky; }

  //! Get the tolerance for maximum correlation during training.
  ElemType Tolerance() const { return tolerance; }
  //! Modify the tolerance for maximum correlation during training.
  ElemType& Tolerance() { return tolerance; }

  //! Get whether or not to fit an intercept.
  bool FitIntercept() const { return fitIntercept; }
  //! Modify whether or not to fit an intercept.
  void FitIntercept(const bool newFitIntercept);

  //! Get whether or not to normalize data during training.
  bool NormalizeData() const { return normalizeData; }
  //! Modify whether or not to normalize data during training.
  void NormalizeData(const bool newNormalizeData);

  //! Access the set of active dimensions in the currently selected model.
  const std::vector<size_t>& ActiveSet() const;

  //! Access the set of coefficients after each iteration; the solution is the
  //! last element.
  const std::vector<ModelColType>& BetaPath() const { return betaPath; }

  //! Access the solution coefficients
  const ModelColType& Beta() const;

  //! Access the set of values for lambda1 after each iteration; the solution is
  //! the last element.
  const std::vector<ElemType>& LambdaPath() const { return lambdaPath; }

  //! Return the intercept (if fitted, otherwise 0).
  ElemType Intercept() const;

  //! Return the intercept path (the intercept for every model).
  const std::vector<ElemType>& InterceptPath() const { return interceptPath; }

  //! Set the model to use the given lambda1 value in the path.
  void SelectBeta(const ElemType lambda1);

  //! Get the L1 penalty parameter corresponding to the currently selected
  //! model.
  ElemType SelectedLambda1() const { return selectedLambda1; }

  //! Access the upper triangular cholesky factor.
  const DenseMatType& MatUtriCholFactor() const { return matUtriCholFactor; }

  /**
   * Serialize the LARS model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  /**
   * Compute cost error of the given data matrix using the
   * currently-trained LARS model. Only ||y-beta*X||2 is used to calculate
   * cost error.
   *
   * @param matX Column-major input data (or row-major input data if colMajor =
   *     false).
   * @param y responses A vector of targets.
   * @param colMajor Should be true if the data points matrix is column-major.
   * @return The minimum cost error.
   */
  template<typename MatType, typename ResponsesType>
  ElemType ComputeError(const MatType& matX,
                        const ResponsesType& y,
                        const bool colMajor = true);

 private:
  //! Gram matrix.
  DenseMatType matGramInternal;

  //! Pointer to the Gram matrix we will use.
  const DenseMatType* matGram;

  //! Upper triangular cholesky factor; initially 0x0 matrix.
  DenseMatType matUtriCholFactor;

  //! Whether or not to use Cholesky decomposition when solving linear system.
  bool useCholesky;

  //! True if this is the LASSO problem.
  bool lasso;
  //! Regularization parameter for l1 penalty.
  ElemType lambda1;

  //! True if this is the elastic net problem.
  bool elasticNet;
  //! Regularization parameter for l2 penalty.
  ElemType lambda2;

  //! Tolerance for main loop.
  ElemType tolerance;

  //! Whether or not to fit an intercept.
  bool fitIntercept;

  //! Whether or not to normalize data during training (i.e. make each feature
  //! have unit variance for training).
  bool normalizeData;

  //! Solution path.
  std::vector<ModelColType> betaPath;

  //! Value of lambda_1 for each solution in solution path.
  std::vector<ElemType> lambdaPath;

  //! Intercept (only if fitIntercept is true).
  std::vector<ElemType> interceptPath;

  //! Active set of dimensions.
  std::vector<size_t> activeSet;

  //! Selected lambda1 value for Predict().
  ElemType selectedLambda1;

  //! Index of selected beta (if selectedLambda1 is in lambdaPath).
  size_t selectedIndex;

  //! Selected beta, if selectedLambda1 is not in lambdaPath.
  ModelColType selectedBeta;

  //! Selected intercept, if selectedLambda1 is not in lambdaPath.
  ElemType selectedIntercept;

  //! Selected active set of dimensions, if selectedLambda1 is not the last
  //! element in the path.
  std::vector<size_t> selectedActiveSet;

  //! Might be needed to compute the intercept for other lambda values.
  ElemType offsetY;

  //! Active set membership indicator (for each dimension).
  std::vector<bool> isActive;

  // Set of variables that are ignored (if any).

  //! Set of ignored variables (for dimensions in span{active set dimensions}).
  std::vector<size_t> ignoreSet;

  //! Membership indicator for set of ignored variables.
  std::vector<bool> isIgnored;

  /**
   * Remove activeVarInd'th element from active set.
   *
   * @param activeVarInd Index of element to remove from active set.
   */
  void Deactivate(const size_t activeVarInd);

  /**
   * Add dimension varInd to active set.
   *
   * @param varInd Dimension to add to active set.
   */
  void Activate(const size_t varInd);

  /**
   * Add dimension varInd to ignores set (never removed).
   *
   * @param varInd Dimension to add to ignores set.
   */
  void Ignore(const size_t varInd);

  // Compute "equiangular" direction in output space.
  template<typename MatType, typename VecType>
  void ComputeYHatDirection(const MatType& matX,
                            const VecType& betaDirection,
                            VecType& yHatDirection);

  // Interpolate to compute last solution vector.
  void InterpolateBeta();

  template<typename VecType, typename MatType>
  void CholeskyInsert(const VecType& newX, const MatType& X);

  template<typename VecType>
  void CholeskyInsert(ElemType sqNormNewX, const VecType& newGramCol);

  template<typename MatType>
  void GivensRotate(const typename arma::Col<ElemType>::template fixed<2>& x,
                    typename arma::Col<ElemType>::template fixed<2>& rotatedX,
                    MatType& G);

  void CholeskyDelete(const size_t colToKill);
};

} // namespace mlpack

CEREAL_TEMPLATE_CLASS_VERSION((typename ModelMatType),
    (mlpack::LARS<ModelMatType>), (1));

// Include implementation of serialize().
#include "lars_impl.hpp"

#endif
