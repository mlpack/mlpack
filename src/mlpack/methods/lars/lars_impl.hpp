/**
 * @file methods/lars/lars_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated LARS functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LARS_LARS_IMPL_HPP
#define MLPACK_METHODS_LARS_LARS_IMPL_HPP

//! In case it hasn't been included yet.
#include "lars.hpp"

namespace mlpack {

template<typename ModelMatType>
inline LARS<ModelMatType>::LARS(
    const bool useCholesky,
    const typename LARS<ModelMatType>::ElemType lambda1,
    const typename LARS<ModelMatType>::ElemType lambda2,
    const typename LARS<ModelMatType>::ElemType tolerance,
    const bool fitIntercept,
    const bool normalizeData) :
    matGram(&matGramInternal),
    useCholesky(useCholesky),
    lasso((lambda1 != 0)),
    lambda1(lambda1),
    elasticNet((lambda1 != 0) && (lambda2 != 0)),
    lambda2(lambda2),
    tolerance(tolerance),
    fitIntercept(fitIntercept),
    normalizeData(normalizeData),
    selectedLambda1(lambda1),
    selectedIndex(0),
    selectedIntercept(0.0),
    offsetY(0.0)
{ /* Nothing left to do. */ }

template<typename ModelMatType>
inline LARS<ModelMatType>::LARS(
    const bool useCholesky,
    const arma::mat& gramMatrix,
    const double lambda1,
    const double lambda2,
    const double tolerance,
    const bool fitIntercept,
    const bool normalizeData) :
    matGram(&gramMatrix),
    useCholesky(useCholesky),
    lasso((lambda1 != 0)),
    lambda1(lambda1),
    elasticNet((lambda1 != 0) && (lambda2 != 0)),
    lambda2(lambda2),
    tolerance(tolerance),
    fitIntercept(fitIntercept),
    normalizeData(normalizeData),
    selectedLambda1(lambda1),
    selectedIndex(0),
    selectedIntercept(0.0),
    offsetY(0.0)
{ /* Nothing left to do */ }

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename>
inline LARS<ModelMatType>::LARS(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::ElemType lambda1,
    const typename LARS<ModelMatType>::ElemType lambda2,
    const typename LARS<ModelMatType>::ElemType tolerance,
    const bool fitIntercept,
    const bool normalizeData) :
    LARS(useCholesky, lambda1, lambda2, tolerance, fitIntercept, normalizeData)
{
  Train(data, responses, colMajor);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename>
inline LARS<ModelMatType>::LARS(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::DenseMatType& gramMatrix,
    const typename LARS<ModelMatType>::ElemType lambda1,
    const typename LARS<ModelMatType>::ElemType lambda2,
    const typename LARS<ModelMatType>::ElemType tolerance,
    const bool fitIntercept,
    const bool normalizeData) :
    matGram(&gramMatrix),
    useCholesky(useCholesky),
    lasso((lambda1 != 0)),
    lambda1(lambda1),
    elasticNet((lambda1 != 0) && (lambda2 != 0)),
    lambda2(lambda2),
    tolerance(tolerance),
    fitIntercept(fitIntercept),
    normalizeData(normalizeData),
    selectedLambda1(lambda1),
    selectedIndex(0),
    selectedIntercept(0.0),
    offsetY(0.0)
{
  Train(data, responses, colMajor);
}

// Copy Constructor.
template<typename ModelMatType>
inline LARS<ModelMatType>::LARS(const LARS<ModelMatType>& other) :
    matGramInternal(other.matGramInternal),
    matGram(other.matGram != &other.matGramInternal ?
        other.matGram : &matGramInternal),
    matUtriCholFactor(other.matUtriCholFactor),
    useCholesky(other.useCholesky),
    lasso(other.lasso),
    lambda1(other.lambda1),
    elasticNet(other.elasticNet),
    lambda2(other.lambda2),
    tolerance(other.tolerance),
    fitIntercept(other.fitIntercept),
    normalizeData(other.normalizeData),
    betaPath(other.betaPath),
    lambdaPath(other.lambdaPath),
    interceptPath(other.interceptPath),
    activeSet(other.activeSet),
    selectedLambda1(other.selectedLambda1),
    selectedIndex(other.selectedIndex),
    selectedBeta(other.selectedBeta),
    selectedIntercept(other.selectedIntercept),
    selectedActiveSet(other.selectedActiveSet),
    offsetY(other.offsetY),
    isActive(other.isActive),
    ignoreSet(other.ignoreSet),
    isIgnored(other.isIgnored)
{
  // Nothing to do here.
}

// Move constructor.
template<typename ModelMatType>
inline LARS<ModelMatType>::LARS(LARS<ModelMatType>&& other) :
    matGramInternal(std::move(other.matGramInternal)),
    matGram(other.matGram != &other.matGramInternal ?
        other.matGram : &matGramInternal),
    matUtriCholFactor(std::move(other.matUtriCholFactor)),
    useCholesky(other.useCholesky),
    lasso(other.lasso),
    lambda1(other.lambda1),
    elasticNet(other.elasticNet),
    lambda2(other.lambda2),
    tolerance(other.tolerance),
    fitIntercept(other.fitIntercept),
    normalizeData(other.normalizeData),
    betaPath(std::move(other.betaPath)),
    lambdaPath(std::move(other.lambdaPath)),
    interceptPath(std::move(other.interceptPath)),
    activeSet(std::move(other.activeSet)),
    selectedLambda1(std::move(other.selectedLambda1)),
    selectedIndex(std::move(other.selectedIndex)),
    selectedBeta(std::move(other.selectedBeta)),
    selectedIntercept(std::move(other.selectedIntercept)),
    selectedActiveSet(std::move(other.selectedActiveSet)),
    offsetY(std::move(other.offsetY)),
    isActive(std::move(other.isActive)),
    ignoreSet(std::move(other.ignoreSet)),
    isIgnored(std::move(other.isIgnored))
{
  // Nothing to do here.
}

// Copy operator.
template<typename ModelMatType>
inline LARS<ModelMatType>& LARS<ModelMatType>::operator=(
    const LARS<ModelMatType>& other)
{
  if (&other == this)
    return *this;

  matGramInternal = other.matGramInternal;
  matGram = other.matGram != &other.matGramInternal ?
      other.matGram : &matGramInternal;
  matUtriCholFactor = other.matUtriCholFactor;
  useCholesky = other.useCholesky;
  lasso = other.lasso;
  lambda1 = other.lambda1;
  elasticNet = other.elasticNet;
  lambda2 = other.lambda2;
  tolerance = other.tolerance;
  fitIntercept = other.fitIntercept;
  normalizeData = other.normalizeData;
  betaPath = other.betaPath;
  lambdaPath = other.lambdaPath;
  interceptPath = other.interceptPath;
  activeSet = other.activeSet;
  selectedLambda1 = other.selectedLambda1;
  selectedIndex = other.selectedIndex;
  selectedBeta = other.selectedBeta;
  selectedIntercept = other.selectedIntercept;
  selectedActiveSet = other.selectedActiveSet;
  offsetY = other.offsetY;
  isActive = other.isActive;
  ignoreSet = other.ignoreSet;
  isIgnored = other.isIgnored;
  return *this;
}

// Move Operator.
template<typename ModelMatType>
inline LARS<ModelMatType>& LARS<ModelMatType>::operator=(
    LARS<ModelMatType>&& other)
{
  if (&other == this)
    return *this;

  matGramInternal = std::move(other.matGramInternal);
  matGram = other.matGram != &other.matGramInternal ?
      other.matGram : &matGramInternal;
  matUtriCholFactor = std::move(other.matUtriCholFactor);
  useCholesky = other.useCholesky;
  lasso = other.lasso;
  lambda1 = other.lambda1;
  elasticNet = other.elasticNet;
  lambda2 = other.lambda2;
  tolerance = other.tolerance;
  fitIntercept = other.fitIntercept;
  normalizeData = other.normalizeData;
  betaPath = std::move(other.betaPath);
  lambdaPath = std::move(other.lambdaPath);
  interceptPath = std::move(other.interceptPath);
  selectedLambda1 = std::move(other.selectedLambda1);
  selectedIndex = std::move(other.selectedIndex);
  selectedBeta = std::move(other.selectedBeta);
  selectedIntercept = std::move(other.selectedIntercept);
  selectedActiveSet = std::move(other.selectedActiveSet);
  offsetY = std::move(other.offsetY);
  activeSet = std::move(other.activeSet);
  isActive = std::move(other.isActive);
  ignoreSet = std::move(other.ignoreSet);
  isIgnored = std::move(other.isIgnored);
  return *this;
}

template<typename ModelMatType>
inline double LARS<ModelMatType>::Train(const arma::mat& matX,
                                        const arma::rowvec& y,
                                        arma::vec& beta,
                                        const bool colMajor)
{
  const double result = Train(matX, y, colMajor);
  beta = betaPath.back();
  return result;
}

// Dummy overload for MetaInfoExtractor.
template<typename ModelMatType>
template<typename MatType>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& data,
                          const arma::rowvec& responses,
                          const bool colMajor)
{
  return Train(data, responses, colMajor, this->useCholesky, this->lambda1,
      this->lambda2, this->tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& data,
                          const ResponsesType& responses,
                          const bool colMajor)
{
  return Train(data, responses, colMajor, this->useCholesky, this->lambda1,
      this->lambda2, this->tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& data,
                          const ResponsesType& responses,
                          const bool colMajor,
                          const bool useCholesky)
{
  return Train(data, responses, colMajor, useCholesky, this->lambda1,
      this->lambda2, this->tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& data,
                          const ResponsesType& responses,
                          const bool colMajor,
                          const bool useCholesky,
                          const typename LARS<ModelMatType>::ElemType lambda1)
{
  return Train(data, responses, colMajor, useCholesky, lambda1,
      this->lambda2, this->tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& data,
                          const ResponsesType& responses,
                          const bool colMajor,
                          const bool useCholesky,
                          const typename LARS<ModelMatType>::ElemType lambda1,
                          const typename LARS<ModelMatType>::ElemType lambda2)
{
  return Train(data, responses, colMajor, useCholesky, lambda1, lambda2,
      this->tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& data,
                          const ResponsesType& responses,
                          const bool colMajor,
                          const bool useCholesky,
                          const typename LARS<ModelMatType>::ElemType lambda1,
                          const typename LARS<ModelMatType>::ElemType lambda2,
                          const typename LARS<ModelMatType>::ElemType tolerance)
{
  return Train(data, responses, colMajor, useCholesky, lambda1, lambda2,
      tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& data,
                          const ResponsesType& responses,
                          const bool colMajor,
                          const bool useCholesky,
                          const typename LARS<ModelMatType>::ElemType lambda1,
                          const typename LARS<ModelMatType>::ElemType lambda2,
                          const typename LARS<ModelMatType>::ElemType tolerance,
                          const bool fitIntercept)
{
  return Train(data, responses, colMajor, useCholesky, lambda1, lambda2,
      tolerance, fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::DenseMatType& gramMatrix)
{
  return Train(data, responses, colMajor, useCholesky, gramMatrix,
      this->lambda1, this->lambda2, this->tolerance, this->fitIntercept,
      this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::DenseMatType& gramMatrix,
    const typename LARS<ModelMatType>::ElemType lambda1)
{
  return Train(data, responses, colMajor, useCholesky, gramMatrix, lambda1,
      this->lambda2, this->tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::DenseMatType& gramMatrix,
    const typename LARS<ModelMatType>::ElemType lambda1,
    const typename LARS<ModelMatType>::ElemType lambda2)
{
  return Train(data, responses, colMajor, useCholesky, gramMatrix, lambda1,
      lambda2, this->tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::DenseMatType& gramMatrix,
    const typename LARS<ModelMatType>::ElemType lambda1,
    const typename LARS<ModelMatType>::ElemType lambda2,
    const typename LARS<ModelMatType>::ElemType tolerance)
{
  return Train(data, responses, colMajor, useCholesky, gramMatrix, lambda1,
      lambda2, tolerance, this->fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::DenseMatType& gramMatrix,
    const typename LARS<ModelMatType>::ElemType lambda1,
    const typename LARS<ModelMatType>::ElemType lambda2,
    const typename LARS<ModelMatType>::ElemType tolerance,
    const bool fitIntercept)
{
  return Train(data, responses, colMajor, useCholesky, gramMatrix, lambda1,
      lambda2, tolerance, fitIntercept, this->normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const bool colMajor,
    const bool useCholesky,
    const typename LARS<ModelMatType>::DenseMatType& gramMatrix,
    const typename LARS<ModelMatType>::ElemType lambda1,
    const typename LARS<ModelMatType>::ElemType lambda2,
    const typename LARS<ModelMatType>::ElemType tolerance,
    const bool fitIntercept,
    const bool normalizeData)
{
  // Set Gram matrix.
  matGramInternal.clear();
  matGram = &gramMatrix;

  return Train(data, responses, colMajor, useCholesky, lambda1, lambda2,
      tolerance, fitIntercept, normalizeData);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Train(const MatType& matX,
                          const ResponsesType& y,
                          const bool colMajor,
                          const bool useCholesky,
                          const typename LARS<ModelMatType>::ElemType lambda1,
                          const typename LARS<ModelMatType>::ElemType lambda2,
                          const typename LARS<ModelMatType>::ElemType tolerance,
                          const bool fitIntercept,
                          const bool normalizeData)
{
  // Update hyperparameter settings.
  this->useCholesky = useCholesky;
  this->lambda1 = lambda1;
  this->lambda2 = lambda2;
  this->tolerance = tolerance;
  this->fitIntercept = fitIntercept;
  this->normalizeData = normalizeData;

  // Clear any previous solution information.
  betaPath.clear();
  lambdaPath.clear();
  activeSet.clear();
  isActive.clear();
  ignoreSet.clear();
  isIgnored.clear();
  matUtriCholFactor.reset();
  selectedBeta.clear();

  // Update values in case lambda1 or lambda2 changed.
  lasso = (lambda1 != 0);
  elasticNet = (lambda1 != 0 && lambda2 != 0);

  // This matrix may end up holding the transpose -- if necessary.
  MatType dataTrans;
  // This vector may hold zero-centered responses, if necessary.
  ResponsesType yCentered;

  // dataRef is row-major.  We can reuse the given matX, but only if we don't
  // need to do any transformations to it.
  const MatType& dataRef =
      (colMajor || fitIntercept || normalizeData) ? dataTrans : matX;
  const ResponsesType& yRef =
      (fitIntercept) ? yCentered : y;

  arma::Col<ElemType> offsetX; // used only if fitting an intercept
  this->offsetY = 0.0; // used only if fitting an intercept
  arma::Col<ElemType> stdX; // used only if normalizing

  if (colMajor)
  {
    if (fitIntercept)
    {
      offsetX = arma::mean(matX, 1);
      dataTrans = (matX.each_col() - offsetX).t();
    }

    if (normalizeData)
    {
      stdX = arma::stddev(matX, 0, 1);
      stdX.replace(0.0, 1.0); // Make sure we don't divide by 0!

      // Check if we have already done the transposition.
      if (!fitIntercept)
        dataTrans = (matX.each_col() / stdX).t();
      else
        dataTrans.each_row() /= stdX.t();
    }

    // Make sure we convert the data to row-major format, even if no
    // transformations were needed.
    if (!fitIntercept && !normalizeData)
      dataTrans = matX.t();
  }
  else
  {
    // We don't need to transpose the data---it's already in row-major form.
    if (fitIntercept)
    {
      offsetX = arma::mean(matX, 0).t();
      dataTrans = (matX.each_row() - offsetX.t());
    }

    if (normalizeData)
    {
      stdX = arma::stddev(matX, 0, 0).t();
      stdX.replace(0.0, 1.0); // Make sure we don't divide by 0!

      // Check if we have already populated `dataTrans`.
      if (!fitIntercept)
        dataTrans = (matX.each_row() / stdX.t());
      else
        dataTrans.each_row() /= stdX.t();
    }

    // If we are not fitting an intercept and we are not normalizing the data,
    // dataTrans already points to matX so we don't need to do anything.
  }

  if (fitIntercept)
  {
    this->offsetY = arma::mean(y);
    yCentered = y - this->offsetY;
  }

  // Compute X' * y.
  arma::Col<ElemType> vecXTy = trans(yRef * dataRef);

  // Set up active set variables.  In the beginning, the active set has size 0
  // (all dimensions are inactive).
  isActive.resize(dataRef.n_cols, false);

  // Set up ignores set variables. Initialized empty.
  isIgnored.resize(dataRef.n_cols, false);

  // Initialize yHat and beta.
  arma::Col<ElemType> beta(dataRef.n_cols);
  arma::Col<ElemType> yHat(dataRef.n_rows);
  arma::Col<ElemType> yHatDirection(dataRef.n_rows, arma::fill::none);

  bool lassocond = false;

  // Compute the initial maximum correlation among all dimensions.
  arma::Col<ElemType> corr = vecXTy;
  ElemType maxCorr = 0;
  size_t changeInd = 0;
  size_t lassocondInd = dataRef.n_cols;
  for (size_t i = 0; i < vecXTy.n_elem; ++i)
  {
    if (fabs(corr(i)) > maxCorr)
    {
      maxCorr = fabs(corr(i));
      changeInd = i;
    }
  }

  betaPath.push_back(beta);
  lambdaPath.push_back(maxCorr);

  // If the maximum correlation is too small, there is no reason to continue.
  if (maxCorr < lambda1)
  {
    lambdaPath[0] = lambda1;

    if (fitIntercept)
      interceptPath.push_back(this->offsetY - dot(offsetX, betaPath[0]));
    else
      interceptPath.push_back(0.0);

    return maxCorr;
  }

  // Compute the Gram matrix.  If this is the elastic net problem, we will add
  // lambda2 * I_n to the matrix.
  if (matGram->n_elem != dataRef.n_cols * dataRef.n_cols)
  {
    // In this case, matGram should reference matGramInternal.
    matGramInternal = trans(dataRef) * dataRef;

    if (elasticNet && !useCholesky)
    {
      matGramInternal += lambda2 *
          arma::eye<ModelMatType>(dataRef.n_cols, dataRef.n_cols);
    }
  }

  // Main loop.
  while (((activeSet.size() + ignoreSet.size()) < dataRef.n_cols) &&
         (maxCorr > tolerance))
  {
    // Compute the maximum correlation among inactive dimensions.
    maxCorr = 0;
    ElemType maxActiveCorr = 0;
    ElemType minActiveCorr = DBL_MAX;
    for (size_t i = 0; i < dataRef.n_cols; ++i)
    {
      if ((!isActive[i]) && (!isIgnored[i]) && (fabs(corr(i)) > maxCorr))
      {
        maxCorr = fabs(corr(i));
        changeInd = i;
      }
      else if (isActive[i] && (matGram != &matGramInternal))
      {
        // Here we will do a sanity check: if the correlation of any dimension
        // is not the maximum correlation, then the user has probably passed a
        // Gram matrix whose properties do not match the value of fitIntercept
        // and normalizeData.
        if (fabs(corr(i)) > maxActiveCorr)
          maxActiveCorr = fabs(corr(i));
        if (fabs(corr(i)) < minActiveCorr)
          minActiveCorr = fabs(corr(i));
      }
    }

    // If the maximum correlation is sufficiently small, don't add this
    // variable; terminate early.
    if (maxCorr < tolerance)
      break;

    // Add the variable to the active set and update the Gram matrix as
    // necessary.
    if (!lassocond)
    {
      if (useCholesky)
      {
        // vec newGramCol = vec(activeSet.size());
        // for (size_t i = 0; i < activeSet.size(); ++i)
        // {
        //   newGramCol[i] = dot(matX.col(activeSet[i]), matX.col(changeInd));
        // }
        // This is equivalent to the above 5 lines.
        arma::Col<ElemType> newGramCol = matGram->elem(
            changeInd * dataRef.n_cols +
            ConvTo<arma::uvec>::From(activeSet));

        CholeskyInsert((*matGram)(changeInd, changeInd), newGramCol);
      }
      Activate(changeInd);
    }

    // Compute signs of correlations.
    arma::Col<ElemType> s(activeSet.size());
    for (size_t i = 0; i < activeSet.size(); ++i)
    {
      const size_t j = activeSet[i];
      s[i] = (ElemType) (corr(j) == 0.0 ? 0.0 : (corr(j) > 0) ? 1.0 : -1.0);
    }

    // Compute the "equiangular" direction in parameter space (betaDirection).
    // We use quotes because in the case of non-unit norm variables, this need
    // not be equiangular.
    arma::Col<ElemType> unnormalizedBetaDirection;
    ElemType normalization;
    arma::Col<ElemType> betaDirection;
    if (useCholesky)
    {
      // Check for singularity.
      const ElemType lastUtriElement = matUtriCholFactor(
          matUtriCholFactor.n_cols - 1, matUtriCholFactor.n_rows - 1);
      if (std::abs(lastUtriElement) > tolerance)
      {
        // Ok, no singularity.
        /**
         * Note that:
         * R^T R % S^T % S = (R % S)^T (R % S)
         * Now, for 1 the ones vector:
         * inv( (R % S)^T (R % S) ) 1
         *    = inv(R % S) inv((R % S)^T) 1
         *    = inv(R % S) Solve((R % S)^T, 1)
         *    = inv(R % S) Solve(R^T, s)
         *    = Solve(R % S, Solve(R^T, s)
         *    = s % Solve(R, Solve(R^T, s))
         */
        unnormalizedBetaDirection = solve(trimatu(matUtriCholFactor),
            solve(trimatl(trans(matUtriCholFactor)), s));

        normalization = 1.0 / std::sqrt(dot(s, unnormalizedBetaDirection));
        betaDirection = normalization * unnormalizedBetaDirection;
      }
      else
      {
        // Singularity, so remove variable from active set, add to ignores set,
        // and look for new variable to add.
        Log::Warn << "Encountered singularity when adding variable "
            << changeInd << " to active set; permanently removing."
            << std::endl;
        Deactivate(activeSet.size() - 1);
        Ignore(changeInd);
        CholeskyDelete(matUtriCholFactor.n_rows - 1);

        // Note that although we are now ignoring this variable, we may still
        // need to take a step with the previous beta direction towards the next
        // variable we will add.
        s = s.subvec(0, activeSet.size() - 1); // Drop last element.
        unnormalizedBetaDirection = solve(trimatu(matUtriCholFactor),
            solve(trimatl(trans(matUtriCholFactor)), s));

        normalization = 1.0 / std::sqrt(dot(s, unnormalizedBetaDirection));
        betaDirection = normalization * unnormalizedBetaDirection;
      }
    }
    else
    {
      MatType matGramActive(activeSet.size(), activeSet.size());
      for (size_t i = 0; i < activeSet.size(); ++i)
        for (size_t j = 0; j < activeSet.size(); ++j)
          matGramActive(i, j) = (*matGram)(activeSet[i], activeSet[j]);

      // Check for singularity.
      MatType matS = s * ones<MatType>(1, activeSet.size());
      const bool solvedOk = solve(unnormalizedBetaDirection,
          matGramActive % trans(matS) % matS,
          ones<MatType>(activeSet.size(), 1));
      if (solvedOk)
      {
        // Ok, no singularity.
        normalization = 1.0 / std::sqrt(sum(unnormalizedBetaDirection));
        betaDirection = normalization * unnormalizedBetaDirection % s;
      }
      else
      {
        // Singularity, so remove variable from active set, add to ignores set,
        // and look for new variable to add.
        Deactivate(activeSet.size() - 1);
        Ignore(changeInd);
        Log::Warn << "Encountered singularity when adding variable "
            << changeInd << " to active set; permanently removing."
            << std::endl;

        // Note that although we are now ignoring this variable, we may still
        // need to take a step with the previous beta direction towards the next
        // variable we will add.
        s = s.subvec(0, activeSet.size() - 1); // Drop last element.
        matGramActive = matGramActive.submat(0, 0, activeSet.size() - 1,
            activeSet.size() - 1);
        matS = s * ones<MatType>(1, activeSet.size());
        // This worked last iteration, so there can't be a singularity.
        solve(unnormalizedBetaDirection,
            matGramActive % trans(matS) % matS,
            ones<MatType>(activeSet.size(), 1));
        normalization = 1.0 / std::sqrt(sum(unnormalizedBetaDirection));
        betaDirection = normalization * unnormalizedBetaDirection % s;
      }
    }

    // compute "equiangular" direction in output space
    ComputeYHatDirection(dataRef, betaDirection, yHatDirection);

    ElemType gamma = maxCorr / normalization;

    // If not all variables are active.
    if ((activeSet.size() + ignoreSet.size()) < dataRef.n_cols)
    {
      // Compute correlations with direction.
      for (size_t ind = 0; ind < dataRef.n_cols; ind++)
      {
        if (isActive[ind] || isIgnored[ind])
          continue;

        const ElemType dirCorr = dot(dataRef.col(ind), yHatDirection);
        const ElemType val1 = (maxCorr - corr(ind)) / (normalization - dirCorr);
        const ElemType val2 = (maxCorr + corr(ind)) / (normalization + dirCorr);

        // If we kicked out a feature due to the LASSO modification last
        // iteration, then we do not allow relaxation of the step size to 0 for
        // that feature in this iteration.
        if (lassocond && (ind == lassocondInd))
        {
          if ((val1 > 0.0) && (val1 < gamma))
            gamma = val1;
          if ((val2 > 0.0) && (val2 < gamma))
            gamma = val2;
        }
        else
        {
          if ((val1 >= 0.0) && (val1 < gamma))
            gamma = val1;
          if ((val2 >= 0.0) && (val2 < gamma))
            gamma = val2;
        }
      }
    }

    // Bound gamma according to LASSO.
    if (lasso)
    {
      lassocond = false;
      lassocondInd = dataRef.n_cols;
      ElemType lassoboundOnGamma = DBL_MAX;
      size_t activeIndToKickOut = -1;

      for (size_t i = 0; i < activeSet.size(); ++i)
      {
        ElemType val = -beta(activeSet[i]) / betaDirection(i);
        if ((val > 0) && (val < lassoboundOnGamma))
        {
          lassoboundOnGamma = val;
          activeIndToKickOut = i;
        }
      }

      if (lassoboundOnGamma < gamma)
      {
        gamma = lassoboundOnGamma;
        lassocond = true;
        changeInd = activeIndToKickOut;
        lassocondInd = activeSet[changeInd];
      }
    }

    // Update the prediction.
    yHat += gamma * yHatDirection;

    // Update the estimator.
    for (size_t i = 0; i < activeSet.size(); ++i)
    {
      beta(activeSet[i]) += gamma * betaDirection(i);
    }

    // Sanity check to make sure the kicked out dimension is actually zero.
    if (lassocond)
    {
      if (beta(activeSet[changeInd]) != 0)
      {
        beta(activeSet[changeInd]) = 0;
      }
    }

    betaPath.push_back(beta);

    if (lassocond)
    {
      // Index is in position changeInd in activeSet.
      if (useCholesky)
        CholeskyDelete(changeInd);

      Deactivate(changeInd);
    }

    corr = vecXTy - trans(dataRef) * yHat;
    if (elasticNet)
      corr -= lambda2 * beta;

    ElemType curLambda = 0;
    for (size_t i = 0; i < activeSet.size(); ++i)
      curLambda += fabs(corr(activeSet[i]));

    curLambda /= ((double) activeSet.size());

    lambdaPath.push_back(curLambda);

    // Time to stop for LASSO?
    if (lasso)
    {
      if (curLambda <= lambda1)
      {
        InterpolateBeta();
        break;
      }
    }
  }

  // Perform un-scaling of learned beta, if needed, to account for
  // normalization.
  if (normalizeData)
  {
    for (size_t i = 0; i < betaPath.size(); ++i)
    {
      betaPath[i] /= stdX;
    }
  }

  // Set the intercept values.  This is needed (for paranoia reasons) even if an
  // intercept isn't fit, in case a user changes `fitIntercept` after training.
  // If an intercept wasn't fit, we set them all to zero.
  if (fitIntercept)
  {
    interceptPath.clear();
    for (size_t i = 0; i < betaPath.size(); ++i)
      interceptPath.push_back(this->offsetY - dot(offsetX, betaPath[i]));
  }
  else
  {
    interceptPath.clear();
    interceptPath.resize(betaPath.size(), 0.0);
  }

  // Make the model we use point to the last element in the path after
  // interpolation.
  selectedLambda1 = lambda1;
  selectedIndex = betaPath.size() - 1;

  return ComputeError(matX, y, colMajor);
}

template<typename ModelMatType>
template<typename VecType>
inline typename LARS<ModelMatType>::ElemType LARS<ModelMatType>::Predict(
    const VecType& point) const
{
  if (!fitIntercept)
    return dot(Beta(), point);
  else
    return dot(Beta(), point) + Intercept();
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType>
inline void LARS<ModelMatType>::Predict(const MatType& points,
                                        ResponsesType& predictions,
                                        const bool colMajor) const
{
  if (!colMajor && !fitIntercept)
    predictions = trans(points * Beta());
  else if (!colMajor)
    predictions = trans(points * Beta()) + Intercept();
  else if (fitIntercept)
    predictions = Beta().t() * points + Intercept();
  else
    predictions = Beta().t() * points;
}

template<typename ModelMatType>
inline void LARS<ModelMatType>::FitIntercept(const bool newFitIntercept)
{
  // If we are storing a Gram matrix internally, but now will be normalizing
  // data, then the Gram matrix we have computed is incorrect and needs to be
  // recomputed.
  if (fitIntercept != newFitIntercept)
  {
    if (matGram != &matGramInternal)
    {
      throw std::invalid_argument("LARS::FitIntercept(): cannot change value "
          "when an external Gram matrix was specified!");
    }

    fitIntercept = newFitIntercept;
    matGramInternal.clear();
  }
}

template<typename ModelMatType>
inline void LARS<ModelMatType>::NormalizeData(const bool newNormalizeData)
{
  // If we are storing a Gram matrix internally, but now will be normalizing
  // data, then the Gram matrix we have computed is incorrect and needs to be
  // recomputed.
  if (normalizeData != newNormalizeData)
  {
    if (matGram != &matGramInternal)
    {
      throw std::invalid_argument("LARS::NormalizeData(): cannot change value"
          " when an external Gram matrix was specified!");
    }

    normalizeData = newNormalizeData;
    matGramInternal.clear();
  }
}

template<typename ModelMatType>
inline const std::vector<size_t>& LARS<ModelMatType>::ActiveSet() const
{
  if (selectedIndex != (betaPath.size() - 1))
    return selectedActiveSet;
  else
    return activeSet;
}

template<typename ModelMatType>
inline const typename LARS<ModelMatType>::ModelColType&
LARS<ModelMatType>::Beta() const
{
  if (selectedIndex < betaPath.size())
    return betaPath[selectedIndex];
  else
    return selectedBeta;
}

template<typename ModelMatType>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::Intercept() const
{
  if (selectedIndex < betaPath.size())
    return interceptPath[selectedIndex];
  else
    return selectedIntercept;
}

template<typename ModelMatType>
inline void LARS<ModelMatType>::SelectBeta(
    const typename LARS<ModelMatType>::ElemType selLambda1)
{
  if (selLambda1 < lambda1)
  {
    std::ostringstream oss;
    oss << "LARS::SelectBeta(): given lambda1 value (" << selLambda1 << ") "
        << "cannot be less than model's Lambda1() value (" << lambda1
        << ")!";
    throw std::invalid_argument(oss.str());
  }
  else if (betaPath.size() == 0)
  {
    throw std::runtime_error("LARS::SelectBeta(): model must be trained "
        "before calling SelectBeta()!");
  }

  this->selectedLambda1 = selLambda1;
  selectedBeta.clear();

  // Find which lambda values we are interpolating between.  lambdaPath is in
  // reverse order (due to the fact that LARS is a stepwise algorithm), so the
  // largest lambdas come first.
  size_t i = 0;
  while (i < lambdaPath.size())
  {
    if (selLambda1 == lambdaPath[i])
    {
      // If it's an exact match, no interpolation is necessary, and we can
      // directly use the element from the path.
      selectedIndex = i;

      // However, we may need to compute the active set.
      if (i != lambdaPath.size() - 1)
      {
        selectedActiveSet = ConvTo<std::vector<size_t>>::From(
            arma::find(betaPath[i] != 0));
      }

      return;
    }
    else if (selLambda1 > lambdaPath[i])
    {
      // It's not an exact match, but lambdaPath[i] is the first lambda element
      // that is smaller than the desired lambda.
      break;
    }

    ++i;
  }

  // In the case where selLambda1 is larger than the largest lambda we have a
  // model for, we can interpolate between the zero vector and the first model.
  if (i == 0)
  {
    const ElemType interp = selLambda1 / lambdaPath[0];
    selectedIndex = betaPath.size();

    selectedLambda1 = interp * lambdaPath[0];
    selectedBeta = interp * betaPath[0];
    // Computing the intercept differs just a little bit from what's expected,
    // because we have to account for the offsetY term, which is not zero even
    // for a zero model.
    selectedIntercept = (1 - interp) * this->offsetY +
        interp * interceptPath[0];
  }
  else if (i == betaPath.size())
  {
    // It's possible that we fit the model perfectly with some lambda1 value
    // less than this->lambda1.  In that case, the interpolated solution is just
    // the last solution.
    selectedIndex = betaPath.size() - 1;
    return;
  }
  else
  {
    const ElemType interp = (lambdaPath[i - 1] - selLambda1) /
        (lambdaPath[i - 1] - lambdaPath[i]);
    selectedIndex = betaPath.size();

    selectedLambda1 = (1 - interp) * lambdaPath[i - 1] + interp * lambdaPath[i];
    selectedBeta = (1 - interp) * betaPath[i - 1] + interp * betaPath[i];
    selectedIntercept = (1 - interp) * interceptPath[i - 1] +
        interp * interceptPath[i];
  }

  // Compute the active set of variables.
  selectedActiveSet = ConvTo<std::vector<size_t>>::From(
      arma::find(selectedBeta != 0));
}

// Private functions.
template<typename ModelMatType>
inline void LARS<ModelMatType>::Deactivate(const size_t activeVarInd)
{
  isActive[activeSet[activeVarInd]] = false;
  activeSet.erase(activeSet.begin() + activeVarInd);
}

template<typename ModelMatType>
inline void LARS<ModelMatType>::Activate(const size_t varInd)
{
  isActive[varInd] = true;
  activeSet.push_back(varInd);
}

template<typename ModelMatType>
inline void LARS<ModelMatType>::Ignore(const size_t varInd)
{
  isIgnored[varInd] = true;
  ignoreSet.push_back(varInd);
}

template<typename ModelMatType>
template<typename MatType, typename VecType>
inline void LARS<ModelMatType>::ComputeYHatDirection(
    const MatType& matX,
    const VecType& betaDirection,
    VecType& yHatDirection)
{
  yHatDirection.fill(0);
  for (size_t i = 0; i < activeSet.size(); ++i)
    yHatDirection += betaDirection(i) * matX.col(activeSet[i]);
}

template<typename ModelMatType>
inline void LARS<ModelMatType>::InterpolateBeta()
{
  const size_t pathLength = betaPath.size();

  // interpolate beta and stop
  ElemType ultimateLambda = lambdaPath[pathLength - 1];
  ElemType penultimateLambda = lambdaPath[pathLength - 2];
  ElemType interp = (penultimateLambda - lambda1)
      / (penultimateLambda - ultimateLambda);

  betaPath[pathLength - 1] = (1 - interp) * (betaPath[pathLength - 2])
      + interp * betaPath[pathLength - 1];

  lambdaPath[pathLength - 1] = lambda1;
}

template<typename ModelMatType>
template<typename VecType, typename MatType>
inline void LARS<ModelMatType>::CholeskyInsert(const VecType& newX,
                                               const MatType& X)
{
  if (matUtriCholFactor.n_rows == 0)
  {
    matUtriCholFactor.set_size(1, 1);

    if (elasticNet)
      matUtriCholFactor(0, 0) = std::sqrt(dot(newX, newX) + lambda2);
    else
      matUtriCholFactor(0, 0) = norm(newX, 2);
  }
  else
  {
    VecType newGramCol = trans(X) * newX;
    CholeskyInsert(dot(newX, newX), newGramCol);
  }
}

template<typename ModelMatType>
template<typename VecType>
inline void LARS<ModelMatType>::CholeskyInsert(
    typename LARS<ModelMatType>::ElemType sqNormNewX,
    const VecType& newGramCol)
{
  int n = matUtriCholFactor.n_rows;

  if (n == 0)
  {
    matUtriCholFactor.set_size(1, 1);

    if (elasticNet)
      matUtriCholFactor(0, 0) = std::sqrt(sqNormNewX + lambda2);
    else
      matUtriCholFactor(0, 0) = std::sqrt(sqNormNewX);
  }
  else
  {
    DenseMatType matNewR(n + 1, n + 1);

    if (elasticNet)
      sqNormNewX += lambda2;

    arma::Col<ElemType> matUtriCholFactork =
        solve(trimatl(trans(matUtriCholFactor)), newGramCol);

    matNewR(arma::span(0, n - 1), arma::span(0, n - 1)) = matUtriCholFactor;
    matNewR(arma::span(0, n - 1), n) = matUtriCholFactork;
    matNewR(n, arma::span(0, n - 1)).fill(0.0);
    matNewR(n, n) = std::sqrt(sqNormNewX - dot(matUtriCholFactork,
                                               matUtriCholFactork));

    matUtriCholFactor = std::move(matNewR);
  }
}

template<typename ModelMatType>
template<typename MatType>
inline void LARS<ModelMatType>::GivensRotate(
    const typename arma::Col<
        typename LARS<ModelMatType>::ElemType
    >::template fixed<2>& x,
    typename arma::Col<
        typename LARS<ModelMatType>::ElemType
    >::template fixed<2>& rotatedX,
    MatType& matG)
{
  if (x(1) == 0)
  {
    matG = arma::eye<MatType>(2, 2);
    rotatedX = x;
  }
  else
  {
    ElemType r = norm(x, 2);
    matG.set_size(2, 2);

    ElemType scaledX1 = x(0) / r;
    ElemType scaledX2 = x(1) / r;

    matG(0, 0) = scaledX1;
    matG(1, 0) = -scaledX2;
    matG(0, 1) = scaledX2;
    matG(1, 1) = scaledX1;

    rotatedX(0) = r;
    rotatedX(1) = 0;
  }
}

template<typename ModelMatType>
inline void LARS<ModelMatType>::CholeskyDelete(const size_t colToKill)
{
  size_t n = matUtriCholFactor.n_rows;

  if (colToKill == (n - 1))
  {
    matUtriCholFactor = matUtriCholFactor(arma::span(0, n - 2),
                                          arma::span(0, n - 2));
  }
  else
  {
    matUtriCholFactor.shed_col(colToKill); // remove column colToKill
    n--;

    for (size_t k = colToKill; k < n; ++k)
    {
      DenseMatType matG;
      typename arma::Col<ElemType>::template fixed<2> rotatedVec;
      GivensRotate(matUtriCholFactor(arma::span(k, k + 1), k), rotatedVec,
          matG);
      matUtriCholFactor(arma::span(k, k + 1), k) = rotatedVec;
      if (k < n - 1)
      {
        matUtriCholFactor(arma::span(k, k + 1), arma::span(k + 1, n - 1)) =
            matG * matUtriCholFactor(arma::span(k, k + 1),
            arma::span(k + 1, n - 1));
      }
    }

    matUtriCholFactor.shed_row(n);
  }
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType>
inline typename LARS<ModelMatType>::ElemType
LARS<ModelMatType>::ComputeError(const MatType& matX,
                                 const ResponsesType& y,
                                 const bool colMajor)
{
  if (!colMajor)
    return accu(pow(y - trans(matX * Beta()) - Intercept(), 2.0));
  else
    return accu(pow(y - Beta().t() * matX - Intercept(), 2.0));
}

/**
 * Serialize the LARS model.
 */
template<typename ModelMatType>
template<typename Archive>
void LARS<ModelMatType>::serialize(Archive& ar, const uint32_t version)
{
  // If we're loading, we have to use the internal storage.
  if (cereal::is_loading<Archive>())
  {
    matGram = &matGramInternal;
    if (version == 0)
    {
      // Older versions stored matGramInternal as type arma::mat.
      arma::mat matGramInternalTmp;
      ar(cereal::make_nvp("matGramInternal", matGramInternalTmp));
      matGramInternal = ConvTo<ModelMatType>::From(matGramInternalTmp);
    }
    else
    {
      ar(CEREAL_NVP(matGramInternal));
    }
  }
  else
  {
    ar(cereal::make_nvp("matGramInternal",
        (const_cast<arma::mat&>(*matGram))));
  }

  if (cereal::is_loading<Archive>() && version == 0)
  {
    // Older versions stored matUtriCholFactor as type arma::mat, and other
    // elements as type double.  This version loads everything as
    // double/arma::mat and converts as needed.
    arma::mat matUtriCholFactorTmp;
    ar(cereal::make_nvp("matUtriCholFactor", matUtriCholFactorTmp));
    matUtriCholFactor = ConvTo<ModelMatType>::From(matUtriCholFactorTmp);

    ar(CEREAL_NVP(useCholesky));
    ar(CEREAL_NVP(lasso));

    double tmp;
    ar(cereal::make_nvp("lambda1", tmp));
    lambda1 = tmp;

    ar(CEREAL_NVP(elasticNet));

    ar(cereal::make_nvp("lambda2", tmp));
    lambda2 = tmp;

    ar(cereal::make_nvp("tolerance", tmp));
    tolerance = tmp;

    ar(CEREAL_NVP(fitIntercept));
    ar(CEREAL_NVP(normalizeData));

    std::vector<arma::vec> betaPathTmp;
    ar(cereal::make_nvp("betaPath", betaPathTmp));
    betaPath.resize(betaPathTmp.size());
    for (size_t i = 0; i < betaPathTmp.size(); ++i)
      betaPath[i] = ConvTo<ModelColType>::From(betaPathTmp[i]);

    std::vector<double> lambdaPathTmp;
    ar(cereal::make_nvp("lambdaPath", lambdaPathTmp));
    lambdaPath.resize(lambdaPathTmp.size());
    for (size_t i = 0; i < lambdaPathTmp.size(); ++i)
      lambdaPath[i] = (ElemType) lambdaPathTmp[i];

    std::vector<double> interceptPathTmp;
    ar(cereal::make_nvp("interceptPath", interceptPathTmp));
    interceptPath.resize(interceptPathTmp.size());
    for (size_t i = 0; i < interceptPathTmp.size(); ++i)
      interceptPath[i] = (ElemType) interceptPathTmp[i];

    ar(CEREAL_NVP(activeSet));
    ar(CEREAL_NVP(isActive));
    ar(CEREAL_NVP(ignoreSet));
    ar(CEREAL_NVP(isIgnored));
  }
  else
  {
    ar(CEREAL_NVP(matUtriCholFactor));
    ar(CEREAL_NVP(useCholesky));
    ar(CEREAL_NVP(lasso));
    ar(CEREAL_NVP(lambda1));
    ar(CEREAL_NVP(elasticNet));
    ar(CEREAL_NVP(lambda2));
    ar(CEREAL_NVP(tolerance));
    ar(CEREAL_NVP(fitIntercept));
    ar(CEREAL_NVP(normalizeData));
    ar(CEREAL_NVP(betaPath));
    ar(CEREAL_NVP(lambdaPath));
    ar(CEREAL_NVP(interceptPath));
    ar(CEREAL_NVP(activeSet));
    ar(CEREAL_NVP(isActive));
    ar(CEREAL_NVP(ignoreSet));
    ar(CEREAL_NVP(isIgnored));
  }

  if (version > 0)
  {
    ar(CEREAL_NVP(selectedLambda1));
    ar(CEREAL_NVP(selectedIndex));
    ar(CEREAL_NVP(selectedBeta));
    ar(CEREAL_NVP(selectedIntercept));
    ar(CEREAL_NVP(selectedActiveSet));
    ar(CEREAL_NVP(offsetY));
  }
  else if (cereal::is_loading<Archive>())
  {
    selectedLambda1 = lambdaPath.back();
    selectedIndex = betaPath.size() - 1;
    selectedBeta.clear();
    selectedIntercept = 0.0;
    selectedActiveSet.clear();
    offsetY = 0.0;
  }
}

} // namespace mlpack

#endif
