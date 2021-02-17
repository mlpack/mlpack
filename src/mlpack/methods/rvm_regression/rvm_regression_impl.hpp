/**
 * @file rvm_regression_impl.cpp
 * @author Clement Mercier
 *
 * Implementation of the Relevance Vector Machine.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RVM_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_RVM_REGRESSION_IMPL_HPP

#include "rvm_regression.hpp"

namespace mlpack {
namespace regression {

template<typename KernelType>
RVMRegression<KernelType>::RVMRegression(const KernelType& kernel,
                                         const bool centerData,
                                         const bool scaleData,
                                         const bool ard,
                                         double alphaThresh,
                                         double tol,
                                         int nIterMax) :

  kernel(kernel),
  centerData(centerData),
  scaleData(scaleData),
  ard(ard),
  alphaThresh(1e4),
  tol(1e-5),
  nIterMax(50) 
  { /*Nothing to do */ }

template <typename KernelType>
RVMRegression<KernelType>::RVMRegression(const KernelType& kernel,
                                         const bool centerData,
                                         const bool scaleData,
                                         const bool ard) :
  kernel(kernel),
  centerData(centerData),
  scaleData(scaleData),
  ard(ard),
  alphaThresh(1e4),
  tol(1e-5),
  nIterMax(50) 
  { /*Nothing to do*/ }

template<typename KernelType>
void RVMRegression<KernelType>::Train(const arma::mat& data,
                                      const arma::rowvec& responses)
{
  std::cout << "center : " << centerData
	    << "scale : " << scaleData
	    << std::endl;
  arma::mat phi;
  arma::rowvec t;

  // Preprocess the data.
  responsesOffset = CenterScaleData(data, responses, phi, t);

  // When ard is set to true the kernel is ignored and we work in the original 
  // input space. 
  if (!ard)
  {
    arma::mat kernelMatrix;
    relevantVectors = phi;
    applyKernel(phi, kernelMatrix);
    phi = std::move(kernelMatrix);
  }
  // Initialize the hyperparameters and begin with an infinitely broad prior.
  alpha = arma::colvec(phi.n_rows).fill(1e-6);
  arma::colvec gammai = arma::zeros<arma::colvec>(phi.n_rows);
  beta =  1 / arma::var(t, 1);
  
  // Loop variables.
  double normOmega = 1.0;
  double crit = 1.0;
  unsigned short i = 0;

  arma::mat subPhi;
  // Initiaze a vector of all the indices from the first
  // to the last point.
  arma::uvec allCols(phi.n_cols);
  for (size_t i = 0; i < phi.n_cols; ++i) { allCols(i) = i; }

  while ((crit > tol) && (i < nIterMax))
  {
    crit = -normOmega;
    activeSet = find(alpha < alphaThresh);

    // Prune out the inactive basis functions. This procedure speeds up
    // the algorithm.
    subPhi = phi.submat(activeSet, allCols);

    // Update the posterior statistics.
    matCovariance  = subPhi * subPhi.t() * beta;
    matCovariance.diag() += alpha(activeSet);
    matCovariance = inv(matCovariance);

    omega = matCovariance * subPhi * t.t() * beta;

    // Update alpha.
    gammai = 1 - matCovariance.diag() % alpha(activeSet);
    alpha(activeSet) = gammai / (omega % omega); 

    // Update beta.
    const arma::rowvec temp = t - omega.t() * subPhi;
    beta = (phi.n_cols - sum(gammai)) / dot(temp, temp);

    // Comptute the stopping criterion.
    normOmega = norm(omega);
    crit = std::abs(crit + normOmega) / normOmega;
    i++;
  }

  if (!ard)
  {
    arma::uvec allRows(relevantVectors.n_rows);
    for (size_t i = 0; i < relevantVectors.n_rows; ++i) { allRows(i) = i; }
    relevantVectors = relevantVectors.submat(allRows, activeSet);
  }
  
  // Keep the active basis functions only.
  else
  {
    if (centerData)
      dataOffset = dataOffset(activeSet);

    if (scaleData)
      dataScale = dataScale(activeSet);
  }
}

template<typename KernelType>
void RVMRegression<KernelType>::Predict(const arma::mat& points,
                                        arma::rowvec& predictions) const
{
  arma::mat matX;
  // Manage the kernel.
  if (!ard)
  {
    arma::mat kernelMatrix;
    CenterScaleDataPred(points, matX);
    applyKernel(relevantVectors, matX, kernelMatrix);
    matX = std::move(kernelMatrix);
  }
  else
  {
    arma::uvec allCols(points.n_cols);
    for ( size_t i = 0; i < allCols.n_elem; ++i) { allCols(i) = i; }
    matX = points.submat(activeSet, allCols);
    CenterScaleDataPred(matX, matX);
  }

  predictions = omega.t() * matX;
  if (centerData) predictions += responsesOffset;
}

template<typename KernelType>
void RVMRegression<KernelType>::Predict(const arma::mat& points,
                                        arma::rowvec& predictions,
                                        arma::rowvec& std) const
{
  arma::mat matX;
  // Manage the kernel.
  if (!ard)
  {
    arma::mat kernelMatrix;
    CenterScaleDataPred(points, matX);
    applyKernel(relevantVectors, matX, kernelMatrix);
    matX = std::move(kernelMatrix);
  }
  else
  {
    arma::uvec allCols(points.n_cols);
    for ( size_t i = 0; i < allCols.n_elem; ++i) { allCols(i) = i; }
    matX = points.submat(activeSet, allCols);
    CenterScaleDataPred(matX, matX);
  }
  
  predictions = omega.t() * matX;
  if (centerData) predictions += responsesOffset;
  // Compute standard devaiations.
  std = sqrt(Variance() + sum(matX % (matCovariance * matX)));
}

template<typename KernelType>
double RVMRegression<KernelType>::RMSE(const arma::mat& data,
                                       const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  Predict(data, predictions);
  return sqrt(mean(square(responses - predictions)));
}

template<typename KernelType>
void RVMRegression<KernelType>::applyKernel(const arma::mat& matX,
                                            const arma::mat& matY,
                                            arma::mat& kernelMatrix) const {

  // Check if the dimensions are consistent.
  if (matX.n_rows != matY.n_rows)
  {
    std::cout << "Error gramm : " << matX.n_rows << "!=" 
              << matY.n_rows << std::endl;
    throw std::invalid_argument("Number of features not consistent");
  }

  kernelMatrix = arma::mat(matX.n_cols, matY.n_cols);
  // Note that we only need to calculate the upper triangular part of the
  // kernel matrix, since it is symmetric. This helps minimize the number of
  // kernel evaluations.
  for (size_t i = 0; i < matX.n_cols; ++i)
    for (size_t j = 0; j < matY.n_cols; ++j)
      kernelMatrix(i, j) = kernel.Evaluate(matX.col(i), matY.col(j));
}

template<typename KernelType>
void RVMRegression<KernelType>::applyKernel(const arma::mat& matX,
					    arma::mat& kernelMatrix) const {


  kernelMatrix = arma::mat(matX.n_cols, matX.n_cols);
  // Note that we only need to calculate the upper triangular part of the
  // kernel matrix, since it is symmetric. This helps minimize the number of
  // kernel evaluations.
  for (size_t i = 0; i < matX.n_cols; ++i)
    for (size_t j = i; j < matX.n_cols; ++j)
      kernelMatrix(i, j) = kernel.Evaluate(matX.col(i), matX.col(j));
    
  // Copy to the lower triangular part of the matrix.
  for (size_t i = 1; i < matX.n_cols; ++i)
    for (size_t j = 0; j < i; ++j)
      kernelMatrix(i, j) = kernelMatrix(j, i);
}

template<typename KernelType>
double RVMRegression<KernelType>::CenterScaleData(
    const arma::mat& data,
    const arma::rowvec& responses,
    arma::mat& dataProc,
    arma::rowvec& responsesProc)
{
  // Initialize the offsets to their neutral forms.
  responsesOffset = 0.0;
  if (!centerData && !scaleData)
  {
    dataProc = arma::mat(const_cast<double*>(data.memptr()), data.n_rows, 
                                             data.n_cols, false, true);
    responsesProc = arma::rowvec(const_cast<double*>(responses.memptr()),
                                                     responses.n_elem, false,
                                                     true);
  }

  else if (centerData && !scaleData)
  {
    dataOffset = mean(data, 1);
    responsesOffset = mean(responses);
    dataProc = data.each_col() - dataOffset;
    responsesProc = responses - responsesOffset;
  }

  else if (!centerData && scaleData)
  {
    dataScale = stddev(data, 1, 1);
    dataProc = data.each_col() / dataScale;
    responsesProc = arma::rowvec(const_cast<double*>(responses.memptr()),
                                                     responses.n_elem, false,
                                                     true);
  }

  else
  {
    dataOffset = mean(data, 1);
    dataScale = stddev(data, 1, 1);
    responsesOffset = mean(responses);
    dataProc = (data.each_col() - dataOffset).each_col() / dataScale;
    responsesProc = responses - responsesOffset;
  }
  return responsesOffset;
}

template<typename KernelType>
void RVMRegression<KernelType>::CenterScaleDataPred(
    const arma::mat& data,
    arma::mat& dataProc) const
{
  if (!centerData && !scaleData)
  {
    dataProc = arma::mat(const_cast<double*>(data.memptr()), data.n_rows, 
                         data.n_cols, false, true);
  }

  else if (centerData && !scaleData)
  {
    dataProc = data.each_col() - dataOffset;
  }

  else if (!centerData && scaleData)
  {  
    dataProc = data.each_col() / dataScale;
  }

  else
  {
    dataProc = (data.each_col() - dataOffset).each_col() / dataScale;
  }
}


/**
 * Serialize the RVM regression model.
*/
template<typename KernelType>
template<typename Archive>
void RVMRegression<KernelType>::serialize(Archive& ar, 
                                          const uint32_t /* version */)
{
  ar(CEREAL_NVP(centerData));
  ar(CEREAL_NVP(scaleData));
  ar(CEREAL_NVP(relevantVectors));
  ar(CEREAL_NVP(omega));
  ar(CEREAL_NVP(responsesOffset));
  ar(CEREAL_NVP(kernel));
}

} // namespace regression
} // namespace mlpack

#endif
