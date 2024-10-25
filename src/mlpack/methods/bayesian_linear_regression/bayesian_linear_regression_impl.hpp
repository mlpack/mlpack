/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression_impl.hpp
 * @author Clement Mercier
 *
 * Implementation of templated BayesianLinearRegression functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP

#include "bayesian_linear_regression.hpp"

namespace mlpack {

template<typename ModelMatType>
inline BayesianLinearRegression<ModelMatType>::BayesianLinearRegression(
    const bool centerData,
    const bool scaleData,
    const size_t maxIterations,
    const double tolerance) :
    centerData(centerData),
    scaleData(scaleData),
    maxIterations(maxIterations),
    tolerance(tolerance),
    responsesOffset(0.0),
    alpha(0.0),
    beta(0.0),
    gamma(0.0)
{ /* Nothing to do */ }

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename>
inline BayesianLinearRegression<ModelMatType>::BayesianLinearRegression(
    const MatType& data,
    const ResponsesType& responses,
    const bool centerData,
    const bool scaleData,
    const size_t maxIterations,
    const double tolerance) :
    centerData(centerData),
    scaleData(scaleData),
    maxIterations(maxIterations),
    tolerance(tolerance),
    responsesOffset(0.0),
    alpha(0.0),
    beta(0.0),
    gamma(0.0)
{
  // Train the model.
  Train(data, responses);
}

template<typename ModelMatType>
template<typename MatType>
inline
typename BayesianLinearRegression<ModelMatType>::ElemType
BayesianLinearRegression<ModelMatType>::Train(
    const MatType& data,
    const arma::rowvec& responses)
{
  return Train(data, responses, this->centerData, this->scaleData,
      this->maxIterations, this->tolerance);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline
typename BayesianLinearRegression<ModelMatType>::ElemType
BayesianLinearRegression<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const std::optional<bool> centerData,
    const std::optional<bool> scaleData,
    const std::optional<size_t> maxIterations)
{
  return Train(data, responses,
      (centerData.has_value()) ? centerData.value() : this->centerData,
      (scaleData.has_value()) ? scaleData.value() : this->scaleData,
      (maxIterations.has_value()) ? maxIterations.value() : this->maxIterations,
      this->tolerance);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline
typename BayesianLinearRegression<ModelMatType>::ElemType
BayesianLinearRegression<ModelMatType>::Train(
    const MatType& data,
    const ResponsesType& responses,
    const bool centerData,
    const bool scaleData,
    const size_t maxIterations,
    const double tolerance)
{
  this->centerData = centerData;
  this->scaleData = scaleData;
  this->maxIterations = maxIterations;
  this->tolerance = tolerance;

  ModelMatType phi;
  DenseRowType t;
  DenseVecType eigVal;
  ModelMatType eigVec;

  // Preprocess the data. Center and scale.
  responsesOffset = CenterScaleData(data, responses, phi, t);

  if (!arma::eig_sym(eigVal, eigVec, arma::symmatu(phi * phi.t())))
  {
    Log::Fatal << "BayesianLinearRegression::Train(): Eigendecomposition "
               << "of covariance failed!" << std::endl;
  }

  // Compute this quantities once and for all.
  const ModelMatType eigVecInv = inv(eigVec);
  const DenseVecType eigVecInvPhitT = eigVecInv * phi * t.t();

  // Initialize the hyperparameters and begin with an infinitely broad prior.
  alpha = ((ElemType) 1e-6);
  beta = ((ElemType) 1 / (var(t, 1) * 0.1));

  unsigned short i = 0;
  ElemType crit = ((ElemType) 1.0);

  while (((double) crit > tolerance) && (i < maxIterations))
  {
    ElemType deltaAlpha = -alpha;
    ElemType deltaBeta = -beta;

    // Update the solution.
    omega = eigVec * diagmat(1 / (eigVal + (alpha / beta))) * eigVecInvPhitT;

    // Update alpha.
    gamma = sum(eigVal / (alpha / beta + eigVal));
    alpha = gamma / dot(omega, omega);

    // Update beta.
    const DenseRowType temp = t - omega.t() * phi;
    beta = (data.n_cols - gamma) / dot(temp, temp);

    // Compute the stopping criterion.
    deltaAlpha += alpha;
    deltaBeta += beta;
    crit = std::abs(deltaAlpha / alpha + deltaBeta / beta);
    i++;
  }
  // Compute the covariance matrix for the uncertainties later.
  matCovariance = eigVec * diagmat(((ElemType) 1) / (beta * eigVal + alpha)) *
      eigVecInv;

  return RMSE(data, responses);
}

template<typename ModelMatType>
template<typename VecType>
inline
typename BayesianLinearRegression<ModelMatType>::ElemType
BayesianLinearRegression<ModelMatType>::Predict(const VecType& point) const
{
  // Center and scale the point before applying the model, if needed.
  if (!centerData && !scaleData)
    return dot(omega, point) + responsesOffset;
  else if (centerData && !scaleData)
    return dot(omega, point - dataOffset) + responsesOffset;
  else if (!centerData && scaleData)
    return dot(omega, point / dataScale) + responsesOffset;
  else
    return dot(omega, (point - dataOffset) / dataScale) + responsesOffset;
}

template<typename ModelMatType>
template<typename VecType>
inline void BayesianLinearRegression<ModelMatType>::Predict(
    const VecType& point,
    typename BayesianLinearRegression<ModelMatType>::ElemType& prediction,
    typename BayesianLinearRegression<ModelMatType>::ElemType& stddev) const
{
  prediction = Predict(point);

  ElemType inner;
  if (!centerData && !scaleData)
  {
    stddev = std::sqrt(Variance() +
        accu(point % (matCovariance * point)));
    inner = accu(point % (matCovariance * point));
  }
  else if (centerData && !scaleData)
  {
    inner = accu((point - dataOffset) %
        (matCovariance * (point - dataOffset)));
  }
  else if (!centerData && scaleData)
  {
    inner = accu((point / dataScale) %
        (matCovariance * (point / dataScale)));
  }
  else
  {
    inner = accu(((point - dataOffset) / dataScale) %
        (matCovariance * ((point - dataOffset) / dataScale)));
  }

  stddev = std::sqrt(Variance() + inner);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename>
inline void BayesianLinearRegression<ModelMatType>::Predict(
    const MatType& points,
    ResponsesType& predictions) const
{
  if (!centerData && !scaleData)
  {
    predictions = omega.t() * points + responsesOffset;
  }
  else
  {
    // Center and scale the points before applying the model.
    arma::Mat<ElemType> pointsProc;
    CenterScaleDataPred(points, pointsProc);

    predictions = omega.t() * pointsProc + responsesOffset;
  }
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename>
inline void BayesianLinearRegression<ModelMatType>::Predict(
    const MatType& points,
    ResponsesType& predictions,
    ResponsesType& std) const
{
  if (!centerData && !scaleData)
  {
    Predict(points, predictions);
    std = sqrt(Variance() + sum(points % (matCovariance * points), 0));
  }
  else
  {
    // Center or scale data.
    arma::Mat<ElemType> pointsProc;
    CenterScaleDataPred(points, pointsProc);

    predictions = omega.t() * pointsProc + responsesOffset;
    std = sqrt(Variance() + sum(pointsProc % (matCovariance * pointsProc), 0));
  }
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename>
inline
typename BayesianLinearRegression<ModelMatType>::ElemType
BayesianLinearRegression<ModelMatType>::RMSE(
    const MatType& data,
    const ResponsesType& responses) const
{
  typename GetDenseRowType<ResponsesType>::type predictions;
  Predict(data, predictions);
  return std::sqrt(mean(square(responses - predictions)));
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType>
inline double BayesianLinearRegression<ModelMatType>::CenterScaleData(
    const MatType& data,
    const ResponsesType& responses,
    MatType& dataProc,
    ResponsesType& responsesProc)
{
  if (!centerData && !scaleData)
  {
    dataProc = MatType(const_cast<ElemType*>(data.memptr()), data.n_rows,
                                             data.n_cols, false, true);
    responsesProc = ResponsesType(const_cast<ElemType*>(responses.memptr()),
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
    dataScale = stddev(data, 0, 1);
    dataProc = data.each_col() / dataScale;
    responsesProc = ResponsesType(const_cast<ElemType*>(responses.memptr()),
                                                        responses.n_elem, false,
                                                        true);
  }
  else
  {
    dataOffset = mean(data, 1);
    dataScale = stddev(data, 0, 1);
    responsesOffset = mean(responses);
    dataProc = (data.each_col() - dataOffset).each_col() / dataScale;
    responsesProc = responses - responsesOffset;
  }

  return responsesOffset;
}

template<typename ModelMatType>
template<typename MatType, typename OutMatType>
inline void BayesianLinearRegression<ModelMatType>::CenterScaleDataPred(
    const MatType& data,
    OutMatType& dataProc) const
{
  if (!centerData && !scaleData)
  {
    return; // Don't modify dataProc.
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
 * Serialize the Bayesian linear regression model.
 */
template<typename ModelMatType>
template<typename Archive>
void BayesianLinearRegression<ModelMatType>::serialize(Archive& ar,
                                                       const uint32_t version)
{
  ar(CEREAL_NVP(centerData));
  ar(CEREAL_NVP(scaleData));
  ar(CEREAL_NVP(maxIterations));
  ar(CEREAL_NVP(tolerance));

  // In older versions, dataOffset and dataScale were of type arma::colvec,
  // responsesOffset, alpha, beta, and gamma were of type double, omega was of
  // type arma::colvec, and matCovariance was of type arma::mat.
  if (cereal::is_loading<Archive>() && version == 0)
  {
    arma::colvec colvecTmp;
    ar(cereal::make_nvp("dataOffset", colvecTmp));
    dataOffset = ConvTo<DenseVecType>::From(colvecTmp);

    ar(cereal::make_nvp("dataScale", colvecTmp));
    dataScale = ConvTo<DenseVecType>::From(colvecTmp);

    double dblTmp;
    ar(cereal::make_nvp("responsesOffset", dblTmp));
    responsesOffset = (ElemType) dblTmp;

    ar(cereal::make_nvp("alpha", dblTmp));
    alpha = (ElemType) dblTmp;

    ar(cereal::make_nvp("beta", dblTmp));
    beta = (ElemType) dblTmp;

    ar(cereal::make_nvp("gamma", dblTmp));
    gamma = (ElemType) dblTmp;

    ar(cereal::make_nvp("omega", colvecTmp));
    omega = ConvTo<DenseVecType>::From(colvecTmp);

    arma::mat matTmp;
    ar(cereal::make_nvp("matCovariance", matTmp));
    matCovariance = ConvTo<ModelMatType>::From(matCovariance);
  }
  else
  {
    ar(CEREAL_NVP(dataOffset));
    ar(CEREAL_NVP(dataScale));
    ar(CEREAL_NVP(responsesOffset));
    ar(CEREAL_NVP(alpha));
    ar(CEREAL_NVP(beta));
    ar(CEREAL_NVP(gamma));
    ar(CEREAL_NVP(omega));
    ar(CEREAL_NVP(matCovariance));
  }
}

} // namespace mlpack

#endif
