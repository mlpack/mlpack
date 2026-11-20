/*
 * @file methods/adaboost/adaboost_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of the AdaBoost class.
 *
 * @code
 * @article{schapire1999improved,
 *   author = {Schapire, Robert E. and Singer, Yoram},
 *   title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
 *   journal = {Machine Learning},
 *   volume = {37},
 *   number = {3},
 *   month = dec,
 *   year = {1999},
 *   issn = {0885-6125},
 *   pages = {297--336},
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP
#define MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP

#include "adaboost.hpp"

namespace mlpack {

// Empty constructor.
template<typename WeakLearnerType, typename MatType>
AdaBoost<WeakLearnerType, MatType>::AdaBoost(const ElemType tolerance) :
    numClasses(0),
    tolerance(tolerance)
{
  // Nothing to do.
}

/**
 * Constructor. Runs the AdaBoost.MH algorithm.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param maxIterations Number of boosting rounds
 * @param tol Tolerance for termination of Adaboost.MH.
 * @param other Weak Learner, which has been initialized already.
 */
template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
AdaBoost<WeakLearnerType, MatType>::AdaBoost(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const WeakLearnerInType& other,
    const size_t maxIterations,
    const typename MatType::elem_type tol,
    const std::enable_if_t<
        std::is_same_v<WeakLearnerType, WeakLearnerInType>>*) :
    maxIterations(maxIterations),
    tolerance(tol)
{
  (void) TrainInternal<true>(data, labels, numClasses, other);
}

/**
 * Constructor. Runs the AdaBoost.MH algorithm.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param maxIterations Number of boosting rounds
 * @param tol Tolerance for termination of Adaboost.MH.
 * @param other Weak Learner, which has been initialized already.
 */
template<typename WeakLearnerType, typename MatType>
template<typename... WeakLearnerArgs>
AdaBoost<WeakLearnerType, MatType>::AdaBoost(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t maxIterations,
    const typename MatType::elem_type tol,
    WeakLearnerArgs&&... weakLearnerArgs) :
    maxIterations(maxIterations),
    tolerance(tol)
{
  WeakLearnerType other; // Will not be used.
  (void) TrainInternal<false>(data, labels, numClasses, other,
      weakLearnerArgs...);
}

// Train AdaBoost with a given weak learner, and set the maximum number of
// iterations and tolerance.
template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
typename MatType::elem_type AdaBoost<WeakLearnerType, MatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const WeakLearnerInType& other,
    const std::optional<size_t> maxIterations,
    const std::optional<double> tolerance,
    const std::enable_if_t<
        std::is_same_v<WeakLearnerType, WeakLearnerInType>>*)
{
  if (maxIterations.has_value())
    this->maxIterations = maxIterations.value();

  if (tolerance.has_value())
    this->tolerance = tolerance.value();

  return TrainInternal<true>(data, labels, numClasses, other);
}

// Train AdaBoost.
template<typename WeakLearnerType, typename MatType>
template<typename... WeakLearnerArgs>
typename MatType::elem_type AdaBoost<WeakLearnerType, MatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const std::optional<size_t> maxIterations,
    const std::optional<double> tolerance,
    WeakLearnerArgs&&... weakLearnerArgs)
{
  if (maxIterations.has_value())
    this->maxIterations = maxIterations.value();

  if (tolerance.has_value())
    this->tolerance = tolerance.value();

  WeakLearnerType other; // Will not be used.
  return TrainInternal<false>(data, labels, numClasses, other,
      weakLearnerArgs...);
}

// Classify the given test point.
template<typename WeakLearnerType, typename MatType>
template<typename VecType>
size_t AdaBoost<WeakLearnerType, MatType>::Classify(const VecType& point) const
{
  arma::Row<ElemType> probabilities;
  size_t prediction;
  Classify(point, prediction, probabilities);

  return prediction;
}

// Classify the given test point and return class probabilities.
template<typename WeakLearnerType, typename MatType>
template<typename VecType>
void AdaBoost<WeakLearnerType, MatType>::Classify(
    const VecType& point,
    size_t& prediction,
    arma::Row<typename MatType::elem_type>& probabilities) const
{
  probabilities.zeros(numClasses);
  for (size_t i = 0; i < wl.size(); ++i)
  {
    prediction = wl[i].Classify(point);
    probabilities(prediction) += alpha[i];
  }

  probabilities /= accu(probabilities);
  arma::uword maxIndex = probabilities.index_max();
  prediction = (size_t) maxIndex;
}

// Classify the given test points.
template<typename WeakLearnerType, typename MatType>
void AdaBoost<WeakLearnerType, MatType>::Classify(
    const MatType& test,
    arma::Row<size_t>& predictedLabels) const
{
  arma::Row<size_t> tempPredictedLabels(test.n_cols);
  arma::Mat<ElemType> probabilities;

  Classify(test, predictedLabels, probabilities);
}

// Classify the given test points.
template<typename WeakLearnerType, typename MatType>
void AdaBoost<WeakLearnerType, MatType>::Classify(
    const MatType& test,
    arma::Row<size_t>& predictedLabels,
    arma::Mat<typename MatType::elem_type>& probabilities) const
{
  probabilities.zeros(numClasses, test.n_cols);
  predictedLabels.set_size(test.n_cols);

  for (size_t i = 0; i < wl.size(); ++i)
  {
    wl[i].Classify(test, predictedLabels);

    for (size_t j = 0; j < predictedLabels.n_cols; ++j)
      probabilities(predictedLabels(j), j) += alpha[i];
  }

  arma::uword maxIndex = 0;

  for (size_t i = 0; i < predictedLabels.n_cols; ++i)
  {
    probabilities.col(i) /= accu(probabilities.col(i));
    maxIndex = probabilities.col(i).index_max();
    predictedLabels(i) = maxIndex;
  }
}

/**
 * Serialize the AdaBoost model.
 */
template<typename WeakLearnerType, typename MatType>
template<typename Archive>
void AdaBoost<WeakLearnerType, MatType>::serialize(Archive& ar,
                                                   const uint32_t version)
{
  // Between version 0 and 1, the maxIterations member was added, and `alpha`
  // was switched to type arma::Row<ElemType> instead of arma::rowvec.  These
  // require a little bit of special handling when loading older versions.
  if (cereal::is_loading<Archive>() && version == 0)
  {
    // This is the legacy version.
    ar(CEREAL_NVP(numClasses));
    ar(CEREAL_NVP(tolerance));
    ar(CEREAL_NVP(alpha));

    // In earlier versions, `alpha` was a vector of doubles---but it might not
    // be now.
    if (std::is_same_v<ElemType, double>)
    {
      ar(CEREAL_NVP(alpha)); // The easy case.
    }
    else
    {
      arma::rowvec alphaTmp;
      // Avoid CEREAL_NVP so we can specify a custom name.
      ar(cereal::make_nvp("alpha", alphaTmp));
      alpha.clear();
      alpha.resize(alphaTmp.size());
      for (size_t i = 0; i < alphaTmp.size(); ++i)
        alpha[i] = (ElemType) alphaTmp[i];
    }

    // Now serialize each weak learner.
    ar(CEREAL_NVP(wl));

    // Attempt to set maxIterations to something reasonable.
    maxIterations = std::max((size_t) 100, alpha.size());
  }
  else
  {
    // This is the current version.
    // (Once there is a major version bump, we should make this version 0.)
    ar(CEREAL_NVP(numClasses));
    ar(CEREAL_NVP(tolerance));
    ar(CEREAL_NVP(maxIterations));
    ar(CEREAL_NVP(alpha));

    // Now serialize each weak learner.
    ar(CEREAL_NVP(wl));
  }
}

template<
    bool UseExistingWeakLearner,
    typename MatType,
    typename WeightsType,
    typename WeakLearnerType,
    typename... WeakLearnerArgs
>
struct WeakLearnerTrainer
{
  static WeakLearnerType Train(
      const MatType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const WeightsType& weights,
      const WeakLearnerType& wl,
      WeakLearnerArgs&&... /* weakLearnerArgs */)
  {
    // Use the existing weak learner to train a new one with new weights.
    // API requirement: there is a constructor with this signature:
    //
    //    WeakLearnerType(const WeakLearnerType&,
    //                    MatType& data,
    //                    LabelsType& labels,
    //                    const size_t numClasses,
    //                    WeightsType& weights)
    //
    // This trains the new WeakLearnerType using the hyperparameters from the
    // given WeakLearnerType.
    return WeakLearnerType(wl, data, labels, numClasses, weights);
  }
};

template<
    typename MatType,
    typename WeightsType,
    typename WeakLearnerType,
    typename... WeakLearnerArgs
>
struct WeakLearnerTrainer<
    false, MatType, WeightsType, WeakLearnerType, WeakLearnerArgs...
>
{
  static WeakLearnerType Train(
      const MatType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const WeightsType& weights,
      const WeakLearnerType& /* wl */,
      WeakLearnerArgs&&... weakLearnerArgs)
  {
    // When UseExistingWeakLearner is false, we use the given hyperparameters.
    // (This is the preferred approach that supports more types of weak
    // learners.)
    return WeakLearnerType(data, labels, numClasses, weights,
        weakLearnerArgs...);
  }
};

template<typename WeakLearnerType, typename MatType>
template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>
typename MatType::elem_type AdaBoost<WeakLearnerType, MatType>::TrainInternal(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const WeakLearnerType& other,
    WeakLearnerArgs&&... weakLearnerArgs)
{
  // Clear information from previous runs.
  wl.clear();
  alpha.clear();

  this->numClasses = numClasses;

  // crt is the cumulative rt value for terminating the optimization when rt is
  // changing by less than the tolerance.
  ElemType rt, crt = 0.0, alphat = 0.0, zt;

  ElemType ztProduct = 1.0;

  // To be used for prediction by the weak learner.
  arma::Row<size_t> predictedLabels(labels.n_cols);

  // Use tempData to modify input data for incorporating weights.
  MatType tempData(data);

  // This matrix is a helper matrix used to calculate the final hypothesis.
  MatType sumFinalH(numClasses, predictedLabels.n_cols);
  sumFinalH.zeros();

  // Load the initial weights into a 2-D matrix.
  const ElemType initWeight = 1.0 / ElemType(data.n_cols * numClasses);
  MatType D(numClasses, data.n_cols);
  D.fill(initWeight);

  // Weights are stored in this row vector.
  arma::Row<ElemType> weights(predictedLabels.n_cols);

  // This is the final hypothesis.
  arma::Row<size_t> finalH(predictedLabels.n_cols);

  // Now, start the boosting rounds.
  for (size_t i = 0; i < maxIterations; ++i)
  {
    // Initialized to zero in every round.  rt is used for calculation of
    // alphat; it is the weighted error.
    // rt = (sum) D(i) y(i) ht(xi)
    rt = 0.0;

    // zt is used for weight normalization.
    zt = 0.0;

    // Build the weight vectors.
    weights = sum(D);

    // This is split into a separate function, so that we can still call
    // AdaBoost::Train() with extra hyperparameters, even when the weak learner
    // type does not support the special constructor that takes another weak
    // learner.
    WeakLearnerType w = WeakLearnerTrainer<
        UseExistingWeakLearner, MatType, arma::Row<ElemType>, WeakLearnerType,
        WeakLearnerArgs...
    >::Train(tempData, labels, numClasses, weights, other, weakLearnerArgs...);

    w.Classify(tempData, predictedLabels);

    // Now, calculate alpha(t) using ht.
    for (size_t j = 0; j < D.n_cols; ++j) // instead of D, ht
    {
      if (predictedLabels(j) == labels(j))
        rt += accu(D.col(j));
      else
        rt -= accu(D.col(j));
    }

    if ((i > 0) && (std::abs(rt - crt) < tolerance))
      break;

    // Check if model has converged.
    if (rt >= 1.0)
    {
      // Save the weak learner and terminate.
      alpha.push_back(1.0);
      wl.push_back(w);
      break;
    }

    crt = rt;

    // Our goal is to find alphat which minimizes or approximately minimizes the
    // value of Z as a function of alpha.
    alphat = 0.5 * std::log((1 + rt) / (1 - rt));

    alpha.push_back(alphat);
    wl.push_back(w);

    // Now start modifying the weights.
    for (size_t j = 0; j < D.n_cols; ++j)
    {
      const ElemType expo = std::exp(alphat);
      if (predictedLabels(j) == labels(j))
      {
        for (size_t k = 0; k < D.n_rows; ++k)
        {
          // We calculate zt, the normalization constant.
          D(k, j) /= expo;
          zt += D(k, j); // * exp(-1 * alphat * yt(j,k) * ht(j,k));

          // Add to the final hypothesis matrix.
          // sumFinalH(k, j) += (alphat * ht(k, j));
          if (k == labels(j))
            sumFinalH(k, j) += (alphat); // * ht(k, j));
          else
            sumFinalH(k, j) -= (alphat);
        }
      }
      else
      {
        for (size_t k = 0; k < D.n_rows; ++k)
        {
          // We calculate zt, the normalization constant.
          D(k, j) *= expo;
          zt += D(k, j);

          // Add to the final hypothesis matrix.
          if (k == labels(j))
            sumFinalH(k, j) += alphat; // * ht(k, j));
          else
            sumFinalH(k, j) -= alphat;
        }
      }
    }

    // Normalize D.
    D /= zt;

    // Accumulate the value of zt for the Hamming loss bound.
    ztProduct *= zt;
  }

  return ztProduct;
}

} // namespace mlpack

#endif
