/**
 * @file methods/sefr/sefr_impl.hpp
 * @author Hamidreza Keshavarz
 *
 * Implementation of the SEFR classifier.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SEFR_SEFR_IMPL_HPP
#define MLPACK_METHODS_SEFR_SEFR_IMPL_HPP

// In case it hasn't been included yet.
#include "sefr.hpp"

namespace mlpack {

template<typename ModelMatType>
SEFR<ModelMatType>::SEFR(const size_t numClasses,
                         const size_t dimensionality) :
    weights(dimensionality, numClasses, arma::fill::zeros),
    biases(numClasses, arma::fill::zeros),
    classSums(dimensionality, numClasses, arma::fill::zeros),
    classCounts(numClasses, arma::fill::zeros),
    offsets(dimensionality, arma::fill::zeros)
{
  // Nothing to do.
}

template<typename ModelMatType>
template<typename MatType>
SEFR<ModelMatType>::SEFR(const MatType& data,
                         const arma::Row<size_t>& labels,
                         const size_t numClasses)
{
  Train(data, labels, numClasses);
}

template<typename ModelMatType>
template<typename MatType, typename WeightsType>
SEFR<ModelMatType>::SEFR(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const WeightsType& instanceWeights,
    const std::enable_if_t<arma::is_arma_type<WeightsType>::value>*)
{
  Train(data, labels, numClasses, instanceWeights);
}

template<typename ModelMatType>
template<typename MatType>
void SEFR<ModelMatType>::Train(const MatType& data,
                               const arma::Row<size_t>& labels,
                               const size_t numClasses)
{
  CheckTrainingData(data, labels, numClasses);

  classSums.zeros(data.n_rows, numClasses);
  classCounts.zeros(numClasses);
  ComputeOffsets(data);
  Accumulate(data, labels, arma::Row<ElemType>(labels.n_elem,
      arma::fill::ones));
  ComputeModel();
}

template<typename ModelMatType>
template<typename MatType, typename WeightsType>
void SEFR<ModelMatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const WeightsType& instanceWeights,
    const std::enable_if_t<arma::is_arma_type<WeightsType>::value>*)
{
  CheckTrainingData(data, labels, numClasses);
  util::CheckSameSizes(data, (size_t) instanceWeights.n_elem, "SEFR::Train()",
      "weights");

  classSums.zeros(data.n_rows, numClasses);
  classCounts.zeros(numClasses);
  ComputeOffsets(data);
  Accumulate(data, labels,
      arma::conv_to<arma::Row<ElemType>>::from(instanceWeights));
  ComputeModel();
}

template<typename ModelMatType>
template<typename VecType>
void SEFR<ModelMatType>::Train(const VecType& point, const size_t label)
{
  util::CheckSameDimensionality(point, classSums.n_rows, "SEFR::Train()",
      "point");
  if (label >= classSums.n_cols)
  {
    std::ostringstream oss;
    oss << "SEFR::Train(): label (" << label << ") must be less than the "
        << "number of classes (" << classSums.n_cols << ")!";
    throw std::invalid_argument(oss.str());
  }

  classSums.col(label) += point;
  classCounts[label] += 1;
  offsets = arma::min(offsets, DenseColType(point));
  ComputeModel();
}

template<typename ModelMatType>
template<typename VecType>
size_t SEFR<ModelMatType>::Classify(const VecType& point) const
{
  size_t label;
  DenseColType scores;
  Classify(point, label, scores);
  return label;
}

template<typename ModelMatType>
template<typename VecType>
void SEFR<ModelMatType>::Classify(const VecType& point,
                                  size_t& label,
                                  DenseColType& scores) const
{
  util::CheckSameDimensionality(point, weights.n_rows, "SEFR::Classify()",
      "point");

  scores = weights.t() * point + biases;
  label = scores.index_max();
}

template<typename ModelMatType>
template<typename MatType>
void SEFR<ModelMatType>::Classify(const MatType& data,
                                  arma::Row<size_t>& labels) const
{
  DenseMatType scores;
  Classify(data, labels, scores);
}

template<typename ModelMatType>
template<typename MatType>
void SEFR<ModelMatType>::Classify(const MatType& data,
                                  arma::Row<size_t>& labels,
                                  DenseMatType& scores) const
{
  static_assert(std::is_same_v<typename MatType::elem_type, ElemType>,
      "SEFR::Classify(): element type of data must match element type of the "
      "model");

  util::CheckSameDimensionality(data, weights.n_rows, "SEFR::Classify()");

  scores = weights.t() * data;
  scores.each_col() += biases;
  labels = arma::conv_to<arma::Row<size_t>>::from(
      arma::index_max(scores, 0));
}

template<typename ModelMatType>
void SEFR<ModelMatType>::Reset()
{
  weights.zeros();
  biases.zeros();
  classSums.zeros();
  classCounts.zeros();
  offsets.zeros();
}

template<typename ModelMatType>
template<typename Archive>
void SEFR<ModelMatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(weights));
  ar(CEREAL_NVP(biases));
  ar(CEREAL_NVP(classSums));
  ar(CEREAL_NVP(classCounts));
  ar(CEREAL_NVP(offsets));
}

template<typename ModelMatType>
template<typename MatType>
void SEFR<ModelMatType>::CheckTrainingData(const MatType& data,
                                           const arma::Row<size_t>& labels,
                                           const size_t numClasses) const
{
  util::CheckSameSizes(data, labels, "SEFR::Train()");
  if (numClasses < 2)
  {
    throw std::invalid_argument("SEFR::Train(): number of classes must be at "
        "least 2!");
  }
  if (labels.n_elem > 0 && labels.max() >= numClasses)
  {
    std::ostringstream oss;
    oss << "SEFR::Train(): labels must be less than the number of classes ("
        << numClasses << ")!";
    throw std::invalid_argument(oss.str());
  }
}

template<typename ModelMatType>
template<typename MatType>
void SEFR<ModelMatType>::ComputeOffsets(const MatType& data)
{
  offsets.zeros(data.n_rows);
  if (data.n_cols == 0)
    return;

  // For sparse data the implicit zeros count, so a non-negative sparse
  // feature gets an offset of 0 and the data stays sparse.
  offsets = arma::min(offsets, DenseColType(arma::min(data, 1)));
}

template<typename ModelMatType>
template<typename MatType, typename WeightsType>
void SEFR<ModelMatType>::Accumulate(const MatType& data,
                                    const arma::Row<size_t>& labels,
                                    const WeightsType& instanceWeights)
{
  static_assert(std::is_same_v<typename MatType::elem_type, ElemType>,
      "SEFR::Train(): element type of data must match element type of the "
      "model");

  if (labels.n_elem == 0)
    return;

  arma::umat locations(2, labels.n_elem);
  locations.row(0) = arma::regspace<arma::urowvec>(0, labels.n_elem - 1);
  locations.row(1) = arma::conv_to<arma::urowvec>::from(labels);
  const arma::SpMat<ElemType> indicator(locations,
      arma::conv_to<arma::Col<ElemType>>::from(instanceWeights),
      data.n_cols, classSums.n_cols);

  classSums += DenseMatType(data * indicator);
  for (size_t i = 0; i < labels.n_elem; ++i)
    classCounts[labels[i]] += instanceWeights[i];
}

template<typename ModelMatType>
void SEFR<ModelMatType>::ComputeModel()
{
  const size_t numClasses = classSums.n_cols;
  weights.set_size(classSums.n_rows, numClasses);
  biases.set_size(numClasses);

  const DenseColType totalSum = arma::sum(classSums, 1);
  const ElemType totalCount = arma::accu(classCounts);
  const ElemType eps = ElemType(1e-7);

  for (size_t c = 0; c < numClasses; ++c)
  {
    const ElemType posCount = classCounts[c];
    const ElemType negCount = totalCount - posCount;
    if (posCount <= 0)
    {
      weights.col(c).zeros();
      biases[c] = std::numeric_limits<ElemType>::lowest();
      continue;
    }
    if (negCount <= 0)
    {
      weights.col(c).zeros();
      biases[c] = 0;
      continue;
    }

    // Means of the shifted data x - offsets, which is non-negative.
    const DenseColType posMean = classSums.col(c) / posCount - offsets;
    const DenseColType negMean = (totalSum - classSums.col(c)) / negCount -
        offsets;
    weights.col(c) = (posMean - negMean) / (posMean + negMean + eps);
    // The bias is computed for shifted data; subtracting w' * offsets lets the
    // model score unshifted data: w' * (x - offsets) + b = w' * x + (b - w' *
    // offsets).
    biases[c] = -(negCount * arma::dot(weights.col(c), posMean) +
        posCount * arma::dot(weights.col(c), negMean)) / totalCount -
        arma::dot(weights.col(c), offsets);
  }
}

} // namespace mlpack

#endif
