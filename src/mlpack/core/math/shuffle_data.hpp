/**
 * @file shuffle_data.hpp
 * @author Ryan Curtin
 *
 * Given data points and labels, shuffle their ordering.
 */
#ifndef MLPACK_CORE_MATH_SHUFFLE_DATA_HPP
#define MLPACK_CORE_MATH_SHUFFLE_DATA_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace math {

/**
 * Shuffle a dataset and associated labels (or responses).  It is expected that
 * inputPoints and inputLabels have the same number of columns (so, be sure that
 * inputLabels, if it is a vector, is a row vector).
 *
 * Shuffled data will be output into outputPoints and outputLabels.
 */
template<typename MatType, typename LabelsType>
void ShuffleData(const MatType& inputPoints,
                 const LabelsType& inputLabels,
                 MatType& outputPoints,
                 LabelsType& outputLabels,
                 const std::enable_if_t<!arma::is_SpMat<MatType>::value>* = 0)
{
  // Generate ordering.
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      inputPoints.n_cols - 1, inputPoints.n_cols));

  outputPoints = inputPoints.cols(ordering);
  outputLabels = inputLabels.cols(ordering);
}

/**
 * Shuffle a sparse dataset and associated labels (or responses).  It is
 * expected that inputPoints and inputLabels have the same number of columns
 * (so, be sure that inputLabels, if it is a vector, is a row vector).
 *
 * Shuffled data will be output into outputPoints and outputLabels.
 */
template<typename MatType, typename LabelsType>
void ShuffleData(const MatType& inputPoints,
                 const LabelsType& inputLabels,
                 MatType& outputPoints,
                 LabelsType& outputLabels,
                 const std::enable_if_t<arma::is_SpMat<MatType>::value>* = 0)
{
  // Generate ordering.
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      inputPoints.n_cols - 1, inputPoints.n_cols));

  // Extract coordinate list representation.
  arma::umat locations(2, inputPoints.n_nonzero);
  arma::Col<typename MatType::elem_type> values(
      const_cast<typename MatType::elem_type*>(inputPoints.values),
      inputPoints.n_nonzero, false, true);
  typename MatType::const_iterator it = inputPoints.begin();
  size_t index = 0;
  while (it != inputPoints.end())
  {
    locations(0, index) = it.row();
    locations(1, index) = ordering[it.col()];
    ++it;
    ++index;
  }

  outputPoints = MatType(locations, values, inputPoints.n_rows,
      inputPoints.n_cols, true);
  outputLabels = inputLabels.cols(ordering);
}

} // namespace math
} // namespace mlpack

#endif
