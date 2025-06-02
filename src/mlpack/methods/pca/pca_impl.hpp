/**
 * @file methods/pca/pca_impl.hpp
 * @author Ajinkya Kale
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of PCA class to perform Principal Components Analysis on the
 * specified data set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_PCA_PCA_IMPL_HPP
#define MLPACK_METHODS_PCA_PCA_IMPL_HPP

#include <mlpack/prereqs.hpp>
#include "pca.hpp"

namespace mlpack {

template<typename DecompositionPolicy>
PCA<DecompositionPolicy>::PCA(
    const bool scaleData, const DecompositionPolicy& decomposition) :
    scaleData(scaleData),
    decomposition(decomposition)
{ }

/**
 * Apply Principal Component Analysis to the provided data set.
 *
 * @param data - Data matrix
 * @param transformedData - Data with PCA applied
 * @param eigVal - contains eigen values in a column vector
 * @param eigvec - PCA Loadings/Coeffs/EigenVectors
 */
template<typename DecompositionPolicy>
template<typename MatType, typename OutMatType, typename VecType>
void PCA<DecompositionPolicy>::Apply(const MatType& data,
                                     OutMatType& transformedData,
                                     VecType& eigVal,
                                     OutMatType& eigvec)
{
  // Sanity checks on input types.
  static_assert(IsBaseMatType<OutMatType>::value,
      "PCA::Apply(): transformedData must be a matrix type!");
  static_assert(IsBaseMatType<VecType>::value,
      "PCA::Apply(): eigVal must be a vector type!");
  static_assert(std::is_same_v<typename MatType::elem_type,
                               typename OutMatType::elem_type>,
      "PCA::Apply(): data and transformedData must have the same element "
      "types!");

  // Center the data into a temporary matrix.
  OutMatType centeredData = arma::conv_to<OutMatType>::from(data);
  centeredData.each_col() -= arma::mean(centeredData, 1);

  // Scale the data if the user asked for it.
  ScaleData(centeredData);

  decomposition.Apply(data, centeredData, transformedData, eigVal, eigvec,
      centeredData.n_rows);
}

/**
 * Apply Principal Component Analysis to the provided data set.
 *
 * @param data - Data matrix
 * @param transformedData - Data with PCA applied
 * @param eigVal - contains eigen values in a column vector
 */
template<typename DecompositionPolicy>
template<typename MatType, typename OutMatType, typename VecType>
void PCA<DecompositionPolicy>::Apply(const MatType& data,
                                     OutMatType& transformedData,
                                     VecType& eigVal)
{
  // Sanity checks on input types.
  static_assert(IsBaseMatType<OutMatType>::value,
      "PCA::Apply(): transformedData must be a matrix type!");
  static_assert(IsBaseMatType<VecType>::value,
      "PCA::Apply(): eigVal must be a vector type!");
  static_assert(std::is_same_v<typename MatType::elem_type,
                               typename OutMatType::elem_type>,
      "PCA::Apply(): data and transformedData must have the same element "
      "types!");

  OutMatType eigvec;
  Apply(data, transformedData, eigVal, eigvec);
}

/**
 * Apply Principal Component Analysis to the provided data set.
 *
 * @param data - Data matrix.
 * @param transformedData Data with PCA applied.
 */
template<typename DecompositionPolicy>
template<typename MatType, typename OutMatType>
void PCA<DecompositionPolicy>::Apply(const MatType& data,
                                     OutMatType& transformedData)
{
  // Sanity checks on input types.
  static_assert(IsBaseMatType<OutMatType>::value,
      "PCA::Apply(): transformedData must be a matrix type!");
  static_assert(std::is_same_v<typename MatType::elem_type,
                               typename OutMatType::elem_type>,
      "PCA::Apply(): data and transformedData must have the same element "
      "types!");

  // It's possible a user didn't pass in a matrix but instead an expression, but
  // we need a type that we can store.
  using BaseColType = typename GetDenseColType<MatType>::type;

  OutMatType eigvec;
  BaseColType eigVal;
  Apply(data, transformedData, eigVal, eigvec);
}

/**
 * Use PCA for dimensionality reduction on the given dataset.  This will save
 * the newDimension largest principal components of the data and remove the
 * rest.  The parameter returned is the amount of variance of the data that is
 * retained; this is a value between 0 and 1.  For instance, a value of 0.9
 * indicates that 90% of the variance present in the data was retained.
 *
 * @param data Data matrix.
 * @param newDimension New dimension of the data.
 * @return Amount of the variance of the data retained (between 0 and 1).
 */
template<typename DecompositionPolicy>
template<typename MatType>
double PCA<DecompositionPolicy>::Apply(MatType& data,
                                       const size_t newDimension)
{
  return Apply(data, data, newDimension);
}

template<typename DecompositionPolicy>
template<typename MatType, typename OutMatType>
double PCA<DecompositionPolicy>::Apply(const MatType& data,
                                       OutMatType& transformedData,
                                       const size_t newDimension)
{
  // Parameter validation.
  if (newDimension == 0)
  {
    std::ostringstream oss;
    oss << "PCA::Apply(): newDimension (" << newDimension << ") cannot be "
        << "zero!";
    throw std::invalid_argument(oss.str());
  }

  using BaseMatType = typename GetDenseMatType<MatType>::type;
  using BaseColType = typename GetDenseColType<MatType>::type;

  BaseMatType eigvec;
  BaseColType eigVal;

  // Center the data into a temporary matrix.
  BaseMatType centeredData = arma::conv_to<OutMatType>::from(data);
  centeredData.each_col() -= arma::mean(centeredData, 1);

  // This check cannot happen until here, as `data` may not have a .n_rows
  // member if it is an expression.
  if (newDimension > centeredData.n_rows)
  {
    std::ostringstream oss;
    oss << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
        << "be greater than the existing dimensionality of the data ("
        << data.n_rows << ")!";
    throw std::invalid_argument(oss.str());
  }

  // Scale the data if the user ask for.
  ScaleData(centeredData);

  decomposition.Apply(data, centeredData, transformedData, eigVal, eigvec,
      newDimension);

  if (newDimension < eigvec.n_rows)
    // Drop unnecessary rows.
    transformedData.shed_rows(newDimension, data.n_rows - 1);

  // The svd method returns only non-zero eigenvalues so we have to calculate
  // the right dimension before calculating the amount of variance retained.
  double eigDim = std::min(newDimension - 1, (size_t) eigVal.n_elem - 1);

  // Calculate the total amount of variance retained.
  return (sum(eigVal.subvec(0, eigDim)) / sum(eigVal));
}

/**
 * Use PCA for dimensionality reduction on the given dataset.  This will save
 * as many dimensions as necessary to retain at least the given amount of
 * variance (specified by parameter varRetained).  The amount should be
 * between 0 and 1; if the amount is 0, then only 1 dimension will be
 * retained.  If the amount is 1, then all dimensions will be retained.
 *
 * The method returns the actual amount of variance retained, which will
 * always be greater than or equal to the varRetained parameter.
 */
template<typename DecompositionPolicy>
template<typename MatType>
double PCA<DecompositionPolicy>::Apply(MatType& data,
                                       const double varRetained)
{
  return Apply(data, data, varRetained);
}

template<typename DecompositionPolicy>
template<typename MatType, typename OutMatType>
double PCA<DecompositionPolicy>::Apply(const MatType& data,
                                       OutMatType& transformedData,
                                       const double varRetained)
{
  // Parameter validation.
  if (varRetained < 0)
  {
    std::ostringstream oss;
    oss << "PCA::Apply(): varRetained (" << varRetained << ") must be greater "
        << "than or equal to 0.";
    throw std::invalid_argument(oss.str());
  }
  else if (varRetained > 1)
  {
    std::ostringstream oss;
    oss << "PCA::Apply(): varRetained (" << varRetained << ") should be less "
        << "than or equal to 1.";
    throw std::invalid_argument(oss.str());
  }

  using BaseMatType = typename GetDenseMatType<MatType>::type;
  using BaseColType = typename GetDenseColType<MatType>::type;

  BaseMatType eigvec;
  BaseColType eigVal;
  BaseMatType out;

  Apply(data, transformedData, eigVal, eigvec);

  // Calculate the dimension we should keep.
  size_t newDimension = 0;
  double varSum = 0.0;
  eigVal /= sum(eigVal); // Normalize eigenvalues.
  while ((varSum < varRetained) && (newDimension < eigVal.n_elem))
  {
    varSum += eigVal[newDimension];
    ++newDimension;
  }

  // varSum is the actual variance we will retain.
  if (newDimension < eigVal.n_elem)
    transformedData.shed_rows(newDimension, transformedData.n_rows - 1);

  return varSum;
}

} // namespace mlpack

#endif
