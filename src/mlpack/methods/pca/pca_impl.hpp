/**
 * @file pca_impl.hpp
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

#include <mlpack/core.hpp>
#include "pca.hpp"

using namespace std;

namespace mlpack {
namespace pca {

template<typename DecompositionPolicy>
PCAType<DecompositionPolicy>::PCAType(const bool scaleData,
                                      const DecompositionPolicy& decomposition) :
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
void PCAType<DecompositionPolicy>::Apply(const arma::mat& data,
                                         arma::mat& transformedData,
                                         arma::vec& eigVal,
                                         arma::mat& eigvec)
{
  Timer::Start("pca");

  // Center the data into a temporary matrix.
  arma::mat centeredData;
  math::Center(data, centeredData);

  // Scale the data if the user ask for.
  ScaleData(centeredData);

  decomposition.Apply(data, centeredData, transformedData, eigVal, eigvec,
      data.n_rows);

  Timer::Stop("pca");
}

/**
 * Apply Principal Component Analysis to the provided data set.
 *
 * @param data - Data matrix
 * @param transformedData - Data with PCA applied
 * @param eigVal - contains eigen values in a column vector
 */
template<typename DecompositionPolicy>
void PCAType<DecompositionPolicy>::Apply(const arma::mat& data,
                                         arma::mat& transformedData,
                                         arma::vec& eigVal)
{
  arma::mat eigvec;
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
double PCAType<DecompositionPolicy>::Apply(arma::mat& data,
                                           const size_t newDimension)
{
  // Parameter validation.
  if (newDimension == 0)
    Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
        << "be zero!" << endl;
  if (newDimension > data.n_rows)
    Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
        << "be greater than the existing dimensionality of the data ("
        << data.n_rows << ")!" << endl;

  arma::mat eigvec;
  arma::vec eigVal;

  Timer::Start("pca");

  // Center the data into a temporary matrix.
  arma::mat centeredData;
  math::Center(data, centeredData);

  // Scale the data if the user ask for.
  ScaleData(centeredData);

  decomposition.Apply(data, centeredData, data, eigVal, eigvec, newDimension);

  if (newDimension < eigvec.n_rows)
    // Drop unnecessary rows.
    data.shed_rows(newDimension, data.n_rows - 1);

  // The svd method returns only non-zero eigenvalues so we have to calculate
  // the right dimension before calculating the amount of variance retained.
  double eigDim = std::min(newDimension - 1, (size_t) eigVal.n_elem - 1);

  Timer::Stop("pca");

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
double PCAType<DecompositionPolicy>::Apply(arma::mat& data,
                                           const double varRetained)
{
  // Parameter validation.
  if (varRetained < 0)
    Log::Fatal << "PCA::Apply(): varRetained (" << varRetained << ") must be "
        << "greater than or equal to 0." << endl;
  if (varRetained > 1)
    Log::Fatal << "PCA::Apply(): varRetained (" << varRetained << ") should be "
        << "less than or equal to 1." << endl;

  arma::mat eigvec;
  arma::vec eigVal;

  Apply(data, data, eigVal, eigvec);

  // Calculate the dimension we should keep.
  size_t newDimension = 0;
  double varSum = 0.0;
  eigVal /= arma::sum(eigVal); // Normalize eigenvalues.
  while ((varSum < varRetained) && (newDimension < eigVal.n_elem))
  {
    varSum += eigVal[newDimension];
    ++newDimension;
  }

  // varSum is the actual variance we will retain.
  if (newDimension < eigVal.n_elem)
    data.shed_rows(newDimension, data.n_rows - 1);

  return varSum;
}

} // namespace pca
} // namespace mlpack

#endif
