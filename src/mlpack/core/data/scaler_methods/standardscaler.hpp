/**
 * @file Standardscaler.hpp
 * @author Jeffin Sam
 *
 * StandardScaler class to scale features.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STANDARD_SCALE_HPP
#define MLPACK_CORE_DATA_STANDARD_SCALE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {
/**
 * A simple Standard Scaler class
 */
template <typename T>
class StandardScaler
{
 public:
  /**
  * Default constructor
  *
  */
  StandardScaler(){}
  /**
  * Default Destructor 
  */
  ~StandardScaler(){}

  /**
  * Function to scale Features
  *
  * @param input Datset to scale features
  */
  void Transform(arma::Mat<T>& input)
  {
    itemMean = arma::mean(input, 1);
    itemStdev = arma::stddev(input, 1, 1);

    // Handline Zeroes in scale vector
    for (size_t i = 0; i < itemStdev.n_elem; i++)
    {
      if (itemStdev(i) == 0)
      {
        itemStdev(i) = 1;
      }
    }
    for (size_t i = 0; i < input.n_rows; i++)
    {
      for (size_t j = 0; j < input.n_cols; j++)
      {
        input(i, j) -= itemMean(i);
        input(i, j) /= itemStdev(i);
      }
    }
  }

  /**
  * Function to retrive original dataset
  *
  * @param input Scaled dataset
  */
  void InverseTransform(arma::Mat<T>& input)
  {
    for (size_t i = 0; i < input.n_rows; i++)
    {
      for (size_t j = 0; j < input.n_cols; j++)
      {
        input(i, j) *= itemStdev(i);
        input(i, j) += itemMean(i);
      }
    }
  }
  //! Get the Min row vector.
  const arma::colvec& ItemMean() const { return itemMean; }
  //! Get the Max row vector.
  const arma::colvec& ItemStdev() const { return itemStdev; }
 private:
  // Min row vector which holds minimum of each feature
  arma::colvec itemMean;
  // Max row vector which holds maximum of each feature
  arma::colvec itemStdev;
}; // class StandardScaler

} // namespace ann
} // namespace mlpack

#endif
