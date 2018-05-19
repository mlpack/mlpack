/**
 * @file overall_mean_normalization.hpp
 * @author Wenhao Huang
 *
 * This class performs overall mean normalization on raw ratings. In another
 * word, this class is used to remove global effect of overall mean.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_NORMALIZATION_OVERALL_MEAN_NORMALIZATION_HPP
#define MLPACK_METHODS_CF_NORMALIZATION_OVERALL_MEAN_NORMALIZATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cf {

/**
 * This normalization class performs overall mean normalization on raw ratings.
 */
class OverallMeanNormalization
{
 public:
  // Empty constructor.
  OverallMeanNormalization() { }

  /**
   * Normalize the data by subtracting the mean of all existing ratings.
   *
   * @param data Input dataset in the form of coordinate list.
   */
  void Normalize(arma::mat& data)
  {
    mean = arma::mean(data.row(2));
    data.row(2) -= mean;
  }

  /**
   * Normalize the data by subtracting the mean of all existing ratings.
   *
   * @param cleanedData Sparse matrix data.
   */
  void Normalize(arma::sp_mat& cleanedData)
  {
    // Caculate mean of all non zero ratings.
    mean = arma::accu(cleanedData) / cleanedData.n_nonzero;
    // Subtract mean from all non zero ratings.
    arma::sp_mat::iterator it = cleanedData.begin();
    arma::sp_mat::iterator it_end = cleanedData.end();
    for (; it != it_end; it++)
      *it = *it - mean;
  }

  /**
   * Denormalize computed rating by adding mean.
   *
   * @param user User ID.
   * @param item Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const int /* user */,
                     const int /* item */,
                     const double rating) const
  {
    return rating + mean;
  }

  /**
   * Denormalize computed rating by adding mean.
   *
   * @param combinations User/Item combinations.
   * @param predictions Predicted ratings for each user/item combination.
   */
  void Denormalize(const arma::Mat<size_t>& /* combinations */,
                   arma::vec& predictions) const
  {
    predictions += mean;
  }

  /**
   * Return mean.
   */
  double Mean() const
  {
    return mean;
  }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(mean);
  }

 private:
  //! Mean of all existing ratings.
  double mean;
};

} // namespace cf
} // namespace mlpack

#endif
