/**
 * @file z_score_normalization.hpp
 * @author Wenhao Huang
 *
 * This class performs z-score normalization on raw ratings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_NORMALIZATION_Z_SCORE_NORMALIZATION_HPP
#define MLPACK_METHODS_CF_NORMALIZATION_Z_SCORE_NORMALIZATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cf {

/**
 * This normalization class performs z-score normalization on raw ratings.
 */
class ZScoreNormalization
{
 public:
  // Empty constructor.
  ZScoreNormalization() { }

  /**
   * Normalize the data to zero mean and one standard deviation.
   *
   * @param data Input dataset in the form of coordinate list.
   */
  void Normalize(arma::mat& data)
  {
    mean = arma::mean(data.row(2));
    stddev = arma::stddev(data.row(2));
    data.row(2) = (data.row(2) - mean) / stddev;
  }

  /**
   * Normalize the data to zero mean and one standard deviation.
   *
   * @param cleanedData Sparse matrix data.
   */
  void Normalize(arma::sp_mat& cleanedData)
  {
    // Caculate mean and stdev of all non zero ratings.
    arma::vec ratings = arma::nonzeros(cleanedData);
    mean = arma::mean(ratings);
    stddev = arma::stddev(ratings);

    // Subtract mean from existing rating and divide it by stddev.
    arma::sp_mat::iterator it = cleanedData.begin();
    arma::sp_mat::iterator it_end = cleanedData.end();
    for (; it != it_end; it++)
      *it = (*it - mean) / stddev;
  }

  /**
   * Denormalize computed rating by adding mean and multiplying stddev.
   *
   * @param user User ID.
   * @param item Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const int /* user */,
                     const int /* item */,
                     const double rating) const
  {
    return (rating + mean) * stddev;
  }

  /**
   * Denormalize computed rating by adding mean and multiplying stddev.
   *
   * @param combinations User/Item combinations.
   * @param predictions Predicted ratings for each user/item combination.
   */
  void Denormalize(const arma::Mat<size_t>& /* combinations */,
                   arma::vec& predictions) const
  {
    predictions = (predictions + mean) * stddev;
  }

  /**
   * Return mean.
   */
  double Mean() const
  {
    return mean;
  }

  /**
   * Return stddev.
   */
  double Stddev() const
  {
    return stddev;
  }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(stddev);
  }

 private:
  //! Mean of all existing ratings.
  double mean;
  //! Standard deviation of all existing ratings.
  double stddev;
};

} // namespace cf
} // namespace mlpack

#endif
