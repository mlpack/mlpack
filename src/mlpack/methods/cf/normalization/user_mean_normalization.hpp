/**
 * @file user_mean_normalization.hpp
 * @author Wenhao Huang
 *
 * This class performs user mean normalization on raw ratings. In another
 * word, this class is used to remove global effect of user mean.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_NORMALIZATION_USER_MEAN_NORMALIZATION_HPP
#define MLPACK_METHODS_CF_NORMALIZATION_USER_MEAN_NORMALIZATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cf {

/**
 * This normalization class performs user mean normalization on raw ratings.
 */
class UserMeanNormalization
{
 public:
  // Empty constructor.
  UserMeanNormalization() { }

  /**
   * Normalize the data by subtracting user mean from each of existing ratings.
   *
   * @param data Input dataset in the form of coordinate list.
   */
  void Normalize(arma::mat& data)
  {
    const size_t userNum = arma::max(data.row(0)) + 1;
    userMean = arma::vec(userNum, arma::fill::zeros);
    // Number of ratings for each user.
    arma::Row<size_t> ratingNum(userNum, arma::fill::zeros);

    // Sum ratings for each user.
    data.each_col([&](arma::vec& datapoint) {
      const size_t user = (size_t) datapoint(0);
      const double rating = datapoint(2);
      userMean(user) += rating;
      ratingNum(user) += 1;
    });

    // Calculate user mean and subtract user mean from ratings.
    // Set user mean to 0 if the user has no rating.
    // Should we use mean of all user means if a user has no rating?
    for (size_t i = 0; i < userNum; i++)
      if (ratingNum(i) != 0)
        userMean(i) /= ratingNum(i);
    data.each_col([&](arma::vec& datapoint) {
      const size_t user = (size_t) datapoint(0);
      datapoint(2) -= userMean(user);
    });
  }

  /**
   * Normalize the data by subtracting user mean from each of existing rating.
   *
   * @param cleanedData Sparse matrix data.
   */
  void Normalize(arma::sp_mat& cleanedData)
  {
    userMean = arma::mean(cleanedData, 0);

    arma::sp_mat::iterator it = cleanedData.begin();
    arma::sp_mat::iterator it_end = cleanedData.end();
    for (; it != it_end; it++)
      *it = *it - userMean(it.col());
  }

  /**
   * Denormalize computed rating by adding user mean.
   *
   * @param user User ID.
   * @param item Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const int user,
                     const int /* item */,
                     const double rating) const
  {
    return rating + userMean(user);
  }

  /**
   * Denormalize computed rating by adding user mean.
   *
   * @param combinations User/Item combinations.
   * @param predictions Predicted ratings for each user/item combination.
   */
  void Denormalize(const arma::Mat<size_t>& combinations,
                   arma::vec& predictions) const
  {
    for (size_t i = 0; i < predictions.n_elem; i++)
    {
      const size_t user = combinations(0, i);
      predictions(i) += userMean(user);
    }
  }

  /**
   * Return user mean.
   */
  arma::vec UserMean() const
  {
    return userMean;
  }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(userMean);
  }

 private:
  //! User mean.
  arma::vec userMean;
};

} // namespace cf
} // namespace mlpack

#endif
