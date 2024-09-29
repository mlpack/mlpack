/**
 * @file methods/cf/normalization/user_mean_normalization.hpp
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

/**
 * This normalization class performs user mean normalization on raw ratings.
 *
 * An example of how to use UserMeanNormalization in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * // Use UserMeanNormalization as normalization method.
 * CFType<NMFPolicy, UserMeanNormalization> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 * @endcode
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
    const size_t userNum = max(data.row(0)) + 1;
    userMean = arma::vec(userNum);
    // Number of ratings for each user.
    arma::Row<size_t> ratingNum(userNum);

    // Sum ratings for each user.
    data.each_col([&](arma::vec& datapoint)
    {
      const size_t user = (size_t) datapoint(0);
      const double rating = datapoint(2);
      userMean(user) += rating;
      ratingNum(user) += 1;
    });

    // Calculate user mean and subtract user mean from ratings.
    // Set user mean to 0 if the user has no rating.
    for (size_t i = 0; i < userNum; ++i)
    {
      if (ratingNum(i) != 0)
        userMean(i) /= ratingNum(i);
    }

    data.each_col([&](arma::vec& datapoint)
    {
      const size_t user = (size_t) datapoint(0);
      datapoint(2) -= userMean(user);
      // The algorithm omits rating of zero. If normalized rating equals zero,
      // it is set to the smallest positive double value.
      if (datapoint(2) == 0)
        datapoint(2) = std::numeric_limits<double>::min();
    });
  }

  /**
   * Normalize the data by subtracting user mean from each of existing rating.
   *
   * @param cleanedData Input data as a sparse matrix.
   */
  void Normalize(arma::sp_mat& cleanedData)
  {
    // Calculate userMean.
    userMean = arma::vec(cleanedData.n_cols);
    arma::Col<size_t> ratingNum(cleanedData.n_cols);
    arma::sp_mat::iterator it = cleanedData.begin();
    arma::sp_mat::iterator it_end = cleanedData.end();
    for (; it != it_end; ++it)
    {
      userMean(it.col()) += *it;
      ratingNum(it.col()) += 1;
    }
    for (size_t i = 0; i < userMean.n_elem; ++i)
    {
      if (ratingNum(i) != 0)
        userMean(i) /= ratingNum(i);
    }

    // Normalize the data.
    it = cleanedData.begin();
    for (; it != cleanedData.end(); ++it)
    {
      double tmp = *it - userMean(it.col());

      // The algorithm omits rating of zero. If normalized rating equals zero,
      // it is set to the smallest positive float value.
      if (tmp == 0)
        tmp = std::numeric_limits<float>::min();

      *it = tmp;
    }
  }

  /**
   * Denormalize computed rating by adding user mean.
   *
   * @param user User ID.
   * @param * (item) Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const size_t user,
                     const size_t /* item */,
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
    for (size_t i = 0; i < predictions.n_elem; ++i)
    {
      const size_t user = combinations(0, i);
      predictions(i) += userMean(user);
    }
  }

  /**
   * Return user mean.
   */
  const arma::vec& Mean() const { return userMean; }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(userMean));
  }

 private:
  //! User mean.
  arma::vec userMean;
};

} // namespace mlpack

#endif
