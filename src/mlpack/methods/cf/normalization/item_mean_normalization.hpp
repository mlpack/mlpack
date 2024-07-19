/**
 * @file methods/cf/normalization/item_mean_normalization.hpp
 * @author Wenhao Huang
 *
 * This class performs item mean normalization on raw ratings. In another
 * word, this class is used to remove global effect of item mean.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_NORMALIZATION_ITEM_MEAN_NORMALIZATION_HPP
#define MLPACK_METHODS_CF_NORMALIZATION_ITEM_MEAN_NORMALIZATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This normalization class performs item mean normalization on raw ratings.
 *
 * An example of how to use ItemMeanNormalization in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * // Use ItemMeanNormalization as normalization method.
 * CFType<NMFPolicy, ItemMeanNormalization> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 * @endcode
 */
class ItemMeanNormalization
{
 public:
  // Empty constructor.
  ItemMeanNormalization() { }

  /**
   * Normalize the data by subtracting item mean from each of existing ratings.
   *
   * @param data Input dataset in the form of coordinate list.
   */
  void Normalize(arma::mat& data)
  {
    const size_t itemNum = max(data.row(1)) + 1;
    itemMean = arma::vec(itemNum);
    // Number of ratings for each item.
    arma::Row<size_t> ratingNum(itemNum);

    // Sum ratings for each item.
    data.each_col([&](arma::vec& datapoint)
    {
      const size_t item = (size_t) datapoint(1);
      const double rating = datapoint(2);
      itemMean(item) += rating;
      ratingNum(item) += 1;
    });

    // Calculate item mean and subtract item mean from ratings.
    // Set item mean to 0 if the item has no rating.
    for (size_t i = 0; i < itemNum; ++i)
    {
      if (ratingNum(i) != 0)
        itemMean(i) /= ratingNum(i);
    }

    data.each_col([&](arma::vec& datapoint)
    {
      const size_t item = (size_t) datapoint(1);
      datapoint(2) -= itemMean(item);
      // The algorithm omits rating of zero. If normalized rating equals zero,
      // it is set to the smallest positive float value.
      if (datapoint(2) == 0)
        datapoint(2) = std::numeric_limits<float>::min();
    });
  }

  /**
   * Normalize the data by subtracting item mean from each of existing ratings.
   *
   * @param cleanedData Input data as a sparse matrix.
   */
  void Normalize(arma::sp_mat& cleanedData)
  {
    // Calculate itemMean.
    itemMean = arma::vec(cleanedData.n_rows);
    arma::Col<size_t> ratingNum(cleanedData.n_rows);
    arma::sp_mat::iterator it = cleanedData.begin();
    arma::sp_mat::iterator it_end = cleanedData.end();
    for (; it != it_end; ++it)
    {
      itemMean(it.row()) += *it;
      ratingNum(it.row()) += 1;
    }
    for (size_t i = 0; i < itemMean.n_elem; ++i)
    {
      if (ratingNum(i) != 0)
        itemMean(i) /= ratingNum(i);
    }

    // Normalize the data.
    it = cleanedData.begin();
    for (; it != cleanedData.end(); ++it)
    {
      double tmp = *it - itemMean(it.row());

      // The algorithm omits rating of zero. If normalized rating equals zero,
      // it is set to the smallest positive double value.
      if (tmp == 0)
        tmp = std::numeric_limits<float>::min();

      *it = tmp;
    }
  }

  /**
   * Denormalize computed rating by adding item mean.
   *
   * @param * (user) User ID.
   * @param item Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const size_t /* user */,
                     const size_t item,
                     const double rating) const
  {
    return rating + itemMean(item);
  }

  /**
   * Denormalize computed rating by adding item mean.
   *
   * @param combinations User/Item combinations.
   * @param predictions Predicted ratings for each user/item combination.
   */
  void Denormalize(const arma::Mat<size_t>& combinations,
                   arma::vec& predictions) const
  {
    for (size_t i = 0; i < predictions.n_elem; ++i)
    {
      const size_t item = combinations(1, i);
      predictions(i) += itemMean(item);
    }
  }

  /**
   * Return item mean.
   */
  const arma::vec& Mean() const { return itemMean; }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(itemMean));
  }

 private:
  //! Item mean.
  arma::vec itemMean;
};

} // namespace mlpack

#endif
