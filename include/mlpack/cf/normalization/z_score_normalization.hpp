/**
 * @file methods/cf/normalization/z_score_normalization.hpp
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

/**
 * This normalization class performs z-score normalization on raw ratings.
 *
 * An example of how to use ZScoreNormalization in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * // Use ZScoreNormalization as normalization method.
 * CFType<NMFPolicy, ZScoreNormalization> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 * @endcode
 */
class ZScoreNormalization
{
 public:
  // Empty constructor.
  ZScoreNormalization() : mean(0), stddev(1) { }

  /**
   * Normalize the data to zero mean and one standard deviation.
   *
   * @param data Input dataset in the form of coordinate list.
   */
  void Normalize(arma::mat& data)
  {
    mean = arma::mean(data.row(2));
    stddev = arma::stddev(data.row(2));

    if (std::fabs(stddev) < 1e-14)
    {
      Log::Fatal << "Standard deviation of all existing ratings is 0! "
          << "This may indicate that all existing ratings are the same."
          << std::endl;
    }

    data.row(2) = (data.row(2) - mean) / stddev;
    // The algorithm omits rating of zero. If normalized rating equals zero,
    // it is set to the smallest positive float value.
    data.row(2).for_each([](double& x)
    {
      if (x == 0)
        x = std::numeric_limits<float>::min();
    });
  }

  /**
   * Normalize the data to zero mean and one standard deviation.
   *
   * @param cleanedData Input data as a sparse matrix.
   */
  void Normalize(arma::sp_mat& cleanedData)
  {
    // Caculate mean and stdev of all non zero ratings.
    arma::vec ratings = arma::nonzeros(cleanedData);
    mean = arma::mean(ratings);
    stddev = arma::stddev(ratings);

    if (std::fabs(stddev) < 1e-14)
    {
      Log::Fatal << "Standard deviation of all existing ratings is 0! "
          << "This may indicate that all existing ratings are the same."
          << std::endl;
    }

    // Subtract mean from existing rating and divide it by stddev.
    // TODO: consider using spmat::transform() instead of spmat iterators
    // TODO: http://arma.sourceforge.net/docs.html#transform
    arma::sp_mat::iterator it = cleanedData.begin();
    arma::sp_mat::iterator it_end = cleanedData.end();
    for (; it != it_end; ++it)
    {
      double tmp = (*it - mean) / stddev;

      // The algorithm omits rating of zero. If normalized rating equals zero,
      // it is set to the smallest positive float value.
      if (tmp == 0)
        tmp = std::numeric_limits<float>::min();

      *it = tmp;
    }
  }

  /**
   * Denormalize computed rating by adding mean and multiplying stddev.
   *
   * @param * (user) User ID.
   * @param * (item) Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const size_t /* user */,
                     const size_t /* item */,
                     const double rating) const
  {
    return rating * stddev + mean;
  }

  /**
   * Denormalize computed rating by adding mean and multiplying stddev.
   *
   * @param * (combinations) User/Item combinations.
   * @param predictions Predicted ratings for each user/item combination.
   */
  void Denormalize(const arma::Mat<size_t>& /* combinations */,
                   arma::vec& predictions) const
  {
    predictions = predictions * stddev + mean;
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
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mean));
    ar(CEREAL_NVP(stddev));
  }

 private:
  //! Mean of all existing ratings.
  double mean;
  //! Standard deviation of all existing ratings.
  double stddev;
};

} // namespace mlpack

#endif
