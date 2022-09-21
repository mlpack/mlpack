/**
 * @file methods/cf/normalization/combined_normalization.hpp
 * @author Wenhao Huang
 *
 * CombinedNormalization is a class template for performing a sequence of data
 * normalization methods which are specified by template parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_NORMALIZATION_COMBINED_NORMALIZATION_HPP
#define MLPACK_METHODS_CF_NORMALIZATION_COMBINED_NORMALIZATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This normalization class performs a sequence of normalization methods on
 * raw ratings.
 *
 * An example of how to use CombinedNormalization in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * CFType<NMFPolicy,
 *        CombinedNormalization<
 *            OverallMeanNormalization,
 *            UserMeanNormalization,
 *            ItemMeanNormalization>> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 * @endcode
 */
template<typename... NormalizationTypes>
class CombinedNormalization
{
 public:
  using TupleType = std::tuple<NormalizationTypes...>;

  // Empty constructor.
  CombinedNormalization() { }

  /**
   * Normalize the data by calling Normalize() in each normalization object.
   *
   * @param data Input dataset.
   */
  template<typename MatType>
  void Normalize(MatType& data)
  {
    SequenceNormalize<0>(data);
  }

  /**
   * Denormalize rating by calling Denormalize() in each normalization object.
   * Note that the order of objects calling Denormalize() should be the
   * reversed order of objects calling Normalize().
   *
   * @param user User ID.
   * @param item Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const size_t user,
                     const size_t item,
                     const double rating) const
  {
    return SequenceDenormalize<0>(user, item, rating);
  }

  /**
   * Denormalize rating by calling Denormalize() in each normalization object.
   * Note that the order of objects calling Denormalize() should be the
   * reversed order of objects calling Normalize().
   *
   * @param combinations User/Item combinations.
   * @param predictions Predicted ratings for each user/item combination.
   */
  void Denormalize(const arma::Mat<size_t>& combinations,
                   arma::vec& predictions) const
  {
    SequenceDenormalize<0>(combinations, predictions);
  }

  /**
   * Return normalizations tuple.
   */
  const TupleType& Normalizations() const
  {
    return normalizations;
  }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version)
  {
    SequenceSerialize<0, Archive>(ar, version);
  }

 private:
  //! A tuple of all normalization objects.
  TupleType normalizations;

  //! Unpack normalizations tuple to normalize data.
  template<
      int I, /* Which normalization in tuple to use */
      typename MatType,
      typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  void SequenceNormalize(MatType& data)
  {
    std::get<I>(normalizations).Normalize(data);
    SequenceNormalize<I + 1>(data);
  }

  //! End of tuple unpacking.
  template<
      int I, /* Which normalization in tuple to use */
      typename MatType,
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  void SequenceNormalize(MatType& /* data */) { }

  //! Unpack normalizations tuple to denormalize.
  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  double SequenceDenormalize(const size_t user,
                             const size_t item,
                             const double rating) const
  {
    // The order of denormalization should be the reversed order
    // of normalization.
    double realRating = SequenceDenormalize<I + 1>(user, item, rating);
    realRating =
        std::get<I>(normalizations).Denormalize(user, item, realRating);
    return realRating;
  }

  //! End of tuple unpacking.
  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  double SequenceDenormalize(const size_t /* user */,
                             const size_t /* item */,
                             const double rating) const
  {
    return rating;
  }

  //! Unpack normalizations tuple to denormalize.
  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  void SequenceDenormalize(const arma::Mat<size_t>& combinations,
                           arma::vec& predictions) const
  {
    // The order of denormalization should be the reversed order
    // of normalization.
    SequenceDenormalize<I+1>(combinations, predictions);
    std::get<I>(normalizations).Denormalize(combinations, predictions);
  }

  //! End of tuple unpacking.
  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  void SequenceDenormalize(const arma::Mat<size_t>& /* combinations */,
                           arma::vec& /* predictions */) const { }

  //! Unpack normalizations tuple to serialize.
  template<
      int I, /* Which normalization in tuple to serialize */
      typename Archive,
      typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  void SequenceSerialize(Archive& ar, const uint32_t version)
  {
    std::string tagName = "normalization_";
    tagName += std::to_string(I);
    ar(cereal::make_nvp(
        tagName.c_str(), std::get<I>(normalizations)));
    SequenceSerialize<I + 1, Archive>(ar, version);
  }

  //! End of tuple unpacking.
  template<
      int I, /* Which normalization in tuple to serialize */
      typename Archive,
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  void SequenceSerialize(Archive& /* ar */, const uint32_t /* version */)
  { }
};

} // namespace mlpack

#endif
