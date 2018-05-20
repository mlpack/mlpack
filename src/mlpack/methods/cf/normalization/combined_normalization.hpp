/**
 * @file combined_normalization.hpp
 * @author Wenhao Huang
 *
 * CombinedNormalization is a class template for performing a sequence of data
 * normalization methods which are specified by template parameters.
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
namespace cf {

/**
 * This normalization class performs a sequence of normalization methods on
 * raw ratings.
 */
template<typename ...NormalizationTypes>
class CombinedNormalization
{
 public:
  using TupleType = std::tuple<NormalizationTypes...>;

  // Empty constructor.
  CombinedNormalization() { }

  /**
   * Normalize the data.
   *
   * @param data Input dataset in the form of coordinate list.
   */
  void Normalize(arma::mat& data)
  {
    SequenceNormalize<0>(data);
  }

  /**
   * Normalize the data by subtracting the mean of all existing ratings.
   *
   * @param cleanedData Sparse matrix data.
   */
  void Normalize(arma::sp_mat& cleanedData)
  {
    SequenceNormalize<0>(cleanedData);
  }

  /**
   * Denormalize computed rating by adding mean.
   *
   * @param user User ID.
   * @param item Item ID.
   * @param rating Computed rating before denormalization.
   */
  double Denormalize(const int user,
                     const int item,
                     const double rating) const
  {
    return SequenceDenormalize<0>(user, item, rating);
  }

  /**
   * Denormalize computed rating by adding mean.
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
   * Return normalizations.
   */
  TupleType Normalizations() const
  {
    return normalizations;
  }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    // Boost does not support tuple serialization???
    ar & BOOST_SERIALIZATION_NVP(normalizations);
  }

 private:

  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  void SequenceNormalize(arma::mat& data)
  {
    std::get<I>(normalizations).Normalize(data);
    SequenceNormalize<I+1>(data);
  }

  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  void SequenceNormalize(arma::mat& /* data */) { }

  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  void SequenceNormalize(arma::sp_mat& cleanedData)
  {
    std::get<I>(normalizations).Normalize(cleanedData);
    SequenceNormalize<I+1>(cleanedData);
  }

  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  void SequenceNormalize(arma::sp_mat& /* cleanedData */) { }

  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  double SequenceDenormalize(const int user,
                             const int item,
                             const double rating) const
  {
    // The order of denormalization should be the reversed order 
    // of normalization.
    double realRating = SequenceDenormalize<I+1>(user, item, rating);
    realRating = std::get<I>(normalizations).Denormalize(user, item, realRating);
    return realRating;
  }

  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  double SequenceDenormalize(const int /* user */,
                             const int /* item */,
                             const double rating) const
  {
    return rating;
  }

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

  template<
      int I, /* Which normalization in tuple to use */
      typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
      typename = void>
  void SequenceDenormalize(const arma::Mat<size_t>& /* combinations */,
                             arma::vec& /* predictions */) const { }

  //! A tuple of all normalizations.
  TupleType normalizations;
};

} // namespace cf
} // namespace mlpack

#endif
