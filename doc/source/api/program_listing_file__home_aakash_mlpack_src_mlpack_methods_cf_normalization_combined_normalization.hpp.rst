
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_combined_normalization.hpp:

Program Listing for File combined_normalization.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_combined_normalization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/normalization/combined_normalization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_NORMALIZATION_COMBINED_NORMALIZATION_HPP
   #define MLPACK_METHODS_CF_NORMALIZATION_COMBINED_NORMALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cf {
   
   template<typename... NormalizationTypes>
   class CombinedNormalization
   {
    public:
     using TupleType = std::tuple<NormalizationTypes...>;
   
     // Empty constructor.
     CombinedNormalization() { }
   
     template<typename MatType>
     void Normalize(MatType& data)
     {
       SequenceNormalize<0>(data);
     }
   
     double Denormalize(const size_t user,
                        const size_t item,
                        const double rating) const
     {
       return SequenceDenormalize<0>(user, item, rating);
     }
   
     void Denormalize(const arma::Mat<size_t>& combinations,
                      arma::vec& predictions) const
     {
       SequenceDenormalize<0>(combinations, predictions);
     }
   
     const TupleType& Normalizations() const
     {
       return normalizations;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version)
     {
       SequenceSerialize<0, Archive>(ar, version);
     }
   
    private:
     TupleType normalizations;
   
     template<
         int I, /* Which normalization in tuple to use */
         typename MatType,
         typename = std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
     void SequenceNormalize(MatType& data)
     {
       std::get<I>(normalizations).Normalize(data);
       SequenceNormalize<I + 1>(data);
     }
   
     template<
         int I, /* Which normalization in tuple to use */
         typename MatType,
         typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
         typename = void>
     void SequenceNormalize(MatType& /* data */) { }
   
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
   
     template<
         int I, /* Which normalization in tuple to serialize */
         typename Archive,
         typename = std::enable_if_t<(I >= std::tuple_size<TupleType>::value)>,
         typename = void>
     void SequenceSerialize(Archive& /* ar */, const uint32_t /* version */)
     { }
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
