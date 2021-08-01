
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_bag_of_words_encoding_policy.hpp:

Program Listing for File bag_of_words_encoding_policy.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_bag_of_words_encoding_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/string_encoding_policies/bag_of_words_encoding_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_STR_ENCODING_POLICIES_BAG_OF_WORDS_ENCODING_POLICY_HPP
   #define MLPACK_CORE_DATA_STR_ENCODING_POLICIES_BAG_OF_WORDS_ENCODING_POLICY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
   #include <mlpack/core/data/string_encoding.hpp>
   
   namespace mlpack {
   namespace data {
   
   class BagOfWordsEncodingPolicy
   {
    public:
     static void Reset()
     {
       // Nothing to do.
     }
   
     template<typename MatType>
     static void InitMatrix(MatType& output,
                            const size_t datasetSize,
                            const size_t /* maxNumTokens */,
                            const size_t dictionarySize)
     {
       output.zeros(dictionarySize, datasetSize);
     }
   
     template<typename ElemType>
     static void InitMatrix(std::vector<std::vector<ElemType>>& output,
                            const size_t datasetSize,
                            const size_t /* maxNumTokens */,
                            const size_t dictionarySize)
     {
       output.resize(datasetSize, std::vector<ElemType>(dictionarySize));
     }
   
     template<typename MatType>
     static void Encode(MatType& output,
                        const size_t value,
                        const size_t line,
                        const size_t /* index */)
     {
       // The labels are assigned sequentially starting from one.
       output(value - 1, line) += 1;
     }
   
     template<typename ElemType>
     static void Encode(std::vector<std::vector<ElemType>>& output,
                        const size_t value,
                        const size_t line,
                        const size_t /* index */)
     {
       // The labels are assigned sequentially starting from one.
       output[line][value - 1] += 1;
     }
   
     static void PreprocessToken(size_t /* line */,
                                 size_t /* index */,
                                 size_t /* value */)
     { }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */)
     {
       // Nothing to serialize.
     }
   };
   
   template<typename TokenType>
   using BagOfWordsEncoding = StringEncoding<BagOfWordsEncodingPolicy,
                                             StringEncodingDictionary<TokenType>>;
   } // namespace data
   } // namespace mlpack
   
   #endif
