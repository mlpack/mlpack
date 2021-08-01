
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_dictionary_encoding_policy.hpp:

Program Listing for File dictionary_encoding_policy.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_dictionary_encoding_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/string_encoding_policies/dictionary_encoding_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_DICTIONARY_ENCODING_POLICY_HPP
   #define MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_DICTIONARY_ENCODING_POLICY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
   #include <mlpack/core/data/string_encoding.hpp>
   
   namespace mlpack {
   namespace data {
   
   class DictionaryEncodingPolicy
   {
    public:
     static void Reset()
     {
       // Nothing to do.
     }
   
     template<typename MatType>
     static void InitMatrix(MatType& output,
                            const size_t datasetSize,
                            const size_t maxNumTokens,
                            const size_t /* dictionarySize */)
     {
       output.zeros(maxNumTokens, datasetSize);
     }
   
     template<typename MatType>
     static void Encode(MatType& output,
                        const size_t value,
                        const size_t line,
                        const size_t index)
     {
       output(index, line) = value;
     }
   
     template<typename ElemType>
     static void Encode(std::vector<ElemType>& output, size_t value)
     {
       output.push_back(value);
     }
   
     static void PreprocessToken(const size_t /* line */,
                                 const size_t /* index */,
                                 const size_t /* value */)
     { }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */)
     {
       // Nothing to serialize.
     }
   };
   
   template<>
   struct StringEncodingPolicyTraits<DictionaryEncodingPolicy>
   {
     static const bool onePassEncoding = true;
   };
   
   template<typename TokenType>
   using DictionaryEncoding = StringEncoding<DictionaryEncodingPolicy,
                                             StringEncodingDictionary<TokenType>>;
   } // namespace data
   } // namespace mlpack
   
   #endif
