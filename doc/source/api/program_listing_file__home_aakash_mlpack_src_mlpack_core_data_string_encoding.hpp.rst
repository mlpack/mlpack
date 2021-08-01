
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_string_encoding.hpp:

Program Listing for File string_encoding.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_string_encoding.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/string_encoding.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_STRING_ENCODING_HPP
   #define MLPACK_CORE_DATA_STRING_ENCODING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
   #include <mlpack/core/data/string_encoding_dictionary.hpp>
   #include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
   #include <vector>
   
   namespace mlpack {
   namespace data {
   
   template<typename EncodingPolicyType,
            typename DictionaryType>
   class StringEncoding
   {
    public:
     template<typename ... ArgTypes>
     StringEncoding(ArgTypes&& ... args);
   
     StringEncoding(EncodingPolicyType encodingPolicy);
   
     StringEncoding(StringEncoding&);
   
     StringEncoding(const StringEncoding&);
   
     StringEncoding& operator=(const StringEncoding&) = default;
   
     StringEncoding(StringEncoding&&);
   
     StringEncoding& operator=(StringEncoding&&) = default;
   
     template<typename TokenizerType>
     void CreateMap(const std::string& input,
                    const TokenizerType& tokenizer);
   
     void Clear();
   
     template<typename OutputType, typename TokenizerType>
     void Encode(const std::vector<std::string>& input,
                 OutputType& output,
                 const TokenizerType& tokenizer);
   
     const DictionaryType& Dictionary() const { return dictionary; }
     DictionaryType& Dictionary() { return dictionary; }
   
     const EncodingPolicyType& EncodingPolicy() const { return encodingPolicy; }
     EncodingPolicyType& EncodingPolicy() { return encodingPolicy; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     template<typename OutputType, typename TokenizerType, typename PolicyType>
     void EncodeHelper(const std::vector<std::string>& input,
                       OutputType& output,
                       const TokenizerType& tokenizer,
                       PolicyType& policy);
   
     template<typename TokenizerType, typename PolicyType, typename ElemType>
     void EncodeHelper(const std::vector<std::string>& input,
                       std::vector<std::vector<ElemType>>& output,
                       const TokenizerType& tokenizer,
                       PolicyType& policy,
                       typename std::enable_if<StringEncodingPolicyTraits<
                           PolicyType>::onePassEncoding>::type* = 0);
   
    private:
     EncodingPolicyType encodingPolicy;
     DictionaryType dictionary;
   };
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation.
   #include "string_encoding_impl.hpp"
   
   #endif
