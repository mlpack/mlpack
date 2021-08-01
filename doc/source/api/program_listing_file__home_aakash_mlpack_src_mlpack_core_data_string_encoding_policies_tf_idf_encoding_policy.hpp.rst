
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_tf_idf_encoding_policy.hpp:

Program Listing for File tf_idf_encoding_policy.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_policies_tf_idf_encoding_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/string_encoding_policies/tf_idf_encoding_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_TF_IDF_ENCODING_POLICY_HPP
   #define MLPACK_CORE_DATA_STRING_ENCODING_POLICIES_TF_IDF_ENCODING_POLICY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
   #include <mlpack/core/data/string_encoding.hpp>
   
   namespace mlpack {
   namespace data {
   
   class TfIdfEncodingPolicy
   {
    public:
     enum class TfTypes
     {
       BINARY,
       RAW_COUNT,
       TERM_FREQUENCY,
       SUBLINEAR_TF,
     };
   
     TfIdfEncodingPolicy(const TfTypes tfType = TfTypes::RAW_COUNT,
                         const bool smoothIdf = true) :
         tfType(tfType),
         smoothIdf(smoothIdf)
     { }
   
     void Reset()
     {
       tokensFrequences.clear();
       numContainingStrings.clear();
       linesSizes.clear();
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
     void Encode(MatType& output,
                 const size_t value,
                 const size_t line,
                 const size_t /* index */)
     {
       const typename MatType::elem_type tf =
           TermFrequency<typename MatType::elem_type>(
               tokensFrequences[line][value], linesSizes[line]);
   
       const typename MatType::elem_type idf =
           InverseDocumentFrequency<typename MatType::elem_type>(
               output.n_cols, numContainingStrings[value]);
   
       output(value - 1, line) =  tf * idf;
     }
   
     template<typename ElemType>
     void Encode(std::vector<std::vector<ElemType>>& output,
                 const size_t value,
                 const size_t line,
                 const size_t /* index */)
     {
       const ElemType tf = TermFrequency<ElemType>(
           tokensFrequences[line][value], linesSizes[line]);
   
       const ElemType idf = InverseDocumentFrequency<ElemType>(
           output.size(), numContainingStrings[value]);
   
       output[line][value - 1] =  tf * idf;
     }
   
     /*
      * The function calculates the necessary statistics for the purpose
      * of the tf-idf algorithm during the first pass through the dataset.
      *
      * @param line The line number at which the encoding is performed.
      * @param index The token sequence number in the line.
      * @param value The encoded token.
      */
     void PreprocessToken(const size_t line,
                          const size_t /* index */,
                          const size_t value)
     {
       if (line >= tokensFrequences.size())
       {
         linesSizes.resize(line + 1);
         tokensFrequences.resize(line + 1);
       }
   
       tokensFrequences[line][value]++;
   
       if (tokensFrequences[line][value] == 1)
         numContainingStrings[value]++;
   
       linesSizes[line]++;
     }
   
     const std::vector<std::unordered_map<size_t, size_t>>&
         TokensFrequences() const { return tokensFrequences; }
     std::vector<std::unordered_map<size_t, size_t>>& TokensFrequences()
     {
       return tokensFrequences;
     }
   
     const std::unordered_map<size_t, size_t>& NumContainingStrings() const
     {
       return numContainingStrings;
     }
   
     std::unordered_map<size_t, size_t>& NumContainingStrings()
     {
       return numContainingStrings;
     }
   
     const std::vector<size_t>& LinesSizes() const { return linesSizes; }
     std::vector<size_t>& LinesSizes() { return linesSizes; }
   
     TfTypes TfType() const { return tfType; }
     TfTypes& TfType() { return tfType; }
   
     bool SmoothIdf() const { return smoothIdf; }
     bool& SmoothIdf() { return smoothIdf; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(tfType));
       ar(CEREAL_NVP(smoothIdf));
     }
   
    private:
     template<typename ValueType>
     ValueType TermFrequency(const size_t numOccurrences,
                             const size_t numTokens)
     {
       switch (tfType)
       {
       case TfTypes::BINARY:
         return numOccurrences > 0;
       case TfTypes::RAW_COUNT:
         return numOccurrences;
       case TfTypes::TERM_FREQUENCY:
         return static_cast<ValueType>(numOccurrences) / numTokens;
       case TfTypes::SUBLINEAR_TF:
         return std::log(static_cast<ValueType>(numOccurrences)) + 1;
       default:
         Log::Fatal << "Incorrect term frequency type!";
         return 0;
       }
     }
   
     template<typename ValueType>
     ValueType InverseDocumentFrequency(const size_t totalNumLines,
                                        const size_t numOccurrences)
     {
       if (smoothIdf)
       {
         return std::log(static_cast<ValueType>(totalNumLines + 1) /
             (1 + numOccurrences)) + 1.0;
       }
       else
       {
         return std::log(static_cast<ValueType>(totalNumLines) /
             numOccurrences) + 1.0;
       }
     }
   
    private:
     std::vector<std::unordered_map<size_t, size_t>> tokensFrequences;
     std::unordered_map<size_t, size_t> numContainingStrings;
     std::vector<size_t> linesSizes;
     TfTypes tfType;
     bool smoothIdf;
   };
   
   template<typename TokenType>
   using TfIdfEncoding = StringEncoding<TfIdfEncodingPolicy,
                                        StringEncodingDictionary<TokenType>>;
   } // namespace data
   } // namespace mlpack
   
   #endif
