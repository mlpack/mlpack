
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_dictionary.hpp:

Program Listing for File string_encoding_dictionary.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_string_encoding_dictionary.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/string_encoding_dictionary.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_STRING_ENCODING_DICTIONARY_HPP
   #define MLPACK_CORE_DATA_STRING_ENCODING_DICTIONARY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
   #include <unordered_map>
   #include <deque>
   #include <array>
   
   namespace mlpack {
   namespace data {
   
   template<typename Token>
   class StringEncodingDictionary
   {
    public:
     using MapType = std::unordered_map<Token, size_t>;
   
     using TokenType = Token;
   
     bool HasToken(const Token& token) const
     {
       return mapping.find(token) != mapping.end();
     }
   
     template<typename T>
     size_t AddToken(T&& token)
     {
       size_t size = mapping.size();
   
       mapping[std::forward<T>(token)] = ++size;
   
       return size;
     }
   
     size_t Value(const Token& token) const
     {
       return mapping.at(token);
     }
   
     size_t Size() const { return mapping.size(); }
   
     void Clear()
     {
       mapping.clear();
     }
   
     const MapType& Mapping() const { return mapping; }
     MapType& Mapping() { return mapping; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(mapping));
     }
   
    private:
     MapType mapping;
   };
   
   /*
    * Specialization of the StringEncodingDictionary class for boost::string_view.
    */
   template<>
   class StringEncodingDictionary<boost::string_view>
   {
    public:
     using MapType = std::unordered_map<
         boost::string_view,
         size_t,
         boost::hash<boost::string_view>>;
   
     using TokenType = boost::string_view;
   
     StringEncodingDictionary() = default;
   
     StringEncodingDictionary(const StringEncodingDictionary& other) :
         tokens(other.tokens)
     {
       for (const std::string& token : tokens)
         mapping[token] = other.mapping.at(token);
     }
   
     StringEncodingDictionary(StringEncodingDictionary&& other) = default;
   
     StringEncodingDictionary& operator=(const StringEncodingDictionary& other)
     {
       tokens = other.tokens;
       mapping.clear();
   
       for (const std::string& token : tokens)
         mapping[token] = other.mapping.at(token);
   
       return *this;
     }
   
     StringEncodingDictionary& operator=(
         StringEncodingDictionary&& other) = default;
   
     bool HasToken(const boost::string_view token) const
     {
       return mapping.find(token) != mapping.end();
     }
   
     size_t AddToken(const boost::string_view token)
     {
       tokens.emplace_back(token);
   
       size_t size = mapping.size();
   
       mapping[tokens.back()] = ++size;
   
       return size;
     }
   
     size_t Value(const boost::string_view token) const
     {
       return mapping.at(token);
     }
   
     size_t Size() const { return mapping.size(); }
   
     void Clear()
     {
       mapping.clear();
       tokens.clear();
     }
   
     const std::deque<std::string>& Tokens() const { return tokens; }
     std::deque<std::string>& Tokens() { return tokens; }
   
     const MapType& Mapping() const { return mapping; }
     MapType& Mapping() { return mapping; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       size_t numTokens = tokens.size();
   
       ar(CEREAL_NVP(numTokens));
   
       if (cereal::is_loading<Archive>())
       {
         tokens.resize(numTokens);
   
         for (std::string& token : tokens)
         {
           ar(CEREAL_NVP(token));
   
           size_t tokenValue = 0;
           ar(CEREAL_NVP(tokenValue));
           mapping[token] = tokenValue;
         }
       }
       if (cereal::is_saving<Archive>())
       {
         for (std::string& token : tokens)
         {
           ar(CEREAL_NVP(token));
   
           size_t tokenValue = mapping.at(token);
           ar(CEREAL_NVP(tokenValue));
         }
       }
     }
   
    private:
     std::deque<std::string> tokens;
   
     MapType mapping;
   };
   
   template<>
   class StringEncodingDictionary<int>
   {
    public:
     using MapType = std::array<size_t, 1 << CHAR_BIT>;
   
     using TokenType = int;
   
     StringEncodingDictionary() :
       size(0)
     {
       mapping.fill(0);
     }
   
     bool HasToken(const int token) const
     {
       return mapping[token] > 0;
     }
   
     size_t AddToken(const int token)
     {
       mapping[token] = ++size;
   
       return size;
     }
   
     size_t Value(const int token) const
     {
       return mapping[token];
     }
   
     size_t Size() const
     {
       return size;
     }
   
     void Clear()
     {
       mapping.fill(0);
     }
   
     const MapType& Mapping() const { return mapping; }
     MapType& Mapping() { return mapping; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(mapping));
       ar(CEREAL_NVP(size));
     }
   
    private:
     MapType mapping;
   
     size_t size;
   };
   
   } // namespace data
   } // namespace mlpack
   
   #endif
