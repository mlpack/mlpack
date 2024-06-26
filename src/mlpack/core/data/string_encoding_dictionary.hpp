/**
 * @file core/data/string_encoding_dictionary.hpp
 * @author Jeffin Sam
 * @author Mikhail Lozhnikov
 *
 * Definition of the StringEncodingDictionary class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_ENCODING_DICTIONARY_HPP
#define MLPACK_CORE_DATA_STRING_ENCODING_DICTIONARY_HPP

#include <mlpack/prereqs.hpp>

#include <array>
#include <deque>
#include <functional>
#include <unordered_map>

namespace mlpack {
namespace data {

/**
 * This class provides a dictionary interface for the purpose of string
 * encoding. It works like an adapter to the internal dictionary.
 *
 * @tparam Token Type of the token that the dictionary stores.
 */
template<typename Token>
class StringEncodingDictionary
{
 public:
  //! A convenient alias for the internal type of the map.
  using MapType = std::unordered_map<Token, size_t>;

  //! The type of the token that the dictionary stores.
  using TokenType = Token;

  /**
   * The function returns true if the dictionary contains the given token.
   *
   * @param token The given token.
   */
  bool HasToken(const Token& token) const
  {
    return mapping.find(token) != mapping.end();
  }

  /**
   * The function adds the given token to the dictionary and assigns a label
   * to the token. The label is equal to the resulting size of the dictionary.
   * The function returns the assigned label.
   *
   * @param token The given token.
   */
  template<typename T>
  size_t AddToken(T&& token)
  {
    size_t size = mapping.size();

    mapping[std::forward<T>(token)] = ++size;

    return size;
  }

  /**
   * The function returns the label assigned to the given token. The function
   * throws std::out_of_range if no such token is found.
   *
   * @param token The given token.
   */
  size_t Value(const Token& token) const
  {
    return mapping.at(token);
  }

  //! Get the size of the dictionary.
  size_t Size() const { return mapping.size(); }

  //! Clear the dictionary.
  void Clear()
  {
    mapping.clear();
  }

  //! Get the mapping.
  const MapType& Mapping() const { return mapping; }
  //! Modify the mapping.
  MapType& Mapping() { return mapping; }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mapping));
  }

 private:
  //! The dictionary itself.
  MapType mapping;
};

/*
 * Specialization of the StringEncodingDictionary class for std::string_view.
 */
template<>
class StringEncodingDictionary<std::string_view>
{
 public:
  //! A convenient alias for the internal type of the map.
  using MapType = std::unordered_map<
      std::string_view,
      size_t,
      std::hash<std::string_view>>;

  //! The type of the token that the dictionary stores.
  using TokenType = std::string_view;

  //! Construct the default class.
  StringEncodingDictionary() = default;

  //! Copy the class from the given object.
  StringEncodingDictionary(const StringEncodingDictionary& other) :
      tokens(other.tokens)
  {
    for (const std::string& token : tokens)
      mapping[token] = other.mapping.at(token);
  }

  //! Standard move constructor.
  StringEncodingDictionary(StringEncodingDictionary&& other) = default;

  //! Copy the class from the given object.
  StringEncodingDictionary& operator=(const StringEncodingDictionary& other)
  {
    tokens = other.tokens;
    mapping.clear();

    for (const std::string& token : tokens)
      mapping[token] = other.mapping.at(token);

    return *this;
  }

  //! Standard move assignment operator.
  StringEncodingDictionary& operator=(
      StringEncodingDictionary&& other) = default;

  /**
   * The function returns true if the dictionary contains the given token.
   *
   * @param token The given token.
   */
  bool HasToken(const std::string_view token) const
  {
    return mapping.find(token) != mapping.end();
  }

  /**
   * The function adds the given token to the dictionary and assigns a label
   * to the token. The label is equal to the resulting size of the dictionary.
   * The function returns the assigned label.
   *
   * @param token The given token.
   */
  size_t AddToken(const std::string_view token)
  {
    tokens.emplace_back(token);

    size_t size = mapping.size();

    mapping[tokens.back()] = ++size;

    return size;
  }

  /**
   * The function returns the label assigned to the given token. The function
   * throws std::out_of_range if no such token is found.
   *
   * @param token The given token.
   */
  size_t Value(const std::string_view token) const
  {
    return mapping.at(token);
  }

  //! Get the size of the dictionary.
  size_t Size() const { return mapping.size(); }

  //! Clear the dictionary.
  void Clear()
  {
    mapping.clear();
    tokens.clear();
  }

  //! Get the tokens.
  const std::deque<std::string>& Tokens() const { return tokens; }
  //! Modify the tokens.
  std::deque<std::string>& Tokens() { return tokens; }

  //! Get the mapping.
  const MapType& Mapping() const { return mapping; }
  //! Modify the mapping.
  MapType& Mapping() { return mapping; }

  /**
   * Serialize the class to the given archive.
   */
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
  //! The tokens that the dictionary stores.
  std::deque<std::string> tokens;

  //! The mapping itself.
  MapType mapping;
};

template<>
class StringEncodingDictionary<int>
{
 public:
  //! A convenient alias for the internal type of the map.
  using MapType = std::array<size_t, 1 << CHAR_BIT>;

  //! The type of the token that the dictionary stores.
  using TokenType = int;

  //! Construct the default class.
  StringEncodingDictionary() :
    size(0)
  {
    mapping.fill(0);
  }

  /**
   * The function returns true if the dictionary contains the given token.
   * The token must belong to [0, 255]; otherwise the behavior is undefined.
   *
   * @param token The given token.
   */
  bool HasToken(const int token) const
  {
    return mapping[token] > 0;
  }

  /**
   * The function adds the given token to the dictionary and assigns a label
   * to the token. The token must belong to [0, 255]; otherwise the behavior
   * is undefined. The label is equal to the resulting size of the dictionary.
   * The function returns the assigned label.
   *
   * @param token The given token.
   */
  size_t AddToken(const int token)
  {
    mapping[token] = ++size;

    return size;
  }

  /**
   * The function returns the label assigned to the given token. The function
   * doesn't verify that the dictionary contains the token. The token must
   * belong to [0, 255]; otherwise the behavior is undefined.
   *
   * @param token The given token.
   */
  size_t Value(const int token) const
  {
    return mapping[token];
  }

  //! Get the size of the dictionary.
  size_t Size() const
  {
    return size;
  }

  //! Clear the dictionary.
  void Clear()
  {
    mapping.fill(0);
  }

  //! Get the mapping.
  const MapType& Mapping() const { return mapping; }
  //! Modify the mapping.
  MapType& Mapping() { return mapping; }

  /**
   * Serialize the class to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mapping));
    ar(CEREAL_NVP(size));
  }

 private:
  //! The mapping itself.
  MapType mapping;

  //! The size of the dictionary.
  size_t size;
};

} // namespace data
} // namespace mlpack

#endif
