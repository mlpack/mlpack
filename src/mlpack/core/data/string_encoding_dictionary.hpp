/**
 * @file string_encoding_dictionary.hpp
 * @author Jeffin Sam
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
#include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
#include <unordered_map>
#include <deque>
#include <array>

namespace mlpack {
namespace data {

/*
 * Definition of the StringEncodingDictionary class.
 *
 * @tparam Token The type of the token that the dictionary stores.
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
   * The function returns true if the dictionary caontains the given token.
   *
   * @param token The given token.
   */
  bool HasToken(const Token& token) const
  {
    return mapping.find(token) != mapping.end();
  }

  /**
   * The function adds the given token to the dictionary and assigns a label
   * to the token.
   *
   * @param token The given token.
   */
  template<typename T>
  void AddToken(T&& token)
  {
    size_t size = mapping.size();

    mapping[std::forward<T>(token)] = size + 1;
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
   * Serialize the dictionary to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(mapping);
  }

 private:
  //! The dictionary itself.
  MapType mapping;
};

/*
 * Specialization of the StringEncodingDictionary class for boost::string_view.
 */
template<>
class StringEncodingDictionary<boost::string_view>
{
 public:
  //! A convenient alias for the internal type of the map.
  using MapType = std::unordered_map<
      boost::string_view,
      size_t,
      boost::hash<boost::string_view>>;

  //! The type of the token that the dictionary stores.
  using TokenType = boost::string_view;

  //! Cunstruct the default class.
  StringEncodingDictionary() = default;

  //! Construct the class using the given object.
  StringEncodingDictionary(const StringEncodingDictionary& other) :
      tokens(other.tokens)
  {
    for (const std::string& token : tokens)
      mapping[token] = other.mapping.at(token);
  }

  //! Standard move constructor.
  StringEncodingDictionary(StringEncodingDictionary&& other) = default;

  //! Copy the class using the given object.
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
   * The function returns true if the dictionary caontains the given token.
   *
   * @param token The given token.
   */
  bool HasToken(boost::string_view token) const
  {
    return mapping.find(token) != mapping.end();
  }

  /**
   * The function adds the given token to the dictionary and assigns a label
   * to the token.
   *
   * @param token The given token.
   */
  void AddToken(boost::string_view token)
  {
    tokens.emplace_back(token);

    size_t size = mapping.size();

    mapping[tokens.back()] = size + 1;
  }

  /**
   * The function returns the label assigned to the given token. The function
   * throws std::out_of_range if no such token is found.
   *
   * @param token The given token.
   */
  size_t Value(boost::string_view token) const
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
   * Serialize the dictionary to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    size_t numTokens = tokens.size();

    ar & BOOST_SERIALIZATION_NVP(numTokens);

    if (Archive::is_loading::value)
    {
      tokens.resize(numTokens);

      for (std::string& token : tokens)
      {
        ar & BOOST_SERIALIZATION_NVP(token);

        size_t tokenValue = 0;
        ar & BOOST_SERIALIZATION_NVP(tokenValue);
        mapping[token] = tokenValue;
      }
    }
    if (Archive::is_saving::value)
    {
      for (std::string& token : tokens)
      {
        ar & BOOST_SERIALIZATION_NVP(token);

        size_t tokenValue = mapping.at(token);
        ar & BOOST_SERIALIZATION_NVP(tokenValue);
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

  //! Cunstruct the default class.
  StringEncodingDictionary() :
    size(0)
  {
    mapping.fill(0);
  }

  /**
   * The function returns true if the dictionary caontains the given token.
   * The given token must belong to [0, 255].
   *
   * @param token The given token.
   */
  bool HasToken(int token) const
  {
    return mapping[token] > 0;
  }

  /**
   * The function adds the given token to the dictionary and assigns a label
   * to the token. The given token must belong to [0, 255].
   *
   * @param token The given token.
   */
  void AddToken(int token)
  {
    mapping[token] = ++size;
  }

  /**
   * The function returns the label assigned to the given token. The function
   * doesn't verify that the dictionary contains the given token.
   *
   * @param token The given token.
   */
  size_t Value(int token) const
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
   * Serialize the dictionary to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(mapping);
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
