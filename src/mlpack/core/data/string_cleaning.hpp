/**
 * @file string_cleaning.hpp
 * @author jeffin sam
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_CLEANING_HPP
#define MLPACK_CORE_DATA_STRING_CLEANING_HPP

#include <mlpack/prereqs.hpp>
#include <boost/range/algorithm/remove_if.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"
#include <mlpack/core/data/tokenizer/strtok.hpp>
#include <boost/functional/hash.hpp>

std::string stopwords =
#include "mlpack/core/data/stopwords.txt"
;

class Hasher
{
 public:
  std::size_t operator()(boost::string_view str)
  {
    return boost::hash_range<const char*>(str.begin(), str.end());
  }
};

namespace mlpack {
namespace data {

/**
 * Function to remove punctuation from a given vector of strings.
 *
 * @param input Vector of strings
 */
void RemovePunctuation(std::vector<std::string>& input)
{
  std::string punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  for (auto& str : input)
  {
    str.erase(boost::remove_if(str, boost::is_any_of(punctuation)), str.end());
  }
}

/**
 * Function to convert given vector of strings to lower case.
 *
 * @param input Vector of strings
 */
void LowerCase(std::vector<std::string>& input)
{
  for (auto& str : input)
  {
    boost::algorithm::to_lower(str);
  }
}

/**
 * Function to remove stopwords from a given vector of strings.
 *
 * @param input Vector of strings
 * @param tokenizer A function that accepts a boost::string_view as
 *                  an argument and returns a token.
 * This can either be a function pointer or function object or a lamda
 * function. Its return value should be a boost::string_view, a token.
 *
 * Definiation of function should be of type
 * boost::string_view fn()(boost::string_view& str)
 *
 */
template<typename TokenizerType>
void RemoveStopWords(std::vector<std::string>& input, TokenizerType tokenizer)
{
  boost::string_view strView(stopwords);
  boost::string_view token;
  std::unordered_map<boost::string_view, std::size_t, Hasher> stopword;
  data::Strtok split("\n");
  token = split(strView);
  while (!token.empty())
  {
    if (stopword.count(token) == 0)
    {
      stopword[std::move(token)] = 1;
    }
    token = split(strView);
  }
  std::string copy;
  std::string tokenStr;
  for (auto& str : input)
  {
    copy = str;
    strView = copy;
    token = tokenizer(strView);
    while (!token.empty())
    {
      if (stopword.count(token) > 0)
      {
        // token is a  a stop word remove it;
        // Search for the stopword in string
        // Converting it to string won't take much of space since there are
        // relatively less stop words
        tokenStr = std::string(token);
        size_t pos = str.find(tokenStr);
        if (pos != std::string::npos)
        {
          // If found then erase it from string
          str.erase(pos, tokenStr.length() + 1);
        }
      }
      token = tokenizer(strView);
    }
  }
}

} // namespace data
} // namespace mlpack

#endif
