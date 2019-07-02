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
#include <unordered_set>

namespace mlpack {
namespace data {

class StringCleaning
{
 public:
  /**
  * Function to remove punctuation from a given vector of strings.
  *
  * @param input Vector of strings
  */
  void RemovePunctuation(std::vector<std::string>& input,
                         const std::string& punctuation = "!\"#$%&'()*+"
                         ",-./:;<=>?@[\\]^_`{|}~")
  { 
    for (auto& str : input)
    {
      str.erase(boost::remove_if(str, boost::is_any_of(punctuation)),
          str.end());
    }
  }

  /**
  * Function to remove a set of chars from a given vector of strings.
  * A function which accepts a function as an argument and removes the
  * character if the funtion returns true.
  * 
  * @param input Vector of strings.
  * @param fun A function which takes const char& c as input and returns bool;
  *
  * This can either be a function pointer or function object or a lamda
  * function. Its return value should be of bool type.
  *
  * Definiation of function should be of type
  * bool fn(const char& c)
  */
  template<typename ReturnType>
  void RemoveChar(std::vector<std::string>& input,const ReturnType fun)
  { 
    for (auto& str : input)
    {
      str.erase(boost::remove_if(str, fun), str.end());
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
  * Function to convert given vector of strings to Upper case.
  *
  * @param input Vector of strings
  */
  void UpperCase(std::vector<std::string>& input)
  {
    for (auto& str : input)
    {
      boost::algorithm::to_upper(str);
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
  */
  template<typename TokenizerType>
  void RemoveStopWords(std::vector<std::string>& input,
                       std::unordered_set<boost::string_view,
                       boost::hash<boost::string_view>>stopwords,
                       const TokenizerType tokenizer)
  {
    std::string copy;
    boost::string_view token;
    boost::string_view strView;
    for (auto& str : input)
    {
      copy.clear();
      strView = str;
      token = tokenizer(strView);
      while (!token.empty())
      {
        if (stopwords.find(token) == stopwords.end())
        {
          // token is not a stop word add it;
          copy += " " + std::string(token);
        }
        token = tokenizer(strView);
      }
      boost::algorithm::trim_left(copy);
      str = copy;
    }
  }
}; // Class StringCleaning

} // namespace data
} // namespace mlpack

#endif
