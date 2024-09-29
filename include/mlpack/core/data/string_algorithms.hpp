/** 
 * @file core/data/string_algorithms.hpp
 * @author Gopi M. Tatiraju
 *
 * Utility functions related to string manipulation
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_STRING_ALGORITHMS_HPP
#define MLPACK_CORE_DATA_STRING_ALGORITHMS_HPP

namespace mlpack {
namespace data {

/**
 * A simple trim function to strip off whitespaces 
 * from both the sides of a string. If input is a string
 * with all spaces then str will be empty string.
 *
 * @param str the string to be trimmed.
 */
inline void Trim(std::string& str)
{
  if (str.find_first_not_of(' ') == std::string::npos)
  {
    str = "";
    return;
  }

  size_t startIndex = 0;

  while (std::isspace(str[startIndex]))
    startIndex++;

  size_t endIndex = str.size() - 1;

  while (std::isspace(str[endIndex]))
    endIndex--;

  std::string trimmedStr;
  
  // Using ternary operator is not recommended here.
  // Ternary operator is only useful for simple expressions
  // that don't involve varrying types.
  // https://en.cppreference.com/w/cpp/language/operator_other
  if (endIndex - startIndex == str.size())
    trimmedStr = std::move(str);
  else
    trimmedStr = str.substr(startIndex, endIndex - startIndex + 1);

  str = trimmedStr;
}

/**
 * Trim off characters from start and end of
 * of the string. The supplied function is
 * used to determine which characters will
 * be trimmed off.
 *
 * @param str the string to be trimmed.
 * @param func function to determine the characters which should be trimmed.
 */
inline void TrimIf(std::string &str, std::function<bool(char)> func)
{
  if (str.find_first_not_of(' ') == std::string::npos)
  {
    str = "";
    return;
  }

  size_t startIndex = 0;

  for (size_t i = 0; i < str.size(); i++)
  {
    bool match = func(str[i]);

    if (match)
      startIndex++;
    else
      break;
  }

  size_t endIndex = str.size() - 1;

  for (int i = str.size() - 1; i >= 0; i--)
  {
    bool match = func(str[i]);
    if (match)
      endIndex--;
    else
      break;
  }

  std::string trimmedStr;

  // Using ternary operator is not recommended here.
  // Ternary operator is only useful for simple expressions
  // that don't involve varrying types.
  // https://en.cppreference.com/w/cpp/language/operator_other
  if (endIndex - startIndex == str.size())
    trimmedStr = std::move(str);
  else
    trimmedStr = str.substr(startIndex, endIndex - startIndex + 1);

  str = trimmedStr;
}

/**
 * Splits the given string into tokens, using the given delimiter to split.
 * An escape character should be specified to indicate escape sequences where
 * the delimiter may be safely used.  For instance, with the delimiter ',' and
 * the escape character '"' (a double quote), the line
 *
 * hello, "one, two"
 *
 * will be split into two tokens: "hello", and "one, two".
 *
 * @param line Input string to tokenize.
 * @param tokenDelim Character to use as delimiter between tokens.
 * @param escape Escape character, usually " or '.
 */
inline std::vector<std::string> Tokenize(
    std::string& line,
    char tokenDelim,
    char escape)
{
  std::vector<std::string> tokens;

  // Shortcut: if the line is empty, it has no tokens.
  if (line.size() == 0)
    return tokens;

  bool inEscape = false;
  bool lastBackslash = false;
  std::string currentToken;
  size_t lastSplitIndex = 0;
  for (size_t currentIndex = 0; currentIndex < line.size(); ++currentIndex)
  {
    char c = line[currentIndex];

    if (c == '\\')
    {
      // Make sure we mark that we just encountered a backslash.
      lastBackslash = true;
      continue;
    }
    else if (c == escape && !lastBackslash)
    {
      // We've encountered one of our escape characters, so if we were already
      // in an escape sequence, we are no longer, and if we weren't, we are now.
      inEscape = !inEscape;
    }
    else if (c == escape && lastBackslash)
    {
      // If we are in an escape sequence and we encounter an escaped delimiter,
      // we want to remove the '\' that prepends the escaped delimiter.
      currentToken.append(line.substr(lastSplitIndex,
          currentIndex - 2 - lastSplitIndex));
      lastSplitIndex = currentIndex;
    }
    else if (c == tokenDelim && !inEscape)
    {
      // If the current character is a delimiter, then finish the previous token
      // and add it to the list of tokens.
      currentToken.append(line.substr(lastSplitIndex,
          currentIndex - lastSplitIndex));
      tokens.push_back(currentToken);
      currentToken.clear();
      lastSplitIndex = currentIndex + 1;
    }

    lastBackslash = false;
  }

  // Push the last token.
  currentToken.append(line.substr(lastSplitIndex));
  tokens.push_back(currentToken);

  return tokens;
}

}  // namespace data
}  // namespace mlpack

#endif 
