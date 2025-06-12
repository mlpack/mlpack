/**
 * @file core/data/utilities.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 * @author Gopi Tatiraju
 *
 * Utilities functions that can be used during loading and saving the data..
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_UTILITIES_HPP
#define MLPACK_CORE_DATA_UTILITIES_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

namespace details {

template<typename Tokenizer>
std::vector<std::string> ToTokens(Tokenizer& lineTok)
{
  std::vector<std::string> tokens;
  std::transform(std::begin(lineTok), std::end(lineTok),
                 std::back_inserter(tokens),
                 [&tokens](std::string const &str)
  {
    std::string trimmedToken(str);
    Trim(trimmedToken);
    return std::move(trimmedToken);
  });

  return tokens;
}

inline
void TransposeTokens(std::vector<std::vector<std::string>> const &input,
                     std::vector<std::string>& output,
                     size_t index)
{
  output.clear();
  for (size_t i = 0; i != input.size(); ++i)
  {
    output.emplace_back(input[i][index]);
  }
}
} // namespace details


} //namespace data
} //namespace mlpack

#endif
