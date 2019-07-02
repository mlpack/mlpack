/**
 * @file char_split.hpp
 * @author Jeffin Sam
 *
 * Implementation of CharSplit class which tokenizes string into character.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CHAR_SPLIT_HPP
#define MLPACK_CORE_DATA_CHAR_SPLIT_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"

namespace mlpack {
namespace data {
/**
 * A simple CharSplit class.
 *
 * The class is used to split the documents into characters.The function
 * returns a char token, and successive calls, would return many such tokens.
 */
class CharSplit
{
 public:
  /**
  * A function object which take boost::string_view as input and
  * return a boost::string_view (token).
  * @param str A string to retriev token from.
  */
  boost::string_view operator()(boost::string_view& str) const
  {
    if (str.empty())
      return str;
    boost::string_view retval = str.substr(0, 1);
    str.remove_prefix(1);
    return retval;
  }
}; // CharSplit class

} // namespace data
} // namespace mlpack

#endif
