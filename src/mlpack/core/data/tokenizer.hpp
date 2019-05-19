/**
 * @file tokenizer.hpp
 * @author Jeffin Sam
 *
 * Implementation of Tokenizer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_TOKENIZER_HPP
#define MLPACK_CORE_DATA_TOKENIZER_HPP

#include <mlpack/prereqs.hpp>
#include <boost/utility/string_view.hpp>

namespace mlpack {
namespace data {
/**
 * A simple Tokenizer class
 */
class Tokenizer {
 public:
  boost::string_view operator()(boost::string_view& str,
                                boost::string_view delimiter) const {
    boost::string_view retval;

    while (retval.empty()) {
      std::size_t pos = str.find_first_of(delimiter);

      if (pos == str.npos) {
        retval = str;
        str.clear();
        return retval;
      }

      retval = str.substr(0, pos);

      str.remove_prefix(pos + 1);
    }
    return retval;
  }
};

} // namespace data
} // namespace mlpack

#endif
