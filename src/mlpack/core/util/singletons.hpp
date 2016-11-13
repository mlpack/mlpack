/**
 * @file singletons.hpp
 * @author Ryan Curtin
 *
 * Definitions of singletons used by libmlpack.so.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_SINGLETONS_HPP
#define MLPACK_CORE_UTIL_SINGLETONS_HPP

#include "cli_deleter.hpp"
#include <mlpack/mlpack_export.hpp>

namespace mlpack {
namespace util {

extern MLPACK_EXPORT CLIDeleter cliDeleter;

} // namespace util
} // namespace mlpack

#endif
