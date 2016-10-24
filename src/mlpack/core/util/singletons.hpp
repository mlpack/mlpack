/**
 * @file singletons.hpp
 * @author Ryan Curtin
 *
 * Definitions of singletons used by libmlpack.so.
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
