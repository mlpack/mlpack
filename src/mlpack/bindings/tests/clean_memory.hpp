/**
 * @file bindings/tests/clean_memory.hpp
 * @author Ryan Curtin
 *
 * Delete any unique pointers that are held by the IO object.  This is similar
 * to the code in end_program.hpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_TESTS_CLEAN_MEMORY_HPP
#define MLPACK_BINDINGS_TESTS_CLEAN_MEMORY_HPP

#include <mlpack/core/util/params.hpp>

namespace mlpack {
namespace bindings {
namespace tests {

/**
 * Delete any unique pointers that are held by the IO object.
 */
void CleanMemory(util::Params& params);

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
