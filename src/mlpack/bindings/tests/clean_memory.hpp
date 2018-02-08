/**
 * @file clean_memory.hpp
 * @author Ryan Curtin
 *
 * Delete any unique pointers that are held by the CLI object.  This is similar
 * to the code in end_program.hpp.
 */
#ifndef MLPACK_BINDINGS_TESTS_CLEAN_MEMORY_HPP
#define MLPACK_BINDINGS_TESTS_CLEAN_MEMORY_HPP

namespace mlpack {
namespace bindings {
namespace tests {

/**
 * Delete any unique pointers that are held by the CLI object.
 */
void CleanMemory();

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
