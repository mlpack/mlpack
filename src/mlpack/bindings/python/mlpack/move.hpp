/**
 * @file move.hpp
 * @author Ryan Curtin
 *
 * Utility function for Cython to use std::move.
 */
#ifndef MLPACK_BINDINGS_PYTHON_CYTHON_MOVE_HPP
#define MLPACK_BINDINGS_PYTHON_CYTHON_MOVE_HPP

#include <utility>

namespace mlpack {
namespace util {

template<typename T>
void MoveToPtr(T* dest, T& src)
{
  *(dest) = std::move(src);
}

template<typename T>
void MoveFromPtr(T& dest, T* src)
{
  dest = std::move(*src);
}

} // namespace util
} // namespace mlpack

#endif
