/**
 * @file core/math/multiply_slices_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of matrix multiplication over slices.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_MATH_MULTIPLY_SLICES_IMPL_HPP
#define MLPACK_CORE_MATH_MULTIPLY_SLICES_IMPL_HPP

#include "multiply_slices.hpp"

namespace mlpack {

template <typename CubeType>
CubeType MultiplyCube2Cube(const CubeType& cubeA,
    const CubeType& cubeB,
    const bool aTranspose,
    const bool bTranspose)
{
  size_t rows = cubeA.n_rows, cols = cubeB.n_cols, slices = cubeA.n_slices;

  if (cubeA.n_slices != cubeB.n_slices)
    Log::Fatal << "Number of slices is not same in both cubes." << std::endl;

  if (aTranspose && bTranspose)
  {
    if (cubeA.n_rows != cubeB.n_cols)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    rows = cubeA.n_cols;
    cols = cubeB.n_rows;
  }
  else if (bTranspose && !aTranspose)
  {
    if (cubeA.n_cols != cubeB.n_cols)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    cols = cubeB.n_rows;
  }
  else if (aTranspose && !bTranspose)
  {
    if (cubeA.n_rows != cubeB.n_rows)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    rows = cubeA.n_cols;
  }
  else
  {
    if (cubeA.n_cols != cubeB.n_rows)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
  }

  CubeType z(rows, cols, slices);

  if (aTranspose && bTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = trans(cubeB.slice(i) * cubeA.slice(i));
  }
  else if (bTranspose && !aTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = cubeA.slice(i) * cubeB.slice(i).t();
  }
  else if (aTranspose && !bTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = cubeA.slice(i).t() * cubeB.slice(i);
  }
  else
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = cubeA.slice(i) * cubeB.slice(i);
  }
  return z;
}

template <typename MatType, typename CubeType>
CubeType MultiplyMat2Cube(const MatType& matA,
    const CubeType& cubeB,
    const bool aTranspose,
    const bool bTranspose)
{
  size_t rows = matA.n_rows, cols = cubeB.n_cols, slices = cubeB.n_slices;

  if (aTranspose && bTranspose)
  {
    if (matA.n_rows != cubeB.n_cols)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    rows = matA.n_cols;
    cols = cubeB.n_rows;
  }
  else if (bTranspose && !aTranspose)
  {
    if (matA.n_cols != cubeB.n_cols)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    cols = cubeB.n_rows;
  }
  else if (aTranspose && !bTranspose)
  {
    if (matA.n_rows != cubeB.n_rows)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    rows = matA.n_cols;
  }
  else
  {
    if (matA.n_cols != cubeB.n_rows)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
  }

  CubeType z(rows, cols, slices);

  if (aTranspose && bTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = trans(cubeB.slice(i) * matA);
  }
  else if (bTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = matA * cubeB.slice(i).t();
  }
  else if (aTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = matA.t() * cubeB.slice(i);
  }
  else
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = matA * cubeB.slice(i);
  }
  return z;
}

template <typename CubeType, typename MatType>
CubeType MultiplyCube2Mat(const CubeType& cubeA,
    const MatType& matB,
    const bool aTranspose,
    const bool bTranspose)
{
  size_t rows = cubeA.n_rows, cols = matB.n_cols, slices = cubeA.n_slices;

  if (aTranspose && bTranspose)
  {
    if (cubeA.n_rows != matB.n_cols)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    rows = cubeA.n_cols;
    cols = matB.n_rows;
  }
  else if (bTranspose && !aTranspose)
  {
    if (cubeA.n_cols != matB.n_cols)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    cols = matB.n_rows;
  }
  else if (aTranspose && !bTranspose)
  {
    if (cubeA.n_rows != matB.n_rows)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;
    rows = cubeA.n_cols;
  }
  else
    if (cubeA.n_cols != matB.n_rows)
      Log::Fatal << "Matrix multiplication invalid!" << std::endl;

  CubeType z(rows, cols, slices);

  if (aTranspose && bTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = trans(matB * cubeA.slice(i));
  }
  else if (bTranspose && !aTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = cubeA.slice(i) * matB.t();
  }
  else if (aTranspose && !bTranspose)
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = cubeA.slice(i).t() * matB;
  }
  else
  {
    for (size_t i = 0; i < slices; ++i)
      z.slice(i) = cubeA.slice(i) * matB;
  }
  return z;
}

} // namespace mlpack

#endif
