/**
 * @file core/math/shuffle_data.hpp
 * @author Ryan Curtin
 *
 * Given data points and labels, shuffle their ordering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_SHUFFLE_DATA_HPP
#define MLPACK_CORE_MATH_SHUFFLE_DATA_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Reorder a dense matrix or row vector.
 */
template<typename UVecType, typename MatType>
void ReorderData(const UVecType& ordering,
                 const MatType& in,
                 MatType& out,
                 const std::enable_if_t<!IsSparse<MatType>::value &&
                     !IsCube<MatType>::value>* = 0)
{
  // Properly handle the case where the input and output data are the same
  // object.
  MatType* outPtr = &out;
  if (&in == &out)
    outPtr = new MatType();

  outPtr->set_size(in.n_rows, in.n_cols);
  outPtr->cols(ordering) = in;

  // Clean up memory if needed.
  if (&in == &out)
  {
    out = std::move(*outPtr);
    delete outPtr;
  }
}

/**
 * Reorder a cube.
 */
template<typename UVecType, typename CubeType>
void ReorderData(const UVecType& ordering,
                 const CubeType& in,
                 CubeType& out,
                 const std::enable_if_t<IsCube<CubeType>::value>* = 0)
{
  // Properly handle the case where the input and output data are the same
  // object.
  CubeType* outPtr = &out;
  if (&in == &out)
    outPtr = new CubeType();

  outPtr->set_size(in.n_rows, in.n_cols,
      in.n_slices);
  for (size_t i = 0; i < ordering.n_elem; ++i)
  {
    outPtr->tube(0, ordering[i], outPtr->n_rows - 1, ordering[i]) =
        in.tube(0, i, in.n_rows - 1, i);
  }

  // Clean up memory if needed.
  if (&in == &out)
  {
    out = std::move(*outPtr);
    delete outPtr;
  }
}

/**
 * Reorder a sparse matrix.
 */
template<typename UVecType, typename SpMatType>
void ReorderData(const UVecType& ordering,
                 const SpMatType& in,
                 SpMatType& out,
                 const std::enable_if_t<IsSparse<SpMatType>::value>* = 0)
{
  // Extract coordinate list representation.
  arma::umat locations(2, in.n_nonzero);
  using ColType = typename GetDenseColType<SpMatType>::type;
  ColType values(in.n_nonzero);
  typename SpMatType::const_iterator it = in.begin();
  size_t index = 0;
  while (it != in.end())
  {
    locations(0, index) = it.row();
    locations(1, index) = ordering[it.col()];
    values(index) = (*it);
    ++it;
    ++index;
  }

  if (&in == &out)
  {
    SpMatType newOut(locations, values, in.n_rows,
        in.n_cols, true);

    out = std::move(newOut);
  }
  else
  {
    out = SpMatType(locations, values, in.n_rows,
        in.n_cols, true);
  }
}

/**
 * Shuffle two objects. It is expected that inputFirst and inputSecond have
 * the same number of columns (so, be sure that, if it is a vector, is a
 * row vector).
 *
 * Shuffled data will be output into outputFirst and outputSecond.
 */
template<typename FirstType, typename SecondType>
void ShuffleData(const FirstType& inputFirst,
                 const SecondType& inputSecond,
                 FirstType& outputFirst,
                 SecondType& outputSecond)
{
  // Generate ordering.
  using UVecType = typename GetURowType<FirstType>::type;
  UVecType ordering = shuffle(linspace<UVecType>(0,
      inputFirst.n_cols - 1, inputFirst.n_cols));

  // Shuffle data with the ordering
  ReorderData(ordering, inputFirst, outputFirst);
  ReorderData(ordering, inputSecond, outputSecond);
}

/**
 * Shuffle three objects. It is expected that inputFirst, inputSecond, and
 * inputThird have the same number of columns (so, be sure that, if it is a
 * vector, is a row vector).
 *
 * Shuffled data will be output into outputFirst, outputSecond, and outputThird.
 */
template<typename FirstType, typename SecondType, typename ThirdType>
void ShuffleData(const FirstType& inputFirst,
                 const SecondType& inputSecond,
                 const ThirdType& inputThird,
                 FirstType& outputFirst,
                 SecondType& outputSecond,
                 ThirdType& outputThird)
{
  // Generate ordering.
  using UVecType = typename GetURowType<FirstType>::type;
  UVecType ordering = shuffle(linspace<UVecType>(0,
      inputFirst.n_cols - 1, inputFirst.n_cols));

  // Shuffle data with the ordering
  ReorderData(ordering, inputFirst, outputFirst);
  ReorderData(ordering, inputSecond, outputSecond);
  ReorderData(ordering, inputThird, outputThird);
}

} // namespace mlpack

#endif
