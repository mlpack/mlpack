/**
 * @file core/math/columns_to_blocks_impl.hpp
 * @author Tham Ngap Wei
 *
 * Implementation of the ColumnsToBlocks class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_COLUMNS_TO_BLOCKS_IMPL_HPP
#define MLPACK_CORE_MATH_COLUMNS_TO_BLOCKS_IMPL_HPP

#include "columns_to_blocks.hpp"

namespace mlpack {

inline ColumnsToBlocks::ColumnsToBlocks(const size_t rows,
                                        const size_t cols,
                                        const size_t blockHeight,
                                        const size_t blockWidth) :
    blockHeight(blockHeight),
    blockWidth(blockWidth),
    bufSize(1),
    bufValue(-1),
    minRange(0),
    maxRange(255),
    scale(false),
    rows(rows),
    cols(cols)
{
}

inline bool ColumnsToBlocks::IsPerfectSquare(const size_t value) const
{
  const size_t root = (size_t) std::round(std::sqrt(value));
  return (value == root * root);
}

inline void ColumnsToBlocks::Transform(const arma::mat& maximalInputs,
                                       arma::mat& output)
{
  if (blockHeight == 0 || blockWidth == 0)
  {
    //! TODO: Maybe replace std::runtime_error with Log::Fatal.
    if (!IsPerfectSquare(maximalInputs.n_rows))
    {
      std::ostringstream oss;
      oss << "ColumnsToBlocks::Transform(): input.n_rows ("
          << maximalInputs.n_rows << ") should be a perfect square!";
      throw std::runtime_error(oss.str());
    }

    size_t const squareRows =
        static_cast<size_t>(std::sqrt(maximalInputs.n_rows));
    blockHeight = squareRows;
    blockWidth = squareRows;
  }

  if (blockHeight * blockWidth != maximalInputs.n_rows)
  {
    std::ostringstream oss;
    oss << "ColumnsToBlocks::Transform(): blockHeight * blockWidth ("
        << blockHeight << " * " << blockWidth << ") should be equal to "
        << "input.n_rows (" << maximalInputs.n_rows << ")!";
    throw std::runtime_error(oss.str());
  }

  const size_t rowOffset = blockHeight + bufSize;
  const size_t colOffset = blockWidth + bufSize;
  output.set_size(bufSize + rows * rowOffset, bufSize + cols * colOffset);
  output.fill(bufValue);

  size_t k = 0;
  const size_t maxSize = std::min(rows * cols, (size_t) maximalInputs.n_cols);
  for (size_t i = 0; i != rows; ++i)
  {
    for (size_t j = 0; j != cols; ++j)
    {
      // Now, copy the elements of the row to the output submatrix.
      const size_t minRow = bufSize + i * rowOffset;
      const size_t minCol = bufSize + j * colOffset;
      const size_t maxRow = bufSize + i * rowOffset + blockHeight - 1;
      const size_t maxCol = bufSize + j * colOffset + blockWidth - 1;

      output.submat(minRow, minCol, maxRow, maxCol) =
          arma::reshape(maximalInputs.col(k++), blockHeight, blockWidth);

      if (k >= maxSize)
        break;
    }
  }

  if (scale)
  {
    const double max = output.max();
    const double min = output.min();
    if ((max - min) != 0)
    {
      output = (output - min) / (max - min) * (maxRange - minRange) + minRange;
    }
  }
}

} // namespace mlpack

#endif
