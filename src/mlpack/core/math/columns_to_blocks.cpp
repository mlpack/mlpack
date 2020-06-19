/**
 * @file columns_to_blocks.cpp
 * @author Tham Ngap Wei
 *
 * Implementation of the ColumnsToBlocks class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "columns_to_blocks.hpp"

namespace mlpack {
namespace math {

ColumnsToBlocks::ColumnsToBlocks(const size_t rows,
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

bool ColumnsToBlocks::IsPerfectSquare(const size_t value) const
{
  const size_t root = (size_t) std::round(std::sqrt(value));
  return (value == root * root);
}

void ColumnsToBlocks::Transform(const arma::mat& maximalInputs,
                                arma::mat& output)
{
  if (!IsPerfectSquare(maximalInputs.n_rows))
  {
    throw std::runtime_error("maximalInputs.n_rows should be perfect square");
  }

  if (blockHeight == 0 || blockWidth == 0)
  {
    size_t const squareRows =
        static_cast<size_t>(std::sqrt(maximalInputs.n_rows));
    blockHeight = squareRows;
    blockWidth = squareRows;
  }

  if (blockHeight * blockWidth != maximalInputs.n_rows)
  {
    throw std::runtime_error("blockHeight * blockWidth should "
                             "equal to maximalInputs.n_rows");
  }

  const size_t rowOffset = blockHeight+bufSize;
  const size_t colOffset = blockWidth+bufSize;
  output.ones(bufSize + rows * rowOffset,
              bufSize + cols * colOffset);
  output *= bufValue;

  size_t k = 0;
  const size_t maxSize = std::min(rows * cols, (size_t) maximalInputs.n_cols);
  for (size_t i = 0; i != rows; ++i)
  {
    for (size_t j = 0; j != cols; ++j)
    {
      // Now, copy the elements of the row to the output submatrix.
      const size_t minRow = bufSize + i * rowOffset;
      const size_t minCol = bufSize + j * colOffset;
      const size_t maxRow = i * rowOffset + blockHeight;
      const size_t maxCol = j * colOffset + blockWidth;

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

} // namespace math
} // namespace mlpack
