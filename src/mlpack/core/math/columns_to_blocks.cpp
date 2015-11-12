#include "columns_to_blocks.hpp"

namespace mlpack {
namespace math {

ColumnsToBlocks::ColumnsToBlocks(arma::uword rows, arma::uword cols) :
    minRange(0),
    maxRange(255),
    scale(false),
    rows(rows),
    cols(cols)
{
}

bool ColumnsToBlocks::IsPerfectSquare(arma::uword value) const
{
  if (value < 0)
  {
    return false;
  }

  const int root = std::round(std::sqrt(value));
  return value == root * root;
}

void ColumnsToBlocks::Transform(const arma::mat &maximalInputs, arma::mat &output)
{
  if(!IsPerfectSquare(maximalInputs.n_rows))
  {
    throw std::runtime_error("maximalInputs.n_rows should be perfect square");
  }

  arma::uword const squareRows = static_cast<arma::uword>(std::sqrt(maximalInputs.n_rows));
  arma::uword const buf = 1;

  arma::uword const offset = squareRows+buf;
  output.ones(buf+ rows*(offset),
              buf+ cols*(offset));
  output *= -1;

  arma::uword k = 0;
  const arma::uword maxSize = rows * cols;
  for(arma::uword i = 0; i != rows; ++i) {
    for(arma::uword j = 0; j != cols; ++j) {
      // Now, copy the elements of the row to the output submatrix.
      const arma::uword minRow = buf + i * offset;
      const arma::uword minCol = buf + j * offset;
      const arma::uword maxRow = i * offset + squareRows;
      const arma::uword maxCol = j * offset + squareRows;

      output.submat(minRow, minCol, maxRow, maxCol) =
        arma::reshape(maximalInputs.col(k++), squareRows, squareRows);
      if(k >= maxSize) {
        break;
      }
    }
  }

  if(scale)
  {
    const double max = output.max();
    const double min = output.min();
    if((max - min) != 0)
    {
      output = (output - min) / (max - min) * (maxRange - minRange) + minRange;
    }
  }
}

} // namespace math
} // namespace mlpack
