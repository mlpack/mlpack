#include "maximal_inputs.hpp"

namespace mlpack {
namespace nn {

namespace {

std::pair<arma::uword, arma::uword> GetSize(arma::mat const &input)
{
  arma::uword rows = 0, cols = 0;
  if(std::pow(std::floor(std::sqrt(input.n_cols)), 2) != input.n_cols)
  {
    cols = static_cast<arma::uword>(std::ceil(std::sqrt(input.n_cols)));
    while(input.n_cols % cols != 0 && cols < 1.2*std::sqrt(input.n_cols))
    {
      ++cols;
    }
    rows = static_cast<arma::uword>
           (std::ceil(input.n_cols/static_cast<double>(cols)));
  }
  else
  {
    cols = static_cast<arma::uword>(std::sqrt(input.n_cols));
    rows = cols;
  }

  return {rows, cols};
}

void MaximizeHiddenUnit(arma::uword rows, arma::uword cols,
                        const arma::mat &input,
                        arma::mat &output)
{
  const arma::uword size = rows * cols;
  output.set_size(input.n_rows, size);
  for(arma::uword i = 0; i != size; ++i)
  {
    const double max  = arma::max(arma::abs(input.col(i)));
    if (max != 0.0)
    {
      output.col(i) = input.col(i) / max;
    }
    else
    {
      output.col(i) = input.col(i);
    }
  }
}

}



std::pair<arma::uword, arma::uword> MaximalInputs(const arma::mat &parameters, arma::mat &output)
{
  arma::mat paramTemp(parameters.submat(0, 0, (parameters.n_rows-1)/2 - 1,
                                        parameters.n_cols - 2).t());
  double const mean = arma::mean(arma::mean(paramTemp));
  paramTemp -= mean;

  const auto Size = GetSize(paramTemp);
  MaximizeHiddenUnit(Size.first, Size.second, paramTemp, output);

  return Size;
}

void ColumnsToBlocks(const arma::mat &maximalInputs,
                     arma::mat &outputs, arma::uword rows, arma::uword cols,
                     bool scale, double minRange,
                     double maxRange)
{
  if(rows * cols != maximalInputs.n_cols)
  {
    throw std::range_error("rows * cols != maximalInputs.n_cols");
  }

  arma::uword const squareRows = static_cast<arma::uword>(std::sqrt(maximalInputs.n_rows));
  arma::uword const buf = 1;

  arma::uword const offset = squareRows+buf;
  outputs.ones(buf+rows*(offset),
               buf+cols*(offset));
  outputs *= -1;

  arma::uword k = 0;
  for(arma::uword i = 0; i != rows; ++i) {
    for(arma::uword j = 0; j != cols; ++j) {
      // Now, copy the elements of the row to the output submatrix.
      const arma::uword minRow = buf + i * offset;
      const arma::uword minCol = buf + j * offset;
      const arma::uword maxRow = i * offset + squareRows;
      const arma::uword maxCol = j * offset + squareRows;

      outputs.submat(minRow, minCol, maxRow, maxCol) =
        arma::reshape(maximalInputs.col(k++), squareRows, squareRows);
    }
  }

  if(scale)
  {
    const double max = outputs.max();
    const double min = outputs.min();
    if((max - min) != 0)
    {
      outputs = (outputs - min) / (max - min) * (maxRange - minRange) + minRange;
    }
  }
}

} // namespace nn
} // namespace mlpack
