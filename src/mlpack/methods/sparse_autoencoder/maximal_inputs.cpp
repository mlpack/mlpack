#include "maximal_inputs.hpp"

namespace mlpack {
namespace nn {

namespace {

void VisualizeHiddenUnit(size_t rows, size_t cols,
                         arma::mat const &input,
                         arma::mat &output)
{
  int const squareRows = static_cast<int>(std::sqrt(input.n_rows));
  int const buf = 1;

  int const offset = squareRows+buf;
  output.ones(buf+rows*(offset),
              buf+cols*(offset));

  int k = 0;
  for(int i = 0; i != rows; ++i) {
    for(int j = 0; j != cols; ++j) {
      if(k >= input.n_cols) {
        continue;
      }
      // Find the maximum element in this row.
      const double max  = arma::max(arma::abs(input.col(k)));
      // Now, copy the elements of the row to the output submatrix.
      const arma::uword minRow = i * offset;
      const arma::uword minCol = j * offset;
      const arma::uword maxRow = i * offset + squareRows - 1;
      const arma::uword maxCol = j * offset + squareRows - 1;
      // Only divide by the max if it's not 0.
      if (max != 0.0)
        output.submat(minRow, minCol, maxRow, maxCol) =
          arma::reshape(input.col(k), squareRows, squareRows) / max;
      else
        output.submat(minRow, minCol, maxRow, maxCol) =
          arma::reshape(input.col(k), squareRows, squareRows);

      ++k;
    }
  }
}

}

void MaximalInputs(arma::mat const &parameters, arma::mat &output)
{
  arma::mat paramTemp(parameters.t());
  double const mean = arma::mean(arma::mean(paramTemp));
  paramTemp -= mean;

  int rows = 0, cols = 0;
  if(std::pow(std::floor(std::sqrt(paramTemp.n_cols)), 2) != paramTemp.n_cols) {
    cols = (int)std::ceil(std::sqrt(paramTemp.n_cols));
    while(paramTemp.n_cols % cols != 0 && cols < 1.2*std::sqrt(paramTemp.n_cols)) {
      ++cols;
    }
    rows = static_cast<int>
           (std::ceil(paramTemp.n_cols/static_cast<double>(cols)));
  }else{
    cols = static_cast<int>(std::sqrt(paramTemp.n_cols));
    rows = cols;
  }

  VisualizeHiddenUnit(rows, cols, paramTemp, output);

  double const max = output.max();
  double const min = output.min();
  if((max - min) != 0) {
    output = (output - min) / (max - min) * 255;
  }
}

} // namespace nn
} // namespace mlpack
