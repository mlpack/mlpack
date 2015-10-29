#include "maximal_inputs.hpp"

namespace mlpack {
namespace nn {

namespace {

void VisualizeHiddenUnit(arma::uword rows, arma::uword cols,
                         const arma::mat &input,
                         arma::mat &output)
{
  arma::uword const squareRows = static_cast<arma::uword>(std::sqrt(input.n_rows));
  arma::uword const buf = 1;

  arma::uword const offset = squareRows+buf;
  output.ones(buf+rows*(offset),
              buf+cols*(offset));

  arma::uword k = 0;
  for(arma::uword i = 0; i != rows; ++i) {
    for(arma::uword j = 0; j != cols; ++j) {
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

void MaximalInputs(const arma::mat &parameters, arma::mat &output, double minRange, double maxRange)
{
  arma::mat paramTemp(parameters.submat(0, 0, (parameters.n_rows-1)/2 - 1,
                                        parameters.n_cols - 2).t());
  double const mean = arma::mean(arma::mean(paramTemp));
  paramTemp -= mean;

  arma::uword rows = 0, cols = 0;
  if(std::pow(std::floor(std::sqrt(paramTemp.n_cols)), 2) != paramTemp.n_cols) {
    cols = static_cast<arma::uword>(std::ceil(std::sqrt(paramTemp.n_cols)));
    while(paramTemp.n_cols % cols != 0 && cols < 1.2*std::sqrt(paramTemp.n_cols)) {
      ++cols;
    }
    rows = static_cast<arma::uword>
           (std::ceil(paramTemp.n_cols/static_cast<double>(cols)));
  }else{
    cols = static_cast<arma::uword>(std::sqrt(paramTemp.n_cols));
    rows = cols;
  }

  VisualizeHiddenUnit(rows, cols, paramTemp, output);

  double const max = output.max();
  double const min = output.min();
  if((max - min) != 0) {
    output = (output - min) / (max - min) * (maxRange - minRange) + minRange;
  }
}

} // namespace nn
} // namespace mlpack
