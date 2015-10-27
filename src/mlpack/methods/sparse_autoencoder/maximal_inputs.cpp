#include "maximal_inputs.hpp"

namespace mlpack {
namespace nn {

namespace {

void VisualizeHiddenUnit(size_t rows, size_t cols,
                         int squareRows,
                         int offset,
                         arma::mat const &input,
                         arma::mat &output)
{
  int k = 0;
  for(int i = 0; i != cols; ++i)
  {
    for(int j = 0; j != rows; ++j)
    {
      if(k >= input.n_cols)
      {
        continue;
      }
      arma::mat reshapeMat(squareRows, squareRows);
      arma::mat const weights = input.row(k);
      std::copy(std::begin(weights),
                std::end(weights),
                std::begin(reshapeMat));
      double const max = arma::abs(input.row(k)).max();
      if(max != 0.0)
      {
        reshapeMat /= max;
      }
      output.submat(i*(offset), j*(offset),
                    i*(offset) + squareRows - 1,
                    j*(offset) + squareRows - 1) = reshapeMat;
      ++k;
    }
  }
}

}

void MaximalInputs(arma::mat const &parameters, arma::mat &output)
{
  //take the encoder part of the paramters
  arma::mat paramTemp = parameters.submat(0, 0, (parameters.n_rows-1)/2-1, parameters.n_cols-2);
  double const mean = arma::mean(arma::mean(paramTemp));
  paramTemp -= mean;

  int rows = 0, cols = (int)std::ceil(std::sqrt(paramTemp.n_rows));
  if(std::pow(std::floor(std::sqrt(paramTemp.n_rows)), 2) != paramTemp.n_rows)
  {
    while(paramTemp.n_rows % cols != 0 && cols < 1.2*std::sqrt(paramTemp.n_rows))
    {
      ++cols;
    }
    rows = (int)std::ceil(paramTemp.n_rows/cols);
  }else
  {
    cols = (int)std::sqrt(paramTemp.n_rows);
    rows = cols;
  }

  int const squareRows = (int)std::sqrt(paramTemp.n_cols);
  int const buf = 1;

  int const offset = squareRows+buf;
  output.ones(buf+rows*(offset),
              buf+cols*(offset));

  VisualizeHiddenUnit(rows, cols, squareRows,
                      offset, paramTemp, output);

  double const max = output.max();
  double const min = output.min();
  if((max - min) != 0)
  {
    output = (output - min) / (max - min) * 255;
  }
}

} // namespace nn
} // namespace mlpack
