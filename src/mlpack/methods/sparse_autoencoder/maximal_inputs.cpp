/**
 * @file maximal_inputs.cpp
 * @author Tham Ngap Wei
 *
 * Implementation of MaximalInputs().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "maximal_inputs.hpp"

namespace mlpack {
namespace nn {

void MaximalInputs(const arma::mat& parameters, arma::mat& output)
{
  arma::mat paramTemp(parameters.submat(0, 0, (parameters.n_rows - 1) / 2 - 1,
                                        parameters.n_cols - 2).t());
  double const mean = arma::mean(arma::mean(paramTemp));
  paramTemp -= mean;

  NormalizeColByMax(paramTemp, output);
}

void NormalizeColByMax(const arma::mat &input,
                       arma::mat &output)
{
  output.set_size(input.n_rows, input.n_cols);
  for (arma::uword i = 0; i != input.n_cols; ++i)
  {
    const double max = arma::max(arma::abs(input.col(i)));
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

} // namespace nn
} // namespace mlpack
