/**
 * @file maximal_inputs.cpp
 * @author Tham Ngap Wei
 *
 * Implementation of MaximalInputs().
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
