/**
 * @file methods/sparse_autoencoder/maximal_inputs_impl.hpp
 * @author Tham Ngap Wei
 *
 * Implementation of MaximalInputs().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NN_MAXIMAL_INPUTS_IMPL_HPP
#define MLPACK_METHODS_NN_MAXIMAL_INPUTS_IMPL_HPP

#include "maximal_inputs.hpp"

namespace mlpack {

inline void MaximalInputs(const arma::mat& parameters, arma::mat& output)
{
  arma::mat paramTemp(parameters.submat(0, 0, (parameters.n_rows - 1) / 2 - 1,
                                        parameters.n_cols - 2).t());
  paramTemp -= arma::mean(arma::mean(paramTemp));

  NormalizeColByMax(paramTemp, output);
}

inline void NormalizeColByMax(const arma::mat &input,
                              arma::mat &output)
{
  output.set_size(input.n_rows, input.n_cols);
  for (arma::uword i = 0; i != input.n_cols; ++i)
  {
    const double colMax = max(arma::abs(input.col(i)));
    if (colMax != 0.0)
    {
      output.col(i) = input.col(i) / colMax;
    }
    else
    {
      output.col(i) = input.col(i);
    }
  }
}

} // namespace mlpack

#endif
