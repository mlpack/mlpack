/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "lrsdp.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace std;

LRSDP::LRSDP(const size_t numConstraints,
             const arma::mat& initialPoint) :
    function(numConstraints, initialPoint),
    augLag(function)
{ }

double LRSDP::Optimize(arma::mat& coordinates)
{
  augLag.Sigma() = 20;
  augLag.Optimize(coordinates, 1000);

  return augLag.Function().Evaluate(coordinates);
}

// Convert the object to a string.
std::string LRSDP::ToString() const
{
  std::ostringstream convert;
  convert << "LRSDP [" << this << "]" << std::endl;
  convert << "  Optimizer: " << util::Indent(augLag.ToString(), 1) << std::endl;
  return convert.str();
}
