/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
