/**
 * @file simple_residue_termination.hpp
 * @author Sumedh Ghaisas
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

class SimpleResidueTermination
{
 public:
  SimpleResidueTermination(const double minResidue = 1e-10,
                           const size_t maxIterations = 10000)
        : minResidue(minResidue), maxIterations(maxIterations) { }

  template<typename MatType>
  void Initialize(const MatType& V)
  {
    residue = minResidue;
    iteration = 1;
    normOld = 0;

    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    nm = n * m;
  }

  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // Calculate norm of WH after each iteration.
    arma::mat WH;

    WH = W * H;
    double norm = sqrt(accu(WH % WH) / nm);

    if (iteration != 0)
    {
      residue = fabs(normOld - norm);
      residue /= normOld;
    }

    normOld = norm;

    iteration++;
    
    if(residue < minResidue || iteration > maxIterations) return true;
    else return false;
  }

  const double& Index() { return residue; }
  const size_t& Iteration() { return iteration; }
  const size_t& MaxIterations() { return maxIterations; }

public:
  double minResidue;
  size_t maxIterations;

  double residue;
  size_t iteration;
  double normOld;

  size_t nm;
}; // class SimpleResidueTermination

}; // namespace amf
}; // namespace mlpack


#endif // _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
