/**
 * @file simple_residue_termination.hpp
 * @author Sumedh Ghaisas
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
#ifndef _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This class implements a simple residue-based termination policy. The
 * termination decision depends on two factors: the value of the residue (the
 * difference between the norm of WH this iteration and the previous iteration),
 * and the number of iterations.  If the current value of residue drops below
 * the threshold or the number of iterations goes above the iteration limit,
 * IsConverged() will return true.  This class is meant for use with the AMF
 * (alternating matrix factorization) class.
 *
 * @see AMF
 */
class SimpleResidueTermination
{
 public:
  /**
   * Construct the SimpleResidueTermination object with the given minimum
   * residue (or the default) and the given maximum number of iterations (or the
   * default).  0 indicates no iteration limit.
   *
   * @param minResidue Minimum residue for termination.
   * @param maxIterations Maximum number of iterations.
   */
  SimpleResidueTermination(const double minResidue = 1e-10,
                           const size_t maxIterations = 10000)
      : minResidue(minResidue), maxIterations(maxIterations) { }

  template<typename MatType>
  void Initialize(const MatType& V)
  {
    // Initialize the things we keep track of.
    residue = DBL_MAX;
    iteration = 1;
    normOld = 0;

    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    nm = n * m;
  }

  /**
   * Check if termination criterion is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // Calculate the norm and compute the residue
    const double norm = arma::norm(W * H, "fro");
    residue = fabs(normOld - norm) / normOld;

    // Store the norm.
    normOld = norm;

    // Increment iteration count
    iteration++;

    // Check if termination criterion is met.
    return (residue < minResidue || iteration > maxIterations);
  }

  const double& Index() { return residue; }
  const size_t& Iteration() { return iteration; }
  const size_t& MaxIterations() { return maxIterations; }

  //! Get current iteration count
  const size_t& Iteration() const { return iteration; }

  //! Access max iteration count
  const size_t& MaxIterations() const { return maxIterations; }
  size_t& MaxIterations() { return maxIterations; }

  //! Access minimum residue value
  const double& MinResidue() const { return minResidue; }
  double& MinResidue() { return minResidue; }

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
