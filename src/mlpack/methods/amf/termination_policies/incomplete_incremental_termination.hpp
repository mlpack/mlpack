/**
 * @file incomplete_incremental_termination.hpp
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
#ifndef _INCOMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED
#define _INCOMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

template <class TerminationPolicy>
class IncompleteIncrementalTermination
{
 public:
  IncompleteIncrementalTermination(TerminationPolicy t_policy = TerminationPolicy())
            : t_policy(t_policy) {}

  template <class MatType>
  void Initialize(const MatType& V)
  {
    t_policy.Initialize(V);

    incrementalIndex = V.n_rows;
    iteration = 0;
  }

  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    iteration++;
    if(iteration % incrementalIndex == 0)  
      return t_policy.IsConverged(W, H);
    else return false;
  }

  const double& Index()
  {
    return t_policy.Index();
  }
  const size_t& Iteration()
  {
    return iteration;
  }
  const size_t& MaxIterations()
  {
    return t_policy.MaxIterations();
  }
  
 private:
  TerminationPolicy t_policy;

  size_t incrementalIndex;
  size_t iteration;
};

}; // namespace amf
}; // namespace mlpack

#endif

