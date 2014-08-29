/**
 * @file complete_incremental_termination_hpp
 * @author Sumedh Ghaisas
 *
 * Complete incremental learning termination policy.
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
#ifndef COMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED
#define COMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED

namespace mlpack
{
namespace amf
{

template <class TerminationPolicy>
class CompleteIncrementalTermination
{
 public:
  CompleteIncrementalTermination(TerminationPolicy t_policy = TerminationPolicy())
            : t_policy(t_policy) {}

  template <class MatType>
  void Initialize(const MatType& V)
  {
    t_policy.Initialize(V);

    incrementalIndex = accu(V != 0);
    iteration = 0;
  }

  void Initialize(const arma::sp_mat& V)
  {
    t_policy.Initialize(V);

    incrementalIndex = V.n_nonzero;
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

} // namespace amf
} // namespace mlpack


#endif // COMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED

