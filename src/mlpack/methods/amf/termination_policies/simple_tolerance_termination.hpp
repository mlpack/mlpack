/**
 * @file simple_tolerance_termination.hpp
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
#ifndef _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

template <class MatType>
class SimpleToleranceTermination
{
 public:
  SimpleToleranceTermination(const double tolerance = 1e-5,
                             const size_t maxIterations = 10000,
                             const size_t reverseStepTolerance = 3)
            : tolerance(tolerance),
              maxIterations(maxIterations),
              reverseStepTolerance(reverseStepTolerance) {}

  void Initialize(const MatType& V)
  {
    residueOld = DBL_MAX;
    iteration = 1;
    residue = DBL_MIN;
    reverseStepCount = 0;

    this->V = &V;
    
    c_index = 0;
    c_indexOld = 0;

    reverseStepCount = 0;
  }

  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // Calculate norm of WH after each iteration.
    arma::mat WH;

    WH = W * H;

    residueOld = residue;
    size_t n = V->n_rows;
    size_t m = V->n_cols;
    double sum = 0;
    size_t count = 0;
    for(size_t i = 0;i < n;i++)
    {
        for(size_t j = 0;j < m;j++)
        {
            double temp = 0;
            if((temp = (*V)(i,j)) != 0)
            {
                temp = (temp - WH(i, j));
                temp = temp * temp;
                sum += temp;
                count++;
            }
        }
    }
    residue = sum / count;
    residue = sqrt(residue);

    iteration++;  
  
    if((residueOld - residue) / residueOld < tolerance && iteration > 4)
    {
      if(reverseStepCount == 0 && isCopy == false)
      {
        isCopy = true;
        this->W = W;
        this->H = H;
        c_index = residue;
        c_indexOld = residueOld;
      }
      reverseStepCount++;
    }
    else
    {
      reverseStepCount = 0;
      if(residue <= c_indexOld && isCopy == true)
      {
        isCopy = false;
      }
    }

    if(reverseStepCount == reverseStepTolerance || iteration > maxIterations)
    {
      if(isCopy)
      {
        W = this->W;
        H = this->H;
        residue = c_index;
      }
      return true;
    }
    else return false;
  }

  const double& Index() { return residue; }
  const size_t& Iteration() { return iteration; }
  const size_t& MaxIterations() { return maxIterations; }

 private:
  double tolerance;
  size_t maxIterations;

  const MatType* V;

  size_t iteration;
  double residueOld;
  double residue;
  double normOld;

  size_t reverseStepTolerance;
  size_t reverseStepCount;
  
  bool isCopy;
  arma::mat W;
  arma::mat H;
  double c_indexOld;
  double c_index;
}; // class SimpleToleranceTermination

}; // namespace amf
}; // namespace mlpack

#endif // _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED

