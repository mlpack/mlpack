/**
 * @file validation_RMSE_termination.hpp
 * @author Sumedh Ghaisas
 *
 * Termination policy that checks validation RMSE.
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
#ifndef VALIDATION_RMSE_TERMINATION_HPP_INCLUDED
#define VALIDATION_RMSE_TERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack
{
namespace amf
{
template <class MatType>
class ValidationRMSETermination
{
 public:
  ValidationRMSETermination(MatType& V,
                            size_t num_test_points,
                            double tolerance = 1e-5,
                            size_t maxIterations = 10000,
                            size_t reverseStepTolerance = 3)
        : tolerance(tolerance),
          maxIterations(maxIterations),
          num_test_points(num_test_points),
          reverseStepTolerance(reverseStepTolerance)
  {
    size_t n = V.n_rows;
    size_t m = V.n_cols;

    test_points.zeros(num_test_points, 3);

    for(size_t i = 0; i < num_test_points; i++)
    {
      double t_val;
      size_t t_row;
      size_t t_col;
      do
      {
        t_row = rand() % n;
        t_col = rand() % m;
      } while((t_val = V(t_row, t_col)) == 0);

      test_points(i, 0) = t_row;
      test_points(i, 1) = t_col;
      test_points(i, 2) = t_val;
      V(t_row, t_col) = 0;
    }
  }

  void Initialize(const MatType& /* V */)
  {
    iteration = 1;

    rmse = DBL_MAX;
    rmseOld = DBL_MAX;

    c_index = 0;
    c_indexOld = 0;

    reverseStepCount = 0;
    isCopy = false;
  }

  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // Calculate norm of WH after each iteration.
    arma::mat WH;

    WH = W * H;

    if (iteration != 0)
    {
      rmseOld = rmse;
      rmse = 0;
      for(size_t i = 0; i < num_test_points; i++)
      {
        size_t t_row = test_points(i, 0);
        size_t t_col = test_points(i, 1);
        double t_val = test_points(i, 2);
        double temp = (t_val - WH(t_row, t_col));
        temp *= temp;
        rmse += temp;
      }
      rmse /= num_test_points;
      rmse = sqrt(rmse);
    }

    iteration++;
  
    if((rmseOld - rmse) / rmseOld < tolerance && iteration > 4)
    {
      if(reverseStepCount == 0 && isCopy == false)
      {
        isCopy = true;
        this->W = W;
        this->H = H;
        c_indexOld = rmseOld;
        c_index = rmse;
      }
      reverseStepCount++;
    }
    else
    {
      reverseStepCount = 0;
      if(rmse <= c_indexOld && isCopy == true)
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
        rmse = c_index;
      }
      return true;
    }
    else return false;
  }
  
  const double& Index() { return rmse; }

  const size_t& Iteration() { return iteration; }
  
  const size_t& MaxIterations() { return maxIterations; }

 private:
  double tolerance;
  size_t maxIterations;
  size_t num_test_points;
  size_t iteration;

  arma::Mat<double> test_points;

  double rmseOld;
  double rmse;

  size_t reverseStepTolerance;
  size_t reverseStepCount;
  
  bool isCopy;
  arma::mat W;
  arma::mat H;
  double c_indexOld;
  double c_index;
};

} // namespace amf
} // namespace mlpack


#endif // VALIDATION_RMSE_TERMINATION_HPP_INCLUDED

