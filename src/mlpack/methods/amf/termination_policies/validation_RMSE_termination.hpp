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
                            size_t maxIterations = 10000)
        : tolerance(tolerance),
          maxIterations(maxIterations),
          num_test_points(num_test_points)
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

  void Initialize(const MatType& V)
  {
    iteration = 1;

    rmse = DBL_MAX;
    rmseOld = DBL_MAX;
    t_count = 0;
  }

  bool IsConverged()
  {
    if((rmseOld - rmse) / rmseOld < tolerance && iteration > 4) t_count++;
    else t_count = 0;

    if(t_count == 3 || iteration > maxIterations) return true;
    else return false;
  }

  void Step(const arma::mat& W, const arma::mat& H)
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
  }

  const double& Index()
  {
    return rmse;
  }

  const size_t& Iteration()
  {
    return iteration;
  }

 private:
  double tolerance;
  size_t maxIterations;
  size_t num_test_points;
  size_t iteration;

  arma::Mat<double> test_points;

  double rmseOld;
  double rmse;

  size_t t_count;
};

} // namespace amf
} // namespace mlpack


#endif // VALIDATION_RMSE_TERMINATION_HPP_INCLUDED

