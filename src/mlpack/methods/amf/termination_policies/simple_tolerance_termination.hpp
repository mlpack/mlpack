/**
 * @file simple_tolerance_termination.hpp
 * @author Sumedh Ghaisas
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
                             const size_t maxIterations = 10000)
            : tolerance(tolerance), maxIterations(maxIterations) {}

  void Initialize(const MatType& V)
  {
    residueOld = DBL_MAX;
    iteration = 1;
    residue = DBL_MIN;

    this->V = &V;
  }

  bool IsConverged()
  {
    if(((residueOld - residue) / residueOld < tolerance && iteration > 4)
        || iteration > maxIterations) return true;
    else return false;
  }

  void Step(const arma::mat& W, const arma::mat& H)
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
  }

  const double& Index() { return residue; }
  const size_t& Iteration() { return iteration; }

 private:
  double tolerance;
  size_t maxIterations;

  const MatType* V;

  size_t iteration;
  double residueOld;
  double residue;
  double normOld;
}; // class SimpleToleranceTermination

}; // namespace amf
}; // namespace mlpack

#endif // _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED

