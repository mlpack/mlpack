/**
 * @file simple_tolerance_termination.hpp
 * @author Sumedh Ghaisas
 */
#ifndef _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

class SimpleToleranceTermination
{
 public:
  SimpleToleranceTermination(const double tolerance = 1e-5,
                             const size_t maxIterations = 10000)
            : tolerance(tolerance), maxIterations(maxIterations) {}

  template<typename MatType>
  void Initialize(MatType& V)
  {
    residueOld = DBL_MAX;
    iteration = 1;
    normOld = 0;
    residue = DBL_MIN;

    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    nm = n * m;
  }

  bool IsConverged()
  {
    if(((residueOld - residue) / residueOld < tolerance && iteration > 4) 
        || iteration > maxIterations) return true;
    else return false;
  }

  template<typename MatType>
  void Step(const MatType& W, const MatType& H)
  {
    // Calculate norm of WH after each iteration.
    arma::mat WH;

    WH = W * H;
    double norm = sqrt(accu(WH % WH) / nm);

    if (iteration != 0)
    {
      residueOld = residue;
      residue = fabs(normOld - norm);
      residue /= normOld;
    }

    normOld = norm;

    iteration++;
  }

  const double& Index() { return residue; }
  const size_t& Iteration() { return iteration; }

 private:
  double tolerance;
  size_t maxIterations;

  size_t iteration;
  double residueOld;
  double residue;
  double normOld;

  size_t nm;
}; // class SimpleToleranceTermination

}; // namespace amf
}; // namespace mlpack

#endif // _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED
