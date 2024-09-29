/**
 * @file methods/amf/termination_policies/simple_tolerance_termination.hpp
 * @author Sumedh Ghaisas
 *
 * Termination policy used in AMF (Alternating Matrix Factorization).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class implements residue tolerance termination policy. Termination
 * criterion is met when increase in residue value drops below the given
 * tolerance.  To accommodate spikes certain number of successive residue drops
 * are accepted.  This upper limit on successive drops can be adjusted with
 * reverseStepCount.  Secondary termination criterion terminates algorithm when
 * iteration count goes above the threshold.
 *
 * @see AMF
 */
template<typename MatType, typename WHMatType = arma::mat>
class SimpleToleranceTermination
{
 public:
  //! empty constructor
  SimpleToleranceTermination(const double tolerance = 1e-5,
                             const size_t maxIterations = 10000,
                             const size_t reverseStepTolerance = 3) :
      tolerance(tolerance),
      maxIterations(maxIterations),
      V(nullptr),
      iteration(1),
      residueOld(DBL_MAX),
      residue(DBL_MIN),
      reverseStepTolerance(reverseStepTolerance),
      reverseStepCount(0),
      isCopy(false),
      cIndexOld(0),
      cIndex(0)
  { }

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param V Input matrix to be factorized.
   */
  void Initialize(const MatType& V)
  {
    residueOld = DBL_MAX;
    iteration = 1;
    residue = DBL_MIN;
    reverseStepCount = 0;
    isCopy = false;

    this->V = &V;

    cIndex = 0;
    cIndexOld = 0;
  }

  /**
   * Check if termination criterion is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  bool IsConverged(WHMatType& W, WHMatType& H)
  {
    WHMatType WH = W * H;

    // Compute residue.
    residueOld = residue;
    size_t n = V->n_rows;
    size_t m = V->n_cols;
    double sum = 0;
    size_t count = 0;
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < m; ++j)
      {
        double temp = 0;
        if ((temp = (*V)(i, j)) != 0)
        {
          temp = (temp - WH(i, j));
          temp = temp * temp;
          sum += temp;
          count++;
        }
      }
    }

    residue = sum;
    if (count > 0)
      residue /= count;
    residue = std::sqrt(residue);

    // Increment iteration count.
    iteration++;
    Log::Info << "Iteration " << iteration << "; residue "
        << ((residueOld - residue) / residueOld) << ".\n";

    // If residue tolerance is not satisfied.
    if ((residueOld - residue) / residueOld < tolerance && iteration > 4)
    {
      // Check if this is a first of successive drops.
      if (reverseStepCount == 0 && isCopy == false)
      {
        // Store a copy of W and H matrix.
        isCopy = true;
        this->W = W;
        this->H = H;
        // Store residue values.
        cIndex = residue;
        cIndexOld = residueOld;
      }
      // Increase successive drop count.
      reverseStepCount++;
    }
    // If tolerance is satisfied.
    else
    {
      // Initialize successive drop count.
      reverseStepCount = 0;
      // If residue is droped below minimum scrap stored values.
      if (residue <= cIndexOld && isCopy == true)
      {
        isCopy = false;
      }
    }

    // Check if termination criterion is met.
    if (reverseStepCount == reverseStepTolerance || iteration > maxIterations)
    {
      // If stored values are present replace them with current value as they
      // represent the minimum residue point.
      if (isCopy)
      {
        W = this->W;
        H = this->H;
        residue = cIndex;
      }
      return true;
    }

    return false;
  }

  //! Get current value of residue.
  const double& Index() const { return residue; }

  //! Get current iteration count.
  const size_t& Iteration() const { return iteration; }

  //! Access upper limit of iteration count.
  const size_t& MaxIterations() const { return maxIterations; }
  size_t& MaxIterations() { return maxIterations; }

  //! Access tolerance value.
  const double& Tolerance() const { return tolerance; }
  double& Tolerance() { return tolerance; }

 private:
  //! Locally-stored tolerance.
  double tolerance;
  //! Locally-stored iteration threshold.
  size_t maxIterations;

  //! Pointer to matrix being factorized.
  const MatType* V;

  //! Current iteration count.
  size_t iteration;

  //! Locally-stored residue values.
  double residueOld;
  double residue;

  //! Tolerance on successive residue drops.
  size_t reverseStepTolerance;
  //! Successive residue drops.
  size_t reverseStepCount;

  //! Indicates whether a copy of information is available which corresponds to
  //! minimum residue point.
  bool isCopy;

  //! Variables to store information of minimum residue poi.
  WHMatType W;
  WHMatType H;
  double cIndexOld;
  double cIndex;
}; // class SimpleToleranceTermination

} // namespace mlpack

#endif // _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED

