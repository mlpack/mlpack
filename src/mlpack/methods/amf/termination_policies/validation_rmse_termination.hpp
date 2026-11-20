/**
 * @file methods/amf/termination_policies/validation_rmse_termination.hpp
 * @author Sumedh Ghaisas
 *
 * Termination policy used in AMF (Alternating Matrix Factorization).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _MLPACK_METHODS_AMF_VALIDATIONRMSETERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_VALIDATIONRMSETERMINATION_HPP_INCLUDED

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class implements validation termination policy based on RMSE index.
 * The input data matrix is divided into 2 sets, training set and validation
 * set.
 *
 * Entries of validation set are nullifed in the input matrix. Termination
 * criterion is met when increase in validation set RMSe value drops below the
 * given tolerance. To accommodate spikes certain number of successive
 * validation RMSE drops are accepted. This upper imit on successive drops can
 * be adjusted with reverseStepCount. Secondary termination criterion terminates
 * algorithm when iteration count goes above the threshold.
 *
 * @note The input matrix is modified by this termination policy.
 *
 * @see AMF
 */
template<typename MatType, typename WHMatType = arma::mat>
class ValidationRMSETermination
{
 public:
  /**
   * Create a validation set according to given parameters and nullifies this
   * set in data matrix(training set).
   *
   * @param V Input matrix to be factorized.
   * @param numTestPoints number of validation test points
   * @param tolerance the tolerance value to compare RMSe against
   * @param maxIterations max iteration count before termination
   * @param reverseStepTolerance max successive RMSE drops allowed
   */
  ValidationRMSETermination(MatType& V,
                            size_t numTestPoints,
                            double tolerance = 1e-5,
                            size_t maxIterations = 10000,
                            size_t reverseStepTolerance = 3)
        : tolerance(tolerance),
          maxIterations(maxIterations),
          numTestPoints(numTestPoints),
          reverseStepTolerance(reverseStepTolerance)
  {
    size_t n = V.n_rows;
    size_t m = V.n_cols;

    // initialize validation set matrix
    testPoints.zeros(numTestPoints, 3);

    // fill validation set matrix with random chosen entries
    for (size_t i = 0; i < numTestPoints; ++i)
    {
      double tVal;
      size_t tRow;
      size_t tCol;

      // pick a random non-zero entry
      do
      {
        tRow = RandInt(n);
        tCol = RandInt(m);
      } while ((tVal = V(tRow, tCol)) == 0);

      // add the entry to the validation set
      testPoints(i, 0) = tRow;
      testPoints(i, 1) = tCol;
      testPoints(i, 2) = tVal;

      // nullify the added entry from data matrix (training set)
      V(tRow, tCol) = 0;
    }
  }

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param * (V) Input matrix to be factorized.
   */
  void Initialize(const MatType& /* V */)
  {
    iteration = 1;

    rmse = DBL_MAX;
    rmseOld = DBL_MAX;

    cIndex = 0;
    cIndexOld = 0;

    reverseStepCount = 0;
    isCopy = false;
  }

  /**
   * Check if termination criterio is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  bool IsConverged(WHMatType& W, WHMatType& H)
  {
    WHMatType WH = W * H;

    // compute validation RMSE
    if (iteration != 0)
    {
      rmseOld = rmse;
      rmse = 0;
      for (size_t i = 0; i < numTestPoints; ++i)
      {
        size_t tRow = testPoints(i, 0);
        size_t tCol = testPoints(i, 1);
        double tVal = testPoints(i, 2);
        double temp = (tVal - WH(tRow, tCol));
        temp *= temp;
        rmse += temp;
      }
      rmse /= numTestPoints;
      rmse = std::sqrt(rmse);
    }

    // increment iteration count
    iteration++;

    // if RMSE tolerance is not satisfied
    if ((rmseOld - rmse) / rmseOld < tolerance && iteration > 4)
    {
      // check if this is a first of successive drops
      if (reverseStepCount == 0 && isCopy == false)
      {
        // store a copy of W and H matrix
        isCopy = true;
        this->W = W;
        this->H = H;
        // store residue values
        cIndexOld = rmseOld;
        cIndex = rmse;
      }
      // increase successive drop count
      reverseStepCount++;
    }
    // if tolerance is satisfied
    else
    {
      // initialize successive drop count
      reverseStepCount = 0;
      // if residue is droped below minimum scrap stored values
      if (rmse <= cIndexOld && isCopy == true)
      {
        isCopy = false;
      }
    }

    // check if termination criterion is met
    if (reverseStepCount == reverseStepTolerance || iteration > maxIterations)
    {
      // if stored values are present replace them with current value as they
      // represent the minimum residue point
      if (isCopy)
      {
        W = this->W;
        H = this->H;
        rmse = cIndex;
      }
      return true;
    }
    else
    {
      return false;
    }
  }

  //! Get current value of residue
  const double& Index() const { return rmse; }
  const double& RMSE() const { return rmse; }

  //! Get current iteration count
  const size_t& Iteration() const { return iteration; }

  //! Get number of validation points
  const size_t& NumTestPoints() const { return numTestPoints; }

  //! Access upper limit of iteration count
  const size_t& MaxIterations() const { return maxIterations; }
  size_t& MaxIterations() { return maxIterations; }

  //! Access tolerance value
  const double& Tolerance() const { return tolerance; }
  double& Tolerance() { return tolerance; }

 private:
  //! tolerance
  double tolerance;
  //! max iteration limit
  size_t maxIterations;
  //! number of validation test points
  size_t numTestPoints;

  //! current iteration count
  size_t iteration;

  //! validation point matrix
  WHMatType testPoints;

  //! rmse values
  double rmseOld;
  double rmse;

  //! tolerance on successive residue drops
  size_t reverseStepTolerance;
  //! successive residue drops
  size_t reverseStepCount;

  //! indicates whether a copy of information is available which corresponds to
  //! minimum residue point
  bool isCopy;

  //! variables to store information of minimum residue point
  WHMatType W;
  WHMatType H;
  double cIndexOld;
  double cIndex;
}; // class ValidationRMSETermination

} // namespace mlpack


#endif // _MLPACK_METHODS_AMF_VALIDATIONRMSETERMINATION_HPP_INCLUDED
