/**
 * @file validation_RMSE_termination.hpp
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

#include <mlpack/core.hpp>

namespace mlpack
{
namespace amf
{

/**
 * This class implements validation termination policy based on RMSE index.
 * The input data matrix is divided into 2 sets, training set and validation set.
 * Entries of validation set are nullifed in the input matrix. Termination
 * criterion is met when increase in validation set RMSe value drops below the
 * given tolerance. To accommodate spikes certain number of successive validation
 * RMSE drops are accepted. This upper imit on successive drops can be adjusted
 * with reverseStepCount. Secondary termination criterion terminates algorithm
 * when iteration count goes above the threshold.
 *
 * @note The input matrix is modified by this termination policy.
 *
 * @see AMF
 */
template <class MatType>
class ValidationRMSETermination
{
 public:
  /**
   * Create a validation set according to given parameters and nullifies this
   * set in data matrix(training set).
   *
   * @param V Input matrix to be factorized.
   * @param num_test_points number of validation test points
   * @param maxIterations max iteration count before termination
   * @param reverseStepTolerance max successive RMSE drops allowed
   */
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

    // initialize validation set matrix
    test_points.zeros(num_test_points, 3);

    // fill validation set matrix with random chosen entries
    for(size_t i = 0; i < num_test_points; i++)
    {
      double t_val;
      size_t t_row;
      size_t t_col;

      // pick a random non-zero entry
      do
      {
        t_row = rand() % n;
        t_col = rand() % m;
      } while((t_val = V(t_row, t_col)) == 0);

      // add the entry to the validation set
      test_points(i, 0) = t_row;
      test_points(i, 1) = t_col;
      test_points(i, 2) = t_val;

      // nullify the added entry from data matrix (training set)
      V(t_row, t_col) = 0;
    }
  }

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param V Input matrix to be factorized.
   */
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

  /**
   * Check if termination criterio is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    arma::mat WH;

    WH = W * H;

    // compute validation RMSE
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
        c_indexOld = rmseOld;
        c_index = rmse;
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
      if (rmse <= c_indexOld && isCopy == true)
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
        rmse = c_index;
      }
      return true;
    }
    else return false;
  }

  //! Get current value of residue
  const double& Index() const { return rmse; }

  //! Get current iteration count
  const size_t& Iteration() const { return iteration; }

  //! Get number of validation points
  const size_t& NumTestPoints() const { return num_test_points; }

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
  size_t num_test_points;

  //! current iteration count
  size_t iteration;

  //! validation point matrix
  arma::mat test_points;

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
  arma::mat W;
  arma::mat H;
  double c_indexOld;
  double c_index;
}; // class ValidationRMSETermination

} // namespace amf
} // namespace mlpack


#endif // _MLPACK_METHODS_AMF_VALIDATIONRMSETERMINATION_HPP_INCLUDED
