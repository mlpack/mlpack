#ifndef MLPACK_LINEAR_REGRESSION_H
#define MLPACK_LINEAR_REGRESSION_H

#include <mlpack/core.h>
namespace mlpack {
  namespace linear_regression {

/** 
 *  A simple linear regresion algorithm using ordinary least squares.
 */
class LinearRegression {
  public:
    /** Create and initialize parameters. y=BX, this creates B.
     *  @param predictors X, matrix of data points to create B with.
     *  @ param responses y, the measured data for each point in X
     */ 
    LinearRegression(arma::mat& predictors, const arma::colvec& responses);

    /** Destructor - no work done.
     */
    ~LinearRegression();

    /** Calculate y_i for each data point in points.
     *  @param predictions y, will contain calculated values on completion.
     *  @param points the data points to calculate with.
     */
    void predict(arma::rowvec& predictions, const arma::mat& points);

    /** Returns B.
     * @return the parameters which describe the least squares solution.
     */
    arma::vec getParameters();

  private:
    /** The calculated B.
     * Initialized and filled by constructor to hold the least squares solution.
     */
    arma::vec parameters;

};

  }
}


#endif
