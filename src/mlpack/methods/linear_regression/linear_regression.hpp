#ifndef __MLPACK_METHODS_LINEAR_REGRESSCLIN_HPP
#define __MLPACK_METHODS_LINEAR_REGRESSCLIN_HPP

#include <mlpack/core.h>
namespace mlpack {
namespace linear_regression {

/** 
 *  A simple linear regresion algorithm using ordinary least squares.
 */
class LinearRegression {
  public:
    /** Initialize parameters. 
     *  @param predictors X, matrix of data points to create B with.
     *  @ param responses y, the measured data for each point in X
     */ 
    LinearRegression(arma::mat& predictors, const arma::colvec& responses);

    /** Destructor - no work done.
     */
    ~LinearRegression();

    /** Create regression coefficients.
     * y=BX, this creates B.
     **/
    void run();

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

    /** The X values.
     * The regressor values.
     */
    arma::mat& predictors;

    /** The y values for the predictors.
     * The response variables, corresponding to points in X.
     */
    const arma::colvec& responses;

};

}; // namespace linear_regression
}; // namespace mlpack

#endif // __MLPACK_METHODS_LINEAR_REGRESSCLIN_HPP
