#ifndef __MLPACK_METHODS_LINEAR_REGRESSCLIN_HPP
#define __MLPACK_METHODS_LINEAR_REGRESSCLIN_HPP

#include <mlpack/core.h>
namespace mlpack {
namespace linear_regression {

/** 
 *  A simple linear regresion algorithm using ordinary least squares.
 */
class LinearRegression
{
  public:
    /** Creates the model.
     *  @param predictors X, matrix of data points to create B with.
     *  @param responses y, the measured data for each point in X
     */ 
    LinearRegression(arma::mat& predictors, const arma::colvec& responses);

    /** Initialize the model from a file.
     *  @param filename the name of the file to load the model from.
     */
    LinearRegression(const std::string& filename);

    /** Destructor - no work done. */
    ~LinearRegression();

    /** Calculate y_i for each data point in points.
     *  @param predictions y, will contain calculated values on completion.
     *  @param points the data points to calculate with.
     */
    void predict(arma::rowvec& predictions, const arma::mat& points);

    /** Returns the model.
     * @return the parameters which describe the least squares solution.
     */
    arma::vec getParameters();

    /** Saves the model.
     *  @param filename the name of the file to load the model from.
     */
    bool save(const std::string& filename);

    /** Loads the model.
     *  @param filename the name of the file to load the model from.
     */
    bool load(const std::string& filename);

  private:
    /** The calculated B.
     * Initialized and filled by constructor to hold the least squares solution.
     */
    arma::vec parameters;

};

}; // namespace linear_regression
}; // namespace mlpack

#endif // __MLPACK_METHODS_LINEAR_REGRESSCLIN_HPP
