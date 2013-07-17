/**
 * @file pca.cpp
 * @author Ajinkya Kale
 *
 * Implementation of PCA class to perform Principal Components Analysis on the
 * specified data set.
 */
#include "pca.hpp"
#include <mlpack/core.hpp>
#include <iostream>
#include <complex>

using namespace std;
using namespace mlpack;
using namespace mlpack::pca;

PCA::PCA(const bool scaleData) :
    scaleData(scaleData)
{ }

/**
 * Apply Principal Component Analysis to the provided data set.
 *
 * @param data - Data matrix
 * @param transformedData - Data with PCA applied
 * @param eigVal - contains eigen values in a column vector
 * @param coeff - PCA Loadings/Coeffs/EigenVectors
 */
void PCA::Apply(const arma::mat& data,
                arma::mat& transformedData,
                arma::vec& eigVal,
                arma::mat& coeff) const
{
  // This matrix will store the right singular values; we do not need them.
  arma::mat v;

  // Center the data into a temporary matrix.
  arma::mat centeredData;
  math::Center(data, centeredData);

  if (scaleData)
  {
    // Scaling the data is when we reduce the variance of each dimension to 1.
    // We do this by dividing each dimension by its standard deviation.
    arma::vec stdDev = arma::stddev(centeredData, 0, 1 /* for each dimension */);

    // If there are any zeroes, make them very small.
    for (size_t i = 0; i < stdDev.n_elem; ++i)
      if (stdDev[i] == 0)
        stdDev[i] = 1e-50;

    centeredData /= arma::repmat(stdDev, 1, centeredData.n_cols);
  }

  // Do singular value decomposition.  Use the economical singular value
  // decomposition if the columns are much larger than the rows.
  if (data.n_rows < data.n_cols)
  {
    // Do economical singular value decomposition and compute only the left
    // singular vectors.
    arma::svd_econ(coeff, eigVal, v, centeredData, 'l');
  }
  else
  {
    arma::svd(coeff, eigVal, v, centeredData);
  }

  // Now we must square the singular values to get the eigenvalues.
  // In addition we must divide by the number of points, because the covariance
  // matrix is X * X' / (N - 1).
  eigVal %= eigVal / (data.n_cols - 1);

  // Project the samples to the principals.
  transformedData = arma::trans(coeff) * centeredData;
}

/**
 * Apply Principal Component Analysis to the provided data set.
 *
 * @param data - Data matrix
 * @param transformedData - Data with PCA applied
 * @param eigVal - contains eigen values in a column vector
 */
void PCA::Apply(const arma::mat& data,
                arma::mat& transformedData,
                arma::vec& eigVal) const
{
  arma::mat coeffs;
  Apply(data, transformedData, eigVal, coeffs);
}

/**
 * Apply Dimensionality Reduction using Principal Component Analysis
 * to the provided data set.
 *
 * @param data - M x N Data matrix
 * @param newDimension - matrix consisting of N column vectors,
 * where each vector is the projection of the corresponding data vector
 * from data matrix onto the basis vectors contained in the columns of
 * coeff/eigen vector matrix with only newDimension number of columns chosen.
 */
void PCA::Apply(arma::mat& data, const size_t newDimension) const
{
  arma::mat coeffs;
  arma::vec eigVal;

  Apply(data, data, eigVal, coeffs);

  if (newDimension < coeffs.n_rows && newDimension > 0)
    data.shed_rows(newDimension, data.n_rows - 1);
}
