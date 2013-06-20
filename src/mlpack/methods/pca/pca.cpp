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
                arma::mat& coeffs) const
{
  // Calculate the covariance matrix, given that the data is column-major (this
  // is why we use ccov() and not cov()).
  arma::mat covMat = ccov(data);

  // Centering is built into ccov(), so we don't need to worry about it.  We
  // only need to scale the data if the user asked for it.
  if (scaleData)
  {
    // Scaling the data is when we reduce the variance of each dimension to 1.
    // Normally you might do this by dividing each dimension by its standard
    // deviation, but since we already have the covariance matrix we can
    // simplify the operation into dividing each element C_ij in the covariance
    // matrix by the standard deviation of dimension i multiplied by the
    // standard deviation of dimension j.
    arma::vec stdDev = sqrt(covMat.diag());

    // If there are any zeroes, make them very small.
    for (size_t i = 0; i < stdDev.n_elem; ++i)
      if (stdDev[i] == 0)
        stdDev[i] = 1e-50;

    covMat /= stdDev * trans(stdDev);
  }

  arma::eig_sym(eigVal, coeffs, covMat);

  int nEigVal = eigVal.n_elem;
  for (int i = 0; i < floor(nEigVal / 2.0); i++)
    eigVal.swap_rows(i, (nEigVal - 1) - i);

  coeffs = arma::fliplr(coeffs);
  transformedData = trans(coeffs) * data;
  arma::colvec transformedDataMean = arma::mean(transformedData, 1);
  transformedData = transformedData - (transformedDataMean *
      arma::ones<arma::rowvec>(transformedData.n_cols));
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
