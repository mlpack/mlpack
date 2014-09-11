/**
 * @file pca.cpp
 * @author Ajinkya Kale
 *
 * Implementation of PCA class to perform Principal Components Analysis on the
 * specified data set.
 */
#include "pca.hpp"
#include <mlpack/core.hpp>

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
  Timer::Start("pca");

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

  Timer::Stop("pca");
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
 * Use PCA for dimensionality reduction on the given dataset.  This will save
 * the newDimension largest principal components of the data and remove the
 * rest.  The parameter returned is the amount of variance of the data that is
 * retained; this is a value between 0 and 1.  For instance, a value of 0.9
 * indicates that 90% of the variance present in the data was retained.
 *
 * @param data Data matrix.
 * @param newDimension New dimension of the data.
 * @return Amount of the variance of the data retained (between 0 and 1).
 */
double PCA::Apply(arma::mat& data, const size_t newDimension) const
{
  // Parameter validation.
  if (newDimension == 0)
    Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
        << "be zero!" << endl;
  if (newDimension > data.n_rows)
    Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
        << "be greater than the existing dimensionality of the data ("
        << data.n_rows << ")!" << endl;

  arma::mat coeffs;
  arma::vec eigVal;

  Apply(data, data, eigVal, coeffs);

  if (newDimension < coeffs.n_rows)
    // Drop unnecessary rows.
    data.shed_rows(newDimension, data.n_rows - 1);

  // The svd method returns only non-zero eigenvalues so we have to calculate
  // the right dimension before calculating the amount of variance retained.
  double eigDim = std::min(newDimension - 1, (size_t) eigVal.n_elem - 1);

  // Calculate the total amount of variance retained.
  return (sum(eigVal.subvec(0, eigDim)) / sum(eigVal));
}

/**
 * Use PCA for dimensionality reduction on the given dataset.  This will save
 * as many dimensions as necessary to retain at least the given amount of
 * variance (specified by parameter varRetained).  The amount should be
 * between 0 and 1; if the amount is 0, then only 1 dimension will be
 * retained.  If the amount is 1, then all dimensions will be retained.
 *
 * The method returns the actual amount of variance retained, which will
 * always be greater than or equal to the varRetained parameter.
 */
double PCA::Apply(arma::mat& data, const double varRetained) const
{
  // Parameter validation.
  if (varRetained < 0)
    Log::Fatal << "PCA::Apply(): varRetained (" << varRetained << ") must be "
        << "greater than or equal to 0." << endl;
  if (varRetained > 1)
    Log::Fatal << "PCA::Apply(): varRetained (" << varRetained << ") should be "
        << "less than or equal to 1." << endl;

  arma::mat coeffs;
  arma::vec eigVal;

  Apply(data, data, eigVal, coeffs);

  // Calculate the dimension we should keep.
  size_t newDimension = 0;
  double varSum = 0.0;
  eigVal /= arma::sum(eigVal); // Normalize eigenvalues.
  while ((varSum < varRetained) && (newDimension < eigVal.n_elem))
  {
    varSum += eigVal[newDimension];
    ++newDimension;
  }

  // varSum is the actual variance we will retain.
  if (newDimension < eigVal.n_elem)
    data.shed_rows(newDimension, data.n_rows - 1);

  return varSum;
}

// return a string of this object.
std::string PCA::ToString() const
{
  std::ostringstream convert;
  convert << "Principal Component Analysis  [" << this << "]" << std::endl;
  if (scaleData)  
    convert << "  Scaling Data: TRUE" << std::endl;
  return convert.str();
}
