/**
 * @file pca.cpp
 * @author Ajinkya Kale
 *
 * Implementation of PCA class to perform Principal Components Analysis on the
 * specified data set.
 *
 * This file is part of MLPACK 1.0.3.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "pca.hpp"
#include <mlpack/core.hpp>
#include <iostream>

using namespace std;
namespace mlpack {
namespace pca {

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
	//Original transpose op goes here.
  arma::mat covMat = ccov(data);

	//Centering is built into ccov
	if (scaleData)
  {
    covMat = covMat / (arma::ones<arma::colvec>(covMat.n_rows))
      * stddev(covMat, 0, 0);
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

PCA::~PCA()
{
}

}; // namespace mlpack
}; // namespace pca
