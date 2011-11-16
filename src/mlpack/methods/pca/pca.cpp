/**
 * @file pca.cpp
 *
 * Implementation of PCA class to perform Principal Components Analysis on the
 * specified data set.
 */
#include "pca.hpp"
#include <mlpack/core.h>
#include <iostream>

using namespace std;
namespace mlpack {
namespace pca {

PCA::PCA()
{
}

void PCA::Apply(const arma::mat& data, arma::mat& transformedData,
           arma::vec& eigVal, arma::mat& coeffs)
{
  arma::mat transData = trans(data);
  arma::vec means = mean(data, 1);
  arma::mat meanSubData = data - means * arma::ones<arma::rowvec>(data.n_cols);

  arma::mat covMat = ccov(meanSubData);
  arma::eig_sym(eigVal, coeffs, covMat);

  int n_eigVal = eigVal.n_elem;
  for(int i = 0; i < floor(n_eigVal/2); i++)
    eigVal.swap_rows(i, (n_eigVal-1)-i);

  coeffs = arma::fliplr(coeffs);

  transformedData = trans(coeffs) * data;
}

/**
 *
 */
void PCA::Apply(const arma::mat& data, arma::mat& transformedData,
           arma::vec& eigVal)
{
  arma::mat transData = trans(data);
  arma::vec means = mean(data, 1);
  arma::mat meanSubData = data - means * arma::ones<arma::rowvec>(data.n_cols);

  arma::mat covMat = ccov(meanSubData);
  arma::mat eigVec;
  arma::eig_sym(eigVal, eigVec, covMat);

  int n_eigVal = eigVal.n_elem;
  for(int i = 0; i < floor(n_eigVal/2); i++)
    eigVal.swap_rows(i, (n_eigVal-1)-i);

  eigVec = arma::fliplr(eigVec);

  transformedData = trans(eigVec) * data;
}

void PCA::Apply(arma::mat& data, const int newDimension)
{
  arma::mat transData = trans(data);
  arma::vec means = mean(data, 1);
  arma::mat meanSubData = data - means * arma::ones<arma::rowvec>(data.n_cols);

  arma::mat covMat = ccov(meanSubData);
  arma::mat eigVec;
  arma::vec eigVal;
  arma::eig_sym(eigVal, eigVec, covMat);

  int n_eigVal = eigVal.n_elem;
  for(int i = 0; i < floor(n_eigVal/2); i++)
    eigVal.swap_rows(i, (n_eigVal-1)-i);

  eigVec = arma::fliplr(eigVec);

  eigVec.shed_cols(newDimension, eigVec.n_cols - 1);
  cout << eigVec << endl;
  cout << data << endl;

  data = trans(eigVec) * data;

  cout << data << endl;
}

PCA::~PCA()
{
}

}; // namespace mlpack
}; // namespace pca
