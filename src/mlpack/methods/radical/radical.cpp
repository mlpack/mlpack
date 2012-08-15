/**
 * @file radical.cpp
 * @author Nishant Mehta
 *
 * Implementation of Radical class
 * This file is part of MLPACK 1.0.2.
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

#include "radical.hpp"

using namespace std;
using namespace arma;

namespace mlpack {
namespace radical {


Radical::Radical(double noiseStdDev, size_t nReplicates, size_t nAngles,
     size_t nSweeps) :
  noiseStdDev(noiseStdDev),
  nReplicates(nReplicates),
  nAngles(nAngles),
  nSweeps(nSweeps),
  m(0)
{
  // nothing to do here
}

Radical::Radical(double noiseStdDev, size_t nReplicates, size_t nAngles,
                  size_t nSweeps, size_t m) :
  noiseStdDev(noiseStdDev),
  nReplicates(nReplicates),
  nAngles(nAngles),
  nSweeps(nSweeps),
  m(m)
{
  // nothing to do here
}


void Radical::CopyAndPerturb(mat& matXNew, const mat& matX)
{
  matXNew = repmat(matX, nReplicates, 1) + noiseStdDev *
                  randn(nReplicates * matX.n_rows, matX.n_cols);
}


double Radical::Vasicek(vec& z)
{
  z = sort(z);

  // Apparently slower
  /*
  vec logs = log(z.subvec(m, nPoints - 1) - z.subvec(0, nPoints - 1 - m));
  //vec val = sum(log(z.subvec(m, nPoints - 1) - z.subvec(0, nPoints - 1 - m)));
  return (double) sum(logs);
  */

  // Apparently faster
  double sum = 0;
  uword range = z.n_elem - m;
  for(uword i = 0; i < range; i++) {
    sum += log(z(i + m) - z(i));
  }
  return sum;
}


double Radical::DoRadical2D(const mat& matX)
{
  mat matXMod;

  CopyAndPerturb(matXMod, matX);

  mat matJacobi(2,2);
  mat candidateY;

  vec thetas = linspace<vec>(0, nAngles - 1, nAngles) /
                ((double) nAngles) * M_PI / 2;
  vec values(nAngles);

  for(size_t i = 0; i < nAngles; i++) {
    double cosTheta = cos(thetas(i));
    double sinTheta = sin(thetas(i));
    matJacobi(0,0) = cosTheta;
    matJacobi(1,0) = -sinTheta;
    matJacobi(0,1) = sinTheta;
    matJacobi(1,1) = cosTheta;

    candidateY = matXMod * matJacobi;
    vec candidateY1 = candidateY.unsafe_col(0);
    vec candidateY2 = candidateY.unsafe_col(1);

    values(i) = Vasicek(candidateY1) + Vasicek(candidateY2);
  }

  uword indOpt;
  values.min(indOpt); // we ignore the return value; we don't care about it
  return thetas(indOpt);
}


void Radical::DoRadical(const mat& matXT, mat& matY, mat& matW)
{

  // matX is nPoints by nDims (although less intuitive than columns being
  // points, and although this is the transpose of the ICA literature, this
  // choice is for computational efficiency when repeatedly generating
  // two-dimensional coordinate projections for Radical2D
  mat matX = trans(matXT);


  // if m was not specified, initialize m as recommended in
  // (Learned-Miller and Fisher, 2003)
  if (m < 1) {
    m = floor(sqrt((double) matX.n_rows));
  }

  const size_t nDims = matX.n_cols;
  const size_t nPoints = matX.n_rows;

  mat matXWhitened;
  mat matWhitening;
  WhitenFeatureMajorMatrix(matX, matY, matWhitening);
  // matY is now the whitened form of matX

  // in the RADICAL code, they do not copy and perturb initially, although the
  // paper does. we follow the code as it should match their reported results
  // and likely does a better job bouncing out of local optima
  //GeneratePerturbedX(X, X);

  // initialize the unmixing matrix to the whitening matrix
  matW = matWhitening;

  mat matYSubspace(nPoints, 2);

  mat matEye = eye(nDims, nDims);

  for(size_t sweepNum = 0; sweepNum < nSweeps; sweepNum++)
  {
    for(size_t i = 0; i < nDims - 1; i++)
    {
      for(size_t j = i + 1; j < nDims; j++)
      {
        matYSubspace.col(0) = matY.col(i);
        matYSubspace.col(1) = matY.col(j);
        double thetaOpt = DoRadical2D(matYSubspace);
        mat matJ = matEye;
        double cosThetaOpt = cos(thetaOpt);
        double sinThetaOpt = sin(thetaOpt);
        matJ(i, i) = cosThetaOpt;
        matJ(j, i) = -sinThetaOpt;
        matJ(i, j) = sinThetaOpt;
        matJ(j, j) = cosThetaOpt;
        matW = matW * matJ;
        matY = matX * matW; // to avoid any issues of mismatch between matW
                            // and matY, do not use matY = matY * matJ,
                            // even though it may be much more efficient
      }
    }
  }

  // the final transposes provide W and Y in the typical form from the ICA literature
  //matW = trans(matWhitening * matW);
  //matY = trans(matY);
  matW = trans(matW);
  matY = trans(matY);
}


void WhitenFeatureMajorMatrix(const mat& matX,
            mat& matXWhitened,
            mat& matWhitening)
{
  mat matU, matV;
  vec s;
  svd(matU, s, matV, cov(matX));
  matWhitening = matU * diagmat(1 / sqrt(s)) * trans(matV);
  matXWhitened = matX * matWhitening;
}


}; // namespace radical
}; // namespace mlpack
