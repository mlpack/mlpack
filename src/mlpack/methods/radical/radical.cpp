/**
 * @file radical.cpp
 * @author Nishant Mehta
 *
 * Implementation of Radical class
 */

#include "radical.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::radical;

// Set the parameters to RADICAL.
Radical::Radical(const double noiseStdDev,
                 const size_t nReplicates,
                 const size_t nAngles,
                 const size_t nSweeps,
                 const size_t m) :
    noiseStdDev(noiseStdDev),
    nReplicates(nReplicates),
    nAngles(nAngles),
    nSweeps(nSweeps),
    m(m)
{
  // Nothing to do here.
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
  for (uword i = 0; i < range; i++)
  {
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

  for (size_t i = 0; i < nAngles; i++)
  {
    double cosTheta = cos(thetas(i));
    double sinTheta = sin(thetas(i));
    matJacobi(0, 0) = cosTheta;
    matJacobi(1, 0) = -sinTheta;
    matJacobi(0, 1) = sinTheta;
    matJacobi(1, 1) = cosTheta;

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
  // two-dimensional coordinate projections for Radical2D.
  mat matX = trans(matXT);

  // If m was not specified, initialize m as recommended in
  // (Learned-Miller and Fisher, 2003).
  if (m < 1)
    m = floor(sqrt((double) matX.n_rows));

  const size_t nDims = matX.n_cols;
  const size_t nPoints = matX.n_rows;

  mat matXWhitened;
  mat matWhitening;
  WhitenFeatureMajorMatrix(matX, matY, matWhitening);
  // matY is now the whitened form of matX

  // In the RADICAL code, they do not copy and perturb initially, although the
  // paper does. we follow the code as it should match their reported results
  // and likely does a better job bouncing out of local optima.
  //GeneratePerturbedX(X, X);

  // Initialize the unmixing matrix to the whitening matrix.
  matW = matWhitening;

  mat matYSubspace(nPoints, 2);

  mat matEye = eye(nDims, nDims);

  for (size_t sweepNum = 0; sweepNum < nSweeps; sweepNum++)
  {
    for (size_t i = 0; i < nDims - 1; i++)
    {
      for (size_t j = i + 1; j < nDims; j++)
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

  // the final transposes provide W and Y in the typical form from the ICA
  // literature
  //matW = trans(matWhitening * matW);
  //matY = trans(matY);
  matW = trans(matW);
  matY = trans(matY);
}


void mlpack::radical::WhitenFeatureMajorMatrix(const mat& matX,
                                               mat& matXWhitened,
                                               mat& matWhitening)
{
  mat matU, matV;
  vec s;
  svd(matU, s, matV, cov(matX));
  matWhitening = matU * diagmat(1 / sqrt(s)) * trans(matV);
  matXWhitened = matX * matWhitening;
}
