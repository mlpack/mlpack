/**
 * @file gmm.cpp
 * @author Patrick Mason
 *
 * MEX function for MATLAB GMM binding.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "mex.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::util;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 3)
  {
    mexErrMsgTxt("Expecting three inputs.");
  }

  if (nlhs != 1)
  {
    mexErrMsgTxt("Output required.");
  }

  size_t seed = (size_t) mxGetScalar(prhs[2]);
  // Check parameters and load data.
  if (seed != 0)
    math::RandomSeed(seed);
  else
    math::RandomSeed((size_t) std::time(NULL));

  // loading the data
  double * mexDataPoints = mxGetPr(prhs[0]);
  size_t numPoints = mxGetN(prhs[0]);
  size_t numDimensions = mxGetM(prhs[0]);
  arma::mat dataPoints(numDimensions, numPoints);
  for (int i = 0, n = numPoints * numDimensions; i < n; ++i)
  {
    dataPoints(i) = mexDataPoints[i];
  }

  int gaussians = (int) mxGetScalar(prhs[1]);
  if (gaussians <= 0)
  {
    std::stringstream ss;
    ss << "Invalid number of Gaussians (" << gaussians << "); must "
        "be greater than or equal to 1." << std::endl;
    mexErrMsgTxt(ss.str().c_str());
  }

  // Calculate mixture of Gaussians.
  GMM<> gmm(size_t(gaussians), dataPoints.n_rows);

  ////// Computing the parameters of the model using the EM algorithm //////
  gmm.Estimate(dataPoints);

  // setting up the matlab structure to be returned
  mwSize ndim = 1;
  mwSize dims[1] = {
    1
  };
  const char * fieldNames[3] = {
    "dimensionality"
    , "weights"
    , "gaussians"
  };

  plhs[0] =  mxCreateStructArray(ndim, dims, 3, fieldNames);

  // dimensionality
  mxArray * field_value;
  field_value = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(field_value) = numDimensions;
  mxSetFieldByNumber(plhs[0], 0, 0, field_value);

  // mixture weights
  field_value = mxCreateDoubleMatrix(gmm.Weights().size(), 1, mxREAL);
  double * values = mxGetPr(field_value);
  for (int i=0; i<gmm.Weights().size(); ++i)
  {
    values[i] = gmm.Weights()[i];
  }
  mxSetFieldByNumber(plhs[0], 0, 1, field_value);

  // gaussian mean/variances
  const char * gaussianNames[2] = {
    "mean"
    , "covariance"
  };
  ndim = 1;
  dims[0] = gmm.Gaussians();

  field_value = mxCreateStructArray(ndim, dims, 2, gaussianNames);
  for (int i=0; i<gmm.Gaussians(); ++i)
  {
    mxArray * tmp;
    double * values;

    // setting the mean
    arma::mat mean = gmm.Means()[i];
    tmp = mxCreateDoubleMatrix(numDimensions, 1, mxREAL);
    values = mxGetPr(tmp);
    for (int j = 0; j < numDimensions; ++j)
    {
      values[j] = mean(j);
    }
    // note: SetField does not copy the data structure.
    // mxDuplicateArray does the necessary copying.
    mxSetFieldByNumber(field_value, i, 0, mxDuplicateArray(tmp));
    mxDestroyArray(tmp);

    // setting the covariance matrix
    arma::mat covariance = gmm.Covariances()[i];
    tmp = mxCreateDoubleMatrix(numDimensions, numDimensions, mxREAL);
    values = mxGetPr(tmp);
    for (int j = 0; j < numDimensions * numDimensions; ++j)
    {
      values[j] = covariance(j);
    }
    mxSetFieldByNumber(field_value, i, 1, mxDuplicateArray(tmp));
    mxDestroyArray(tmp);
  }
  mxSetFieldByNumber(plhs[0], 0, 2, field_value);
}
