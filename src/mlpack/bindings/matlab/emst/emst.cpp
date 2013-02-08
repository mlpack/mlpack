/**
 * @file emst.cpp
 * @author Patrick Mason
 *
 * MEX function for MATLAB EMST binding.
 */
#include "mex.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/emst/dtb.hpp>

#include <iostream>

using namespace mlpack;
using namespace mlpack::emst;
using namespace mlpack::tree;

// The gateway, required by all mex functions.
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // Argument checks.
  if (nrhs != 3)
  {
    mexErrMsgTxt("Expecting an datapoints matrix, isBoruvka, and leafSize.");
  }

  if (nlhs != 1)
  {
    mexErrMsgTxt("Output required.");
  }

  const size_t numPoints = mxGetN(prhs[0]);
  const size_t numDimensions = mxGetM(prhs[0]);

  // Converting from mxArray to armadillo matrix.
  arma::mat dataPoints(numDimensions, numPoints);

  // Set the values.
  double* mexDataPoints = mxGetPr(prhs[0]);
  for (int i = 0, n = numPoints * numDimensions; i < n; ++i)
  {
    dataPoints(i) = mexDataPoints[i];
  }

  const bool isBoruvka = (mxGetScalar(prhs[1]) == 1.0);

  // Run the computation.
  arma::mat result;
  if (isBoruvka)
  {
    // Get the number of leaves.
    const size_t leafSize = (size_t) mxGetScalar(prhs[2]);

    DualTreeBoruvka<> dtb(dataPoints, false, leafSize);
    dtb.ComputeMST(result);
  }
  else
  {
    DualTreeBoruvka<> naive(dataPoints, true);
    naive.ComputeMST(result);
  }

  // Construct matrix to return to MATLAB.
  plhs[0] = mxCreateDoubleMatrix(3, numPoints - 1, mxREAL);

  double* out = mxGetPr(plhs[0]);
  for (int i = 0, n = (numPoints - 1) * 3; i < n; ++i)
  {
    out[i] = result(i);
  }
}
