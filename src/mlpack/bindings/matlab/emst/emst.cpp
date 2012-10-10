#include "mex.h"

#include "dtb.hpp"
#include <mlpack/core.hpp>

#include <iostream>

using namespace mlpack;
using namespace mlpack::emst;
using namespace mlpack::tree;

// the gateway, required by all mex functions
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 3) 
  {
    mexErrMsgTxt("Expecting an datapoints matrix, isBoruvka, and leafSize.");
  }

  if (nlhs != 1) 
  {
    mexErrMsgTxt("Output required.");
  }

  // getting the dimensions of the input matrix
  const size_t numPoints = mxGetN(prhs[0]);
  const size_t numDimensions = mxGetM(prhs[0]);

  // converting from mxArray to armadillo matrix
  arma::mat dataPoints(numDimensions, numPoints);

  // setting the values. 
  double * mexDataPoints = mxGetPr(prhs[0]);
  for (int i = 0, n = numPoints * numDimensions; i < n; ++i) 
  {
    dataPoints(i) = mexDataPoints[i];
  }

  // getting the isBoruvka bit
  const bool isBoruvka = (mxGetScalar(prhs[1]) == 1.0);

  // running the computation
  arma::mat result;
  if (isBoruvka) 
  {
    // getting the number of leaves
    const size_t leafSize = (size_t) mxGetScalar(prhs[2]);

    DualTreeBoruvka<> dtb(dataPoints, false, leafSize);
    dtb.ComputeMST(result);
  }
  else 
  {
    DualTreeBoruvka<> naive(dataPoints, true);
    naive.ComputeMST(result);
  }

  // constructing matrix to return to matlab
  plhs[0] = mxCreateDoubleMatrix(3, numPoints-1, mxREAL);

  // setting the values
  double * out = mxGetPr(plhs[0]);
  for (int i = 0, n = (numPoints - 1) * 3; i < n; ++i) 
  {
    out[i] = result(i);
  }

  return;
}
