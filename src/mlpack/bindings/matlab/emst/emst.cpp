/**
 * @file emst.cpp
 * @author Patrick Mason
 *
 * MEX function for MATLAB EMST binding.
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
