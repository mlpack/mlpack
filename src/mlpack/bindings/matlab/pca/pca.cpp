#include "mex.h"

#include <mlpack/core.hpp>

#include <mlpack/methods/pca/pca.hpp>

using namespace mlpack;
using namespace mlpack::pca;
using namespace std;

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

  // loading the data
  double * mexDataPoints = mxGetPr(prhs[0]);
  size_t numPoints = mxGetN(prhs[0]);
  size_t numDimensions = mxGetM(prhs[0]);
  arma::mat dataset(numDimensions, numPoints);
  for (int i = 0, n = numPoints * numDimensions; i < n; ++i)
    dataset(i) = mexDataPoints[i];

  // Find out what dimension we want.
  size_t newDimension = dataset.n_rows; // No reduction, by default.

  if (mxGetScalar(prhs[1]) != 0.0)
  {
    // Validate the parameter.
    newDimension = (size_t) mxGetScalar(prhs[1]);
    if (newDimension > dataset.n_rows)
    {
      std::stringstream ss;
      ss << "New dimensionality (" << newDimension
          << ") cannot be greater than existing dimensionality ("
          << dataset.n_rows << ")!";
      mexErrMsgTxt(ss.str().c_str());
    }
  }

  // Get the options for running PCA.
  const bool scale = (mxGetScalar(prhs[2]) == 1.0);

  // Perform PCA.
  PCA p(scale);
  p.Apply(dataset, newDimension);

  // Now returning results to matlab
  plhs[0] = mxCreateDoubleMatrix(dataset.n_rows, dataset.n_cols, mxREAL);
  double * values = mxGetPr(plhs[0]);
  for (int i = 0; i < dataset.n_rows * dataset.n_cols; ++i)
    values[i] = dataset(i);
}
