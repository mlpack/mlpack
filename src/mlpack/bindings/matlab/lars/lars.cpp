#include "mex.h"

#include <mlpack/core.hpp>

#include <mlpack/methods/lars/lars.hpp>

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 4)
  {
    mexErrMsgTxt("Expecting four inputs.");
  }

  if (nlhs != 1)
  {
    mexErrMsgTxt("Output required.");
  }

  double lambda1 = mxGetScalar(prhs[2]);
  double lambda2 = mxGetScalar(prhs[3]);
  bool useCholesky = (mxGetScalar(prhs[3]) == 1.0);

  // loading covariates
  mat matX(mxGetM(prhs[0]), mxGetN(prhs[0]));
  double * values = mxGetPr(prhs[0]);
  for (int i=0, num=mxGetNumberOfElements(prhs[0]); i<num; ++i)
    matX(i) = values[i];

  // loading responses
  mat matY(mxGetM(prhs[1]), mxGetN(prhs[1]));
  values = mxGetPr(prhs[1]);
  for (int i=0, num=mxGetNumberOfElements(prhs[1]); i<num; ++i)
    matY(i) = values[i];

  if (matY.n_cols > 1)
    mexErrMsgTxt("Only one column or row allowed in responses file!");

  if (matY.n_elem != matX.n_rows)
    mexErrMsgTxt("Number of responses must be equal to number of rows of X!");

  // Do LARS.
  LARS lars(useCholesky, lambda1, lambda2);
  vec beta;
  lars.Regress(matX, matY.unsafe_col(0), beta, false /* do not transpose */);

  // return to matlab
  plhs[0] = mxCreateDoubleMatrix(beta.n_elem, 1, mxREAL);
  values = mxGetPr(plhs[0]);
  for (int i = 0; i < beta.n_elem; ++i)
    values[i] = beta(i);
}
