#include "mex.h"

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

#include <mlpack/methods/nca/nca.hpp>

using namespace mlpack;
using namespace mlpack::nca;
using namespace mlpack::metric;
using namespace std;
using namespace arma;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 2)
  {
    mexErrMsgTxt("Expecting two inputs.");
  }

  if (nlhs != 1)
  {
    mexErrMsgTxt("Output required.");
  }

  // Load data.
  mat data(mxGetM(prhs[0]), mxGetN(prhs[0]));
  double * values = mxGetPr(prhs[0]);
  for (int i=0, num=mxGetNumberOfElements(prhs[0]); i<num; ++i)
    data(i) = values[i];

  // load labels
  umat labels(mxGetNumberOfElements(prhs[1]), 1);
  values = mxGetPr(prhs[1]);
  for (int i=0, num=mxGetNumberOfElements(prhs[1]); i<num; ++i)
    labels(i) = (int) values[i];

  // dimension checks
  if (labels.n_elem != data.n_cols)
    mexErrMsgTxt("Labels vector and data have unmatching dimensions.");

  // Now create the NCA object and run the optimization.
  NCA<LMetric<2> > nca(data, labels.unsafe_col(0));

  mat distance;
  nca.LearnDistance(distance);

  // return to matlab
  plhs[0] = mxCreateDoubleMatrix(distance.n_rows, distance.n_cols, mxREAL);
  values = mxGetPr(plhs[0]);
  for (int i = 0; i < distance.n_elem; ++i)
    values[i] = distance(i);
}
