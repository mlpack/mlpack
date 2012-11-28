#include "mex.h"

#include <mlpack/core.hpp>

#include <mlpack/methods/nmf/nmf.hpp>

#include <mlpack/methods/nmf/random_init.hpp>
#include <mlpack/methods/nmf/mult_dist_update_rules.hpp>
#include <mlpack/methods/nmf/mult_div_update_rules.hpp>
#include <mlpack/methods/nmf/als_update_rules.hpp>

using namespace mlpack;
using namespace mlpack::nmf;
using namespace std;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 6)
  {
    mexErrMsgTxt("Expecting six inputs.");
  }

  if (nlhs != 2)
  {
    mexErrMsgTxt("Two outputs required.");
  }

  const size_t seed = (size_t) mxGetScalar(prhs[5]);

  // Initialize random seed.
  if (seed != 0)
    math::RandomSeed(seed);
  else
    math::RandomSeed((size_t) std::time(NULL));

  // Gather parameters.
  const size_t r = (size_t) mxGetScalar(prhs[1]);
  const size_t maxIterations = (size_t) mxGetScalar(prhs[2]);
  const double minResidue = mxGetScalar(prhs[3]);

  // update rule
  int bufLength = mxGetNumberOfElements(prhs[4]) + 1;
  char * buf = (char *) mxCalloc(bufLength, sizeof(char));
  mxGetString(prhs[4], buf, bufLength);
  string updateRules(buf);
  mxFree(buf);

  // Validate rank.
  if (r < 1)
  {
    mexErrMsgTxt("The rank of the factorization cannot be less than 1.");
  }

  if ((updateRules != "multdist") &&
      (updateRules != "multdiv") &&
      (updateRules != "als"))
  {
    stringstream ss;
    ss << "Invalid update rules ('" << updateRules << "'); must be '"
        << "multdist', 'multdiv', or 'als'.";
    mexErrMsgTxt(ss.str().c_str());
  }

  // Load input dataset.
  arma::mat V(mxGetM(prhs[0]), mxGetN(prhs[0]));
  double * values = mxGetPr(prhs[0]);
  for (int i=0, num=mxGetNumberOfElements(prhs[0]); i<num; ++i)
    V(i) = values[i];

  arma::mat W;
  arma::mat H;

  // Perform NMF with the specified update rules.
  if (updateRules == "multdist")
  {
    NMF<> nmf(maxIterations, minResidue);
    nmf.Apply(V, r, W, H);
  }
  else if (updateRules == "multdiv")
  {
    NMF<RandomInitialization,
        WMultiplicativeDivergenceRule,
        HMultiplicativeDivergenceRule> nmf(maxIterations, minResidue);
    nmf.Apply(V, r, W, H);
  }
  else if (updateRules == "als")
  {
    NMF<RandomInitialization,
        WAlternatingLeastSquaresRule,
        HAlternatingLeastSquaresRule> nmf(maxIterations, minResidue);
    nmf.Apply(V, r, W, H);
  }

  // return to matlab
  plhs[0] = mxCreateDoubleMatrix(W.n_rows, W.n_cols, mxREAL);
  values = mxGetPr(plhs[0]);
  for (int i = 0; i < W.n_elem; ++i)
    values[i] = W(i);

  plhs[1] = mxCreateDoubleMatrix(H.n_rows, H.n_cols, mxREAL);
  values = mxGetPr(plhs[0]);
  for (int i = 0; i < H.n_elem; ++i)
    values[i] = H(i);
}
