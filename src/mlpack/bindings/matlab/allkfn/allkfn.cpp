/**
 * @file allkfn.cpp
 * @author Patrick Mason
 *
 * MEX function for MATLAB All-kFN binding.
 */
#include "mex.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // Check the inputs.
  if (nrhs != 6)
  {
    mexErrMsgTxt("Expecting seven arguments.");
  }

  if (nlhs != 2)
  {
    mexErrMsgTxt("Two outputs required.");
  }

  size_t numPoints = mxGetN(prhs[0]);
  size_t numDimensions = mxGetM(prhs[0]);

  // Create the reference matrix.
  arma::mat referenceData(numDimensions, numPoints);
  // setting the values.
  double * mexDataPoints = mxGetPr(prhs[0]);
  for (int i = 0, n = numPoints * numDimensions; i < n; ++i)
  {
    referenceData(i) = mexDataPoints[i];
  }

  // getting the leafsize
  int lsInt = (int) mxGetScalar(prhs[3]);

  // getting k
  size_t k = (int) mxGetScalar(prhs[1]);

  // naive algorithm?
  bool naive = (mxGetScalar(prhs[4]) == 1.0);

  // single mode?
  bool singleMode = (mxGetScalar(prhs[5]) == 1.0);

  // the query matrix
  double * mexQueryPoints = mxGetPr(prhs[2]);
  arma::mat queryData;
  bool hasQueryData = ((mxGetM(prhs[2]) != 0) && (mxGetN(prhs[2]) != 0));

  // Sanity check on k value: must be greater than 0, must be less than the
  // number of reference points.
  if (k > referenceData.n_cols)
  {
    stringstream os;
    os << "Invalid k: " << k << "; must be greater than 0 and less ";
    os << "than or equal to the number of reference points (";
    os << referenceData.n_cols << ")." << endl;
    mexErrMsgTxt(os.str().c_str());
  }

  // Sanity check on leaf size.
  if (lsInt < 0)
  {
    stringstream os;
    os << "Invalid leaf size: " << lsInt << ".  Must be greater ";
    os << "than or equal to 0." << endl;
    mexErrMsgTxt(os.str().c_str());
  }
  size_t leafSize = lsInt;

  // Naive mode overrides single mode.
  if (singleMode && naive)
  {
    mexWarnMsgTxt("single_mode ignored because naive is present.");
  }

  if (naive)
    leafSize = referenceData.n_cols;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  AllkFN* allkfn = NULL;

  std::vector<size_t> oldFromNewRefs;

  // Build trees by hand, so we can save memory: if we pass a tree to
  // NeighborSearch, it does not copy the matrix.
  BinarySpaceTree<bound::HRectBound<2>, QueryStat<FurthestNeighborSort> >
      refTree(referenceData, oldFromNewRefs, leafSize);
  BinarySpaceTree<bound::HRectBound<2>, QueryStat<FurthestNeighborSort> >*
      queryTree = NULL; // Empty for now.

  std::vector<size_t> oldFromNewQueries;

  if (hasQueryData)
  {
    // setting the values.
    mexDataPoints = mxGetPr(prhs[2]);
    numPoints = mxGetN(prhs[2]);
    numDimensions = mxGetM(prhs[2]);
    queryData = arma::mat(numDimensions, numPoints);
    for (int i = 0, n = numPoints * numDimensions; i < n; ++i)
    {
      queryData(i) = mexDataPoints[i];
    }

    if (naive && leafSize < queryData.n_cols)
      leafSize = queryData.n_cols;

    // Build trees by hand, so we can save memory: if we pass a tree to
    // NeighborSearch, it does not copy the matrix.
    queryTree = new BinarySpaceTree<bound::HRectBound<2>,
        QueryStat<FurthestNeighborSort> >(queryData, oldFromNewQueries,
        leafSize);

    allkfn = new AllkFN(&refTree, queryTree, referenceData, queryData,
        singleMode);
  }
  else
  {
    allkfn = new AllkFN(&refTree, referenceData, singleMode);
  }

  allkfn->Search(k, neighbors, distances);

  // We have to map back to the original indices from before the tree
  // construction.
  arma::mat distancesOut(distances.n_rows, distances.n_cols);
  arma::Mat<size_t> neighborsOut(neighbors.n_rows, neighbors.n_cols);

  // Do the actual remapping.
  if (hasQueryData)
  {
    for (size_t i = 0; i < distances.n_cols; ++i)
    {
      // Map distances (copy a column).
      distancesOut.col(oldFromNewQueries[i]) = distances.col(i);

      // Map indices of neighbors.
      for (size_t j = 0; j < distances.n_rows; ++j)
      {
        neighborsOut(j, oldFromNewQueries[i]) = oldFromNewRefs[neighbors(j, i)];
      }
    }
  }
  else
  {
    for (size_t i = 0; i < distances.n_cols; ++i)
    {
      // Map distances (copy a column).
      distancesOut.col(oldFromNewRefs[i]) = distances.col(i);

      // Map indices of neighbors.
      for (size_t j = 0; j < distances.n_rows; ++j)
      {
        neighborsOut(j, oldFromNewRefs[i]) = oldFromNewRefs[neighbors(j, i)];
      }
    }
  }

  // Clean up.
  if (queryTree)
    delete queryTree;

  // constructing matrix to return to matlab
  plhs[0] = mxCreateDoubleMatrix(distances.n_rows, distances.n_cols, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(neighbors.n_rows, neighbors.n_cols, mxREAL);

  // setting the values
  double * out = mxGetPr(plhs[0]);
  for (int i = 0, n = distances.n_rows * distances.n_cols; i < n; ++i)
  {
    out[i] = distances(i);
  }
  out = mxGetPr(plhs[1]);
  for (int i = 0, n = neighbors.n_rows * neighbors.n_cols; i < n; ++i)
  {
    out[i] = neighbors(i);
  }

  // More clean up.
  delete allkfn;
}
