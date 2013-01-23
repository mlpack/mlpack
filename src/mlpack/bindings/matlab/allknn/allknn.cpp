/**
 * @file allknn.cpp
 * @author Patrick Mason
 *
 * MEX function for MATLAB All-kNN binding.
 */
#include "mex.h"

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;

// the gateway, required by all mex functions
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // checking inputs
  if (nrhs != 7)
  {
    mexErrMsgTxt("Expecting seven arguments.");
  }

  if (nlhs != 2)
  {
    mexErrMsgTxt("Two outputs required.");
  }

  // getting the dimensions of the reference matrix
  size_t numPoints = mxGetN(prhs[0]);
  size_t numDimensions = mxGetM(prhs[0]);

  // feeding the referenceData matrix
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

  // cover-tree?
  bool usesCoverTree = (mxGetScalar(prhs[6]) == 1.0);

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
    os << "Invalid leaf size: " << lsInt << ".  Must be greater "
        "than or equal to 0." << endl;
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

  //if (!CLI::HasParam("cover_tree"))
  if (usesCoverTree)
  {
    // Because we may construct it differently, we need a pointer.
    AllkNN* allknn = NULL;

    // Mappings for when we build the tree.
    std::vector<size_t> oldFromNewRefs;

    // Build trees by hand, so we can save memory: if we pass a tree to
    // NeighborSearch, it does not copy the matrix.

    BinarySpaceTree<bound::HRectBound<2>, QueryStat<NearestNeighborSort> >
      refTree(referenceData, oldFromNewRefs, leafSize);
    BinarySpaceTree<bound::HRectBound<2>, QueryStat<NearestNeighborSort> >*
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
      if (!singleMode)
      {
        queryTree = new BinarySpaceTree<bound::HRectBound<2>,
            QueryStat<NearestNeighborSort> >(queryData, oldFromNewQueries,
            leafSize);
      }

      allknn = new AllkNN(&refTree, queryTree, referenceData, queryData,
          singleMode);
    }
    else
    {
      allknn = new AllkNN(&refTree, referenceData, singleMode);
    }

    arma::mat distancesOut;
    arma::Mat<size_t> neighborsOut;

    allknn->Search(k, neighborsOut, distancesOut);

    // We have to map back to the original indices from before the tree
    // construction.
    neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
    distances.set_size(distancesOut.n_rows, distancesOut.n_cols);

    // Do the actual remapping.
    if ((hasQueryData) && !singleMode)
    {
      for (size_t i = 0; i < distancesOut.n_cols; ++i)
      {
        // Map distances (copy a column) and square root.
        distances.col(oldFromNewQueries[i]) = sqrt(distancesOut.col(i));

        // Map indices of neighbors.
        for (size_t j = 0; j < distancesOut.n_rows; ++j)
        {
          neighbors(j, oldFromNewQueries[i]) =
              oldFromNewRefs[neighborsOut(j, i)];
        }
      }
    }
    else if ((hasQueryData) && singleMode)
    {
      // No remapping of queries is necessary.  So distances are the same.
      distances = sqrt(distancesOut);

      // The neighbor indices must be mapped.
      for (size_t j = 0; j < neighborsOut.n_elem; ++j)
      {
        neighbors[j] = oldFromNewRefs[neighborsOut[j]];
      }
    }
    else
    {
      for (size_t i = 0; i < distancesOut.n_cols; ++i)
      {
        // Map distances (copy a column).
        distances.col(oldFromNewRefs[i]) = sqrt(distancesOut.col(i));

        // Map indices of neighbors.
        for (size_t j = 0; j < distancesOut.n_rows; ++j)
        {
          neighbors(j, oldFromNewRefs[i]) = oldFromNewRefs[neighborsOut(j, i)];
        }
      }
    }

    // Clean up.
    if (queryTree)
      delete queryTree;

    delete allknn;
  }
  else // Cover trees.
  {
    // Build our reference tree.
    CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
        QueryStat<NearestNeighborSort> > referenceTree(referenceData, 1.3);
    CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
        QueryStat<NearestNeighborSort> >* queryTree = NULL;

    NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
        CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
        QueryStat<NearestNeighborSort> > >* allknn = NULL;

    // See if we have query data.
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

      // Build query tree.
      if (!singleMode)
      {
        queryTree = new CoverTree<metric::LMetric<2, true>,
            tree::FirstPointIsRoot, QueryStat<NearestNeighborSort> >(queryData,
            1.3);
      }

      allknn = new NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
          QueryStat<NearestNeighborSort> > >(&referenceTree, queryTree,
          referenceData, queryData, singleMode);
    }
    else
    {
      allknn = new NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          CoverTree<metric::LMetric<2, true>, tree::FirstPointIsRoot,
          QueryStat<NearestNeighborSort> > >(&referenceTree, referenceData,
          singleMode);
    }

    allknn->Search(k, neighbors, distances);

    delete allknn;

    if (queryTree)
      delete queryTree;
  }

  // writing back to matlab
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

}
