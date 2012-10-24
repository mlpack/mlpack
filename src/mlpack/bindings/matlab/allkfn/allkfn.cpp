#include "mex.h"

#include <mlpack/core.hpp>

#include <string>
#include <fstream>
#include <iostream>

#include "neighbor_search.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;

/*
// Information about the program itself.
PROGRAM_INFO("All K-Furthest-Neighbors",
    "This program will calculate the all k-furthest-neighbors of a set of "
    "points. You may specify a separate set of reference points and query "
    "points, or just a reference set which will be used as both the reference "
    "and query set."
    "\n\n"
    "For example, the following will calculate the 5 furthest neighbors of each"
    "point in 'input.csv' and store the distances in 'distances.csv' and the "
    "neighbors in the file 'neighbors.csv':"
    "\n\n"
    "$ allkfn --k=5 --reference_file=input.csv --distances_file=distances.csv\n"
    "  --neighbors_file=neighbors.csv"
    "\n\n"
    "The output files are organized such that row i and column j in the "
    "neighbors output file corresponds to the index of the point in the "
    "reference set which is the i'th furthest neighbor from the point in the "
    "query set with index j.  Row i and column j in the distances output file "
    "corresponds to the distance between those two points.");

// Define our input parameters that this program will take.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
    "r");
PARAM_INT_REQ("k", "Number of furthest neighbors to find.", "k");
PARAM_STRING_REQ("distances_file", "File to output distances into.", "d");
PARAM_STRING_REQ("neighbors_file", "File to output neighbors into.", "n");

PARAM_STRING("query_file", "File containing query points (optional).", "q", "");

PARAM_INT("leaf_size", "Leaf size for tree building.", "l", 20);
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed to "
    "dual-tree search.", "s");
*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // checking inputs
	if (nrhs != 6) 
  {
    mexErrMsgTxt("Expecting seven arguments.");
  }

  if (nlhs != 2) 
  {
    mexErrMsgTxt("Two outputs required.");
  }

	/*
  // Give CLI the command line parameters the user passed in.
  CLI::ParseCommandLine(argc, argv);

  // Get all the parameters.
  string referenceFile = CLI::GetParam<string>("reference_file");

  string distancesFile = CLI::GetParam<string>("distances_file");
  string neighborsFile = CLI::GetParam<string>("neighbors_file");

  int lsInt = CLI::GetParam<int>("leaf_size");

  size_t k = CLI::GetParam<int>("k");

  bool naive = CLI::HasParam("naive");
  bool singleMode = CLI::HasParam("single_mode");
	*/

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

  //arma::mat referenceData;
  //arma::mat queryData; // So it doesn't go out of scope.
  //data::Load(referenceFile.c_str(), referenceData, true);

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

  AllkFN* allkfn = NULL;

  std::vector<size_t> oldFromNewRefs;

  // Build trees by hand, so we can save memory: if we pass a tree to
  // NeighborSearch, it does not copy the matrix.
  BinarySpaceTree<bound::HRectBound<2>, QueryStat<FurthestNeighborSort> >
      refTree(referenceData, oldFromNewRefs, leafSize);
  BinarySpaceTree<bound::HRectBound<2>, QueryStat<FurthestNeighborSort> >*
      queryTree = NULL; // Empty for now.

  std::vector<size_t> oldFromNewQueries;

  //f (CLI::GetParam<string>("query_file") != "")
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
  //if (CLI::GetParam<string>("query_file") != "")
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
