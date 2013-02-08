/**
 * @file range_search.cpp
 * @author Patrick Mason
 *
 * MEX function for MATLAB range search binding.
 */
#include "mex.h"

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/range_search/range_search.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::range;
using namespace mlpack::tree;

typedef RangeSearch<metric::SquaredEuclideanDistance,
    BinarySpaceTree<bound::HRectBound<2>, EmptyStatistic> > RSType;

// the gateway, required by all mex functions
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // Give CLI the command line parameters the user passed in.
  //CLI::ParseCommandLine(argc, argv);

  // Get all the parameters.
  //string referenceFile = CLI::GetParam<string>("reference_file");
  //string distancesFile = CLI::GetParam<string>("distances_file");
  //string neighborsFile = CLI::GetParam<string>("neighbors_file");

  //int lsInt = CLI::GetParam<int>("leaf_size");
  //double max = CLI::GetParam<double>("max");
  //double min = CLI::GetParam<double>("min");
  //bool naive = CLI::HasParam("naive");
  //bool singleMode = CLI::HasParam("single_mode");

  // argument checks
  if (nrhs != 7)
  {
    mexErrMsgTxt("Expecting an datapoints matrix, isBoruvka, and leafSize.");
  }

  if (nlhs != 1)
  {
    mexErrMsgTxt("Output required.");
  }

  double max = mxGetScalar(prhs[1]);
  double min = mxGetScalar(prhs[2]);
  int lsInt = (int) mxGetScalar(prhs[4]);
  bool naive = (mxGetScalar(prhs[5]) == 1.0);
  bool singleMode = (mxGetScalar(prhs[6]) == 1.0);

  // checking for query data
  bool hasQueryData = ((mxGetM(prhs[3]) != 0) && (mxGetN(prhs[3]) != 0));
  arma::mat queryData;

  // setting the dataset values.
  double * mexDataPoints = mxGetPr(prhs[0]);
  size_t numPoints = mxGetN(prhs[0]);
  size_t numDimensions = mxGetM(prhs[0]);
  arma::mat referenceData(numDimensions, numPoints);
  for (int i = 0, n = numPoints * numDimensions; i < n; ++i)
  {
    referenceData(i) = mexDataPoints[i];
  }

  //if (!data::Load(referenceFile.c_str(), referenceData))
  //  Log::Fatal << "Reference file " << referenceFile << "not found." << endl;

  //Log::Info << "Loaded reference data from '" << referenceFile << "'." << endl;

  // Sanity check on range value: max must be greater than min.
  if (max <= min)
  {
    stringstream ss;
    ss << "Invalid range: maximum (" << max << ") must be greater than "
        << "minimum (" << min << ").";
    mexErrMsgTxt(ss.str().c_str());
  }

  // Sanity check on leaf size.
  if (lsInt < 0)
  {
    stringstream ss;
    ss << "Invalid leaf size: " << lsInt << ".  Must be greater "
        "than or equal to 0.";
    mexErrMsgTxt(ss.str().c_str());
  }

  size_t leafSize = lsInt;

  // Naive mode overrides single mode.
  if (singleMode && naive)
  {
    mexWarnMsgTxt("single_mode ignored because naive is present.");
  }

  if (naive)
    leafSize = referenceData.n_cols;

  vector<vector<size_t> > neighbors;
  vector<vector<double> > distances;

  // Because we may construct it differently, we need a pointer.
  RSType* rangeSearch = NULL;

  // Mappings for when we build the tree.
  vector<size_t> oldFromNewRefs;

  // Build trees by hand, so we can save memory: if we pass a tree to
  // NeighborSearch, it does not copy the matrix.
  //Log::Info << "Building reference tree..." << endl;
  //Timer::Start("tree_building");

  BinarySpaceTree<bound::HRectBound<2>, tree::EmptyStatistic>
      refTree(referenceData, oldFromNewRefs, leafSize);
  BinarySpaceTree<bound::HRectBound<2>, tree::EmptyStatistic>*
      queryTree = NULL; // Empty for now.

  //Timer::Stop("tree_building");

  std::vector<size_t> oldFromNewQueries;

  //if (CLI::GetParam<string>("query_file") != "")
  if (hasQueryData)
  {
    //string queryFile = CLI::GetParam<string>("query_file");
    //if (!data::Load(queryFile.c_str(), queryData))
    //  Log::Fatal << "Query file " << queryFile << " not found" << endl;

    // setting the values.
    mexDataPoints = mxGetPr(prhs[3]);
    numPoints = mxGetN(prhs[3]);
    numDimensions = mxGetM(prhs[3]);
    queryData = arma::mat(numDimensions, numPoints);
    for (int i = 0, n = numPoints * numDimensions; i < n; ++i)
    {
      queryData(i) = mexDataPoints[i];
    }

    if (naive && leafSize < queryData.n_cols)
      leafSize = queryData.n_cols;

    //Log::Info << "Loaded query data from '" << queryFile << "'." << endl;

    //Log::Info << "Building query tree..." << endl;

    // Build trees by hand, so we can save memory: if we pass a tree to
    // NeighborSearch, it does not copy the matrix.
    //Timer::Start("tree_building");

    queryTree = new BinarySpaceTree<bound::HRectBound<2>,
        tree::EmptyStatistic >(queryData, oldFromNewQueries,
        leafSize);

    //Timer::Stop("tree_building");

    rangeSearch = new RSType(&refTree, queryTree, referenceData, queryData,
        singleMode);

    //Log::Info << "Tree built." << endl;
  }
  else
  {
    rangeSearch = new RSType(&refTree, referenceData, singleMode);

    //Log::Info << "Trees built." << endl;
  }

  //Log::Info << "Computing neighbors within range [" << min << ", " << max
  //    << "]." << endl;

  math::Range r = math::Range(min, max);
  rangeSearch->Search(r, neighbors, distances);

  //Log::Info << "Neighbors computed." << endl;

  // We have to map back to the original indices from before the tree
  // construction.
  //Log::Info << "Re-mapping indices..." << endl;

  vector<vector<double> > distancesOut;
  distancesOut.resize(distances.size());
  vector<vector<size_t> > neighborsOut;
  neighborsOut.resize(neighbors.size());

  // Do the actual remapping.
  //if (CLI::GetParam<string>("query_file") != "")
  if (hasQueryData)
  {
    for (size_t i = 0; i < distances.size(); ++i)
    {
      // Map distances (copy a column).
      distancesOut[oldFromNewQueries[i]] = distances[i];

      // Map indices of neighbors.
      neighborsOut[oldFromNewQueries[i]].resize(neighbors[i].size());
      for (size_t j = 0; j < distances[i].size(); ++j)
      {
        neighborsOut[oldFromNewQueries[i]][j] = oldFromNewRefs[neighbors[i][j]];
      }
    }
  }
  else
  {
    for (size_t i = 0; i < distances.size(); ++i)
    {
      // Map distances (copy a column).
      distancesOut[oldFromNewRefs[i]] = distances[i];

      // Map indices of neighbors.
      neighborsOut[oldFromNewRefs[i]].resize(neighbors[i].size());
      for (size_t j = 0; j < distances[i].size(); ++j)
      {
        neighborsOut[oldFromNewRefs[i]][j] = oldFromNewRefs[neighbors[i][j]];
      }
    }
  }

  // Setting values to be returned to matlab
  mwSize ndim = 1;
  mwSize dims[1] = {distancesOut.size()};
  const char * fieldNames[2] = {
    "neighbors"
    , "distances"
  };

  plhs[0] = mxCreateStructArray(ndim, dims, 2, fieldNames);

  // setting the structure elements
  for (int i=0; i<distancesOut.size(); ++i)
  {
    mxArray * tmp;
    double * values;

    // settings the neighbors
    const size_t numElements = distancesOut[i].size();
    tmp = mxCreateDoubleMatrix(1, numElements, mxREAL);
    values = mxGetPr(tmp);
    for (int j=0; j<numElements; ++j)
    {
      // converting to matlab's index offset
      values[j] = neighborsOut[i][j] + 1;
    }
    // note: SetField does not copy the data structure.
    // mxDuplicateArray does the necessary copying.
    mxSetFieldByNumber(plhs[0], i, 0, mxDuplicateArray(tmp));
    mxDestroyArray(tmp);

    // setting the distances
    tmp = mxCreateDoubleMatrix(1, numElements, mxREAL);
    values = mxGetPr(tmp);
    for (int j=0; j<numElements; ++j)
    {
      values[j] = distancesOut[i][j];
    }
    mxSetFieldByNumber(plhs[0], i, 1, mxDuplicateArray(tmp));
    mxDestroyArray(tmp);
  }

  // Clean up.
  if (queryTree)
    delete queryTree;
  delete rangeSearch;

  /*
  // Save output.  We have to do this by hand.
  fstream distancesStr(distancesFile.c_str(), fstream::out);
  if (!distancesStr.is_open())
  {
    Log::Warn << "Cannot open file '" << distancesFile << "' to save output "
        << "distances to!" << endl;
  }
  else
  {
    // Loop over each point.
    for (size_t i = 0; i < distancesOut.size(); ++i)
    {
      // Store the distances of each point.  We may have 0 points to store, so
      // we must account for that possibility.
      for (size_t j = 0; j + 1 < distancesOut[i].size(); ++j)
      {
        distancesStr << distancesOut[i][j] << ", ";
      }

      if (distancesOut[i].size() > 0)
        distancesStr << distancesOut[i][distancesOut[i].size() - 1];

      distancesStr << endl;
    }

    distancesStr.close();
  }

  fstream neighborsStr(neighborsFile.c_str(), fstream::out);
  if (!neighborsStr.is_open())
  {
    Log::Warn << "Cannot open file '" << neighborsFile << "' to save output "
        << "neighbor indices to!" << endl;
  }
  else
  {
    // Loop over each point.
    for (size_t i = 0; i < neighborsOut.size(); ++i)
    {
      // Store the neighbors of each point.  We may have 0 points to store, so
      // we must account for that possibility.
      for (size_t j = 0; j + 1 < neighborsOut[i].size(); ++j)
      {
        neighborsStr << neighborsOut[i][j] << ", ";
      }

      if (neighborsOut[i].size() > 0)
        neighborsStr << neighborsOut[i][neighborsOut[i].size() - 1];

      neighborsStr << endl;
    }

    neighborsStr.close();
  }
  */
}
