/**
 * @file lsh_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of the LSHSearch class.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP

#include <map>

#include <mlpack/core.hpp>

using namespace mlpack::neighbor;

// Construct the object.
template<typename SortPolicy, typename MetricType>
LSHSearch<SortPolicy, MetricType>::
LSHSearch(const arma::mat& referenceSet,
          const arma::mat& querySet,
          const size_t numProj,
          const size_t numTables,
          const double hashWidth,
          const size_t secondHashSize,
          const size_t bucketSize,
          const MetricType metric) :
  referenceSet(referenceSet),
  querySet(querySet),
  numProj(numProj),
  numTables(numTables),
  hashWidth(hashWidth),
  secondHashSize(secondHashSize),
  bucketSize(bucketSize),
  metric(metric)
{
  BuildHash();
}

template<typename SortPolicy, typename MetricType>
LSHSearch<SortPolicy, MetricType>::
LSHSearch(const arma::mat& referenceSet,
          const size_t numProj,
          const size_t numTables,
          const double hashWidth,
          const size_t secondHashSize,
          const size_t bucketSize,
          const MetricType metric) :
  referenceSet(referenceSet),
  querySet(referenceSet),
  numProj(numProj),
  numTables(numTables),
  hashWidth(hashWidth),
  secondHashSize(secondHashSize),
  bucketSize(bucketSize),
  metric(metric)
{
  BuildHash();
}


template<typename SortPolicy, typename MetricType>
LSHSearch<SortPolicy, MetricType>::
~LSHSearch()
{ }


template<typename SortPolicy, typename MetricType>
void LSHSearch<SortPolicy, MetricType>::
InsertNeighbor(const size_t queryIndex,
               const size_t pos,
               const size_t neighbor,
               const double distance)
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (distancePtr->n_rows - 1))
  {
    int len = (distancePtr->n_rows - 1) - pos;
    memmove(distancePtr->colptr(queryIndex) + (pos + 1),
        distancePtr->colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(neighborPtr->colptr(queryIndex) + (pos + 1),
        neighborPtr->colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  (*distancePtr)(pos, queryIndex) = distance;
  (*neighborPtr)(pos, queryIndex) = neighbor;
}



template<typename SortPolicy, typename MetricType>
inline force_inline
double LSHSearch<SortPolicy, MetricType>::
BaseCase(const size_t queryIndex, const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return 0.0;

  double distance = metric.Evaluate(querySet.unsafe_col(queryIndex),
                                    referenceSet.unsafe_col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distancePtr->unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, distance);

  return distance;
}


template<typename SortPolicy, typename MetricType>
void LSHSearch<SortPolicy, MetricType>::
ReturnIndicesFromTable(const size_t queryIndex,
                       arma::uvec& referenceIndices)
{
  // Hash the query in each of the 'numTables' hash tables using the
  // 'numProj' projections for each table.
  // This gives us 'numTables' keys for the query where each key
  // is a 'numProj' dimensional integer vector
  //
  // compute the projection of the query in each table
  arma::mat allProjInTables(numProj, numTables);
  for (size_t i = 0; i < numTables; i++)
    allProjInTables.unsafe_col(i)
      = projections[i].t() * querySet.unsafe_col(queryIndex);
  allProjInTables += offsets;
  allProjInTables /= hashWidth;

  // compute the hash value of each key of the query into a bucket of the
  // 'secondHashTable' using the 'secondHashWeights'.
  arma::rowvec hashVec = secondHashWeights.t() * arma::floor(allProjInTables);

  for (size_t i = 0; i < hashVec.n_elem; i++)
    hashVec[i] = (double)((size_t) hashVec[i] % secondHashSize);

  assert(hashVec.n_elem == numTables);

  // For all the buckets that the query is hashed into, sequentially
  // collect the indices in those buckets.
  arma::Col<size_t> refPointsConsidered;
  refPointsConsidered.zeros(referenceSet.n_cols);

  for (size_t i = 0; i < hashVec.n_elem; i++)
  {
    size_t hashInd = (size_t) hashVec[i];

    if (bucketContentSize[hashInd] > 0)
    {
      // Pick the indices in the bucket corresponding to 'hashInd'
      size_t tableRow = bucketRowInHashTable[hashInd];
      assert(tableRow < secondHashSize);
      assert(tableRow < secondHashTable.n_rows);

      for (size_t j = 0; j < bucketContentSize[hashInd]; j++)
        refPointsConsidered[secondHashTable(tableRow, j)]++;
    }
  } // for all tables

  referenceIndices = arma::find(refPointsConsidered > 0);
  return;
}


template<typename SortPolicy, typename MetricType>
void LSHSearch<SortPolicy, MetricType>::
Search(const size_t k,
       arma::Mat<size_t>& resultingNeighbors,
       arma::mat& distances)
{
  neighborPtr = &resultingNeighbors;
  distancePtr = &distances;

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());
  neighborPtr->fill(referenceSet.n_cols);

  size_t avgIndicesReturned = 0;

  Timer::Start("computing_neighbors");

  // go through every query point sequentially
  for (size_t i = 0; i < querySet.n_cols; i++)
  {
    // For hash every query into every hash tables and eventually
    // into the 'secondHashTable' to obtain the neighbor candidates
    arma::uvec refIndices;
    ReturnIndicesFromTable(i, refIndices);

    // Just an informative book-keeping for the number of neighbor candidates
    // returned on average
    avgIndicesReturned += refIndices.n_elem;

    // Sequentially go through all the candidates and save the best 'k'
    // candidates
    for (size_t j = 0; j < refIndices.n_elem; j++)
      BaseCase(i, (size_t) refIndices[j]);
  }

  Timer::Stop("computing_neighbors");

  avgIndicesReturned /= querySet.n_cols;
  Log::Info << avgIndicesReturned << " distinct indices returned on average." <<
    std::endl;

  return;
}

template<typename SortPolicy, typename MetricType>
void LSHSearch<SortPolicy, MetricType>::
BuildHash()
{
  // The first level hash for a single table outputs a 'numProj'-dimensional
  // integer key for each point in the set -- (key, pointID)
  // The key creation details are presented below
  //
  // The second level hash is performed by hashing the key to
  // an integer in the range [0, 'secondHashSize').
  //
  // This is done by creating a weight vector 'secondHashWeights' of
  // length 'numProj' with each entry an integer randomly chosen
  // between [0, 'secondHashSize').
  //
  // Then the bucket for any key and its corresponding point is
  // given by <key, 'secondHashWeights'> % 'secondHashSize'
  // and the corresponding point ID is put into that bucket.

  //////////////////////////////////////////
  // Step I: Preparing the second level hash
  ///////////////////////////////////////////

  // obtain the weights for the second hash
  secondHashWeights = arma::floor(arma::randu(numProj)
                                  * (double) secondHashSize);

  // The 'secondHashTable' is initially an empty matrix of size
  // ('secondHashSize' x 'bucketSize'). But by only filling the buckets
  // as points land in them allows us to shrink the size of the
  // 'secondHashTable' at the end of the hashing.

  // Start filling up the second hash table
  secondHashTable.set_size(secondHashSize, bucketSize);

  // Fill the second hash table n = referenceSet.n_cols
  // This is because no point has index 'n' so the presence of
  // this in the bucket denotes that there are no more points
  // in this bucket.
  secondHashTable.fill(referenceSet.n_cols);

  // Keeping track of the size of each bucket in the hash.
  // At the end of hashing most buckets will be empty.
  bucketContentSize.zeros(secondHashSize);

  // Instead of putting the points in the row corresponding to
  // the bucket, we chose the next empty row and keep track of
  // the row in which the bucket lies. This allows us to
  // stack together and slice out the empty buckets at the
  // end of the hashing.
  bucketRowInHashTable.set_size(secondHashSize);
  bucketRowInHashTable.fill(secondHashSize);

  // keeping track of number of non-empty rows in the 'secondHashTable'
  size_t numRowsInTable = 0;


  /////////////////////////////////////////////////////////
  // Step II: The offsets for all projections in all tables
  /////////////////////////////////////////////////////////

  // Since the 'offsets' are in [0, hashWidth], we obtain the 'offsets'
  // as randu(numProj, numTables) * hashWidth
  offsets.randu(numProj, numTables);
  offsets *= hashWidth;

  /////////////////////////////////////////////////////////////////
  // Step III: Creating each hash table in the first level hash
  // one by one and putting them directly into the 'secondHashTable'
  // for memory efficiency.
  /////////////////////////////////////////////////////////////////

  for(size_t i = 0; i < numTables; i++)
  {
    //////////////////////////////////////////////////////////////
    // Step IV: Obtaining the 'numProj' projections for each table
    //////////////////////////////////////////////////////////////
    //
    // For L2 metric, 2-stable distributions are used, and
    // the normal Z ~ N(0, 1) is a 2-stable distribution.
    arma::mat projMat;
    projMat.randn(referenceSet.n_rows, numProj);

    // save the projection matrix for querying
    projections.push_back(projMat);

    ///////////////////////////////////////////////////////////////
    // Step V: create the 'numProj'-dimensional key for each point
    // in each table.
    //////////////////////////////////////////////////////////////

    // The following set of lines performs the task of
    // hashing each point to a 'numProj'-dimensional integer key.
    // Hence you get a ('numProj' x 'referenceSet.n_cols') key matrix
    //
    // For a single table, let the 'numProj' projections be denoted
    // by 'proj_i' and the corresponding offset be 'offset_i'.
    // Then the key of a single point is obtained as:
    // key = { floor( (<proj_i, point> + offset_i) / 'hashWidth' ) forall i }
    arma::mat offsetMat = arma::repmat(offsets.unsafe_col(i),
                                       1, referenceSet.n_cols);
    arma::mat hashMat = projMat.t() * referenceSet;
    hashMat += offsetMat;
    hashMat /= hashWidth;

    ////////////////////////////////////////////////////////////
    // Step VI: Putting the points in the 'secondHashTable' by
    // hashing the key.
    ///////////////////////////////////////////////////////////

    // Now we hash every key, point ID to its corresponding bucket
    arma::rowvec secondHashVec = secondHashWeights.t()
      * arma::floor(hashMat);

    // This gives us the bucket for the corresponding point ID
    for (size_t j = 0; j < secondHashVec.n_elem; j++)
      secondHashVec[j] = (double)((size_t) secondHashVec[j] % secondHashSize);

    assert(secondHashVec.n_elem == referenceSet.n_cols);

    // Inserting the point in the corresponding row to its bucket
    // in the 'secondHashTable'.
    for (size_t j = 0; j < secondHashVec.n_elem; j++)
    {
      // This is the bucket number
      size_t hashInd = (size_t) secondHashVec[j];
      // The point ID is 'j'

      // If this is currently an empty bucket, start a new row
      // keep track of which row corresponds to the bucket.
      if (bucketContentSize[hashInd] == 0)
      {
        // start a new row for hash
        bucketRowInHashTable[hashInd] = numRowsInTable;
        secondHashTable(numRowsInTable, 0) = j;

        numRowsInTable++;
      }
      // If bucket already present in the 'secondHashTable', find
      // the corresponding row and insert the point ID in this row
      // unless the bucket is full, in which case, do nothing.
      else
      {
        // if bucket not full, insert point here
        if (bucketContentSize[hashInd] < bucketSize)
          secondHashTable(bucketRowInHashTable[hashInd],
                          bucketContentSize[hashInd]) = j;
        // else just ignore as suggested
      }

      // increment the count of the points in this bucket
      if (bucketContentSize[hashInd] < bucketSize)
        bucketContentSize[hashInd]++;
    } // loop over all points in the reference set
  } // loop over tables


  /////////////////////////////////////////////////
  // Step VII: Condensing the 'secondHashTable'
  /////////////////////////////////////////////////

  size_t maxBucketSize = 0;
  for (size_t i = 0; i < bucketContentSize.n_elem; i++)
    if (bucketContentSize[i] > maxBucketSize)
      maxBucketSize = bucketContentSize[i];

  Log::Info << "Final hash table size: (" << numRowsInTable << " x "
            << maxBucketSize << ")" << std::endl;
  secondHashTable.resize(numRowsInTable, maxBucketSize);

  return;
}


#endif
