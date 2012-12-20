/**
 * @file lsh_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of LSHSearch class.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP

#include <map>

#include <mlpack/core.hpp>

using namespace mlpack::neighbor;

// Construct the object.
template<typename SortPolicy, typename MetricType, typename eT>
LSHSearch<SortPolicy, MetricType, eT>::
LSHSearch(const MatType& referenceSet,
          const MatType& querySet,
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
  // Get a (N^K key, point index) pair for all tables and all points
  MatType allKeysPointsMat(numProj + 1, referenceSet.n_cols * numTables);
  BuildFirstLevelHash(&allKeysPointsMat);

  // Condense the (N^K key, point index) pairs into a single table
  BuildSecondLevelHash(allKeysPointsMat);
}

template<typename SortPolicy, typename MetricType, typename eT>
LSHSearch<SortPolicy, MetricType, eT>::
LSHSearch(const MatType& referenceSet,
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
  // Get a (N^K key, point index) pair for all tables and all points
  MatType allKeysPointsMat(numProj + 1, referenceSet.n_cols * numTables);
  BuildFirstLevelHash(&allKeysPointsMat);

  // Condense the (N^K key, point index) pairs into a single table
  BuildSecondLevelHash(allKeysPointsMat);
}


template<typename SortPolicy, typename MetricType, typename eT>
LSHSearch<SortPolicy, MetricType, eT>::
~LSHSearch()
{ }


template<typename SortPolicy, typename MetricType, typename eT>
void LSHSearch<SortPolicy, MetricType, eT>::
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



template<typename SortPolicy, typename MetricType, typename eT>
inline void LSHSearch<SortPolicy, MetricType, eT>::
BaseCase(const size_t queryIndex,
         const size_t referenceIndex)
{
  // If the datasets are the same, then this search is only using one dataset
  // and we should not return identical points.
  if ((&querySet == &referenceSet) && (queryIndex == referenceIndex))
    return;

  double distance = metric.Evaluate(querySet.col(queryIndex),
                                    referenceSet.col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distancePtr->unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(queryIndex, insertPosition, referenceIndex, distance);

  return;
}


template<typename SortPolicy, typename MetricType, typename eT>
void LSHSearch<SortPolicy, MetricType, eT>::
ReturnIndicesFromTable(const size_t queryIndex,
                       arma::uvec& referenceIndices)
{
  // compute the projection of the query in each table
  MatType allProjInTables(numProj, numTables);

  for (size_t i = 0; i < numTables; i++)
    allProjInTables.col(i) = projections[i].t() * querySet.col(queryIndex);

  allProjInTables += offsets;
  allProjInTables /= hashWidth;

  // compute the hash value of each projection of the query
  // in the second hash table
  RowType hashVec = secondHashWeights.t() * arma::floor(allProjInTables);

  assert(hashVec.n_elem == numTables);

  for (size_t i = 0; i < hashVec.n_elem; i++)
    hashVec[i] = (double)((size_t) hashVec[i] % secondHashSize);

  arma::Col<size_t> refPointsConsidered;
  refPointsConsidered.zeros(referenceSet.n_cols);

  for (size_t i = 0; i < hashVec.n_elem; i++)
  {
    size_t hashInd = (size_t) hashVec[i];

    if (bucketContentSize[hashInd] > 0)
    {
      // Pick the indices in that 'hashInd'
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



template<typename SortPolicy, typename MetricType, typename eT>
void LSHSearch<SortPolicy, MetricType, eT>::
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
  neighborPtr->fill((size_t) -1);


  size_t avgIndicesReturned = 0;


  Timer::Start("computing_neighbors");

  // go through every query point
  for (size_t i = 0; i < querySet.n_cols; i++)
  {
    arma::uvec refIndices;
    ReturnIndicesFromTable(i, refIndices);

    avgIndicesReturned += refIndices.n_elem;

    for (size_t j = 0; j < refIndices.n_elem; j++)
      BaseCase(i, (size_t) refIndices[j]);
  }

  Timer::Stop("computing_neighbors");

  avgIndicesReturned /= querySet.n_cols;
  Log::Info << avgIndicesReturned << " distinct indices returned on average." 
            << std::endl;

  return;
}


template<typename SortPolicy, typename MetricType, typename eT>
void LSHSearch<SortPolicy, MetricType, eT>::
BuildFirstLevelHash(MatType* allKeysPointsMat)
{
  // A row with all the indices
  RowType allIndRow(referenceSet.n_cols);
  for (size_t i = 0; i < allIndRow.n_elem; i++)
    allIndRow[i] = i;

  // Obtain all the projection matrices and the offset matrix
  offsets.randu(numProj, numTables);
  offsets *= hashWidth;

  for(size_t i = 0; i < numTables; i++)
  {
    MatType projMat;
    projMat.randn(referenceSet.n_rows, numProj);

    MatType offsetMat = arma::repmat(offsets.col(i), 1, referenceSet.n_cols);
    MatType hashMat = projMat.t() * referenceSet;

    hashMat += offsetMat;
    hashMat /= hashWidth;

    hashMat.resize(hashMat.n_rows + 1, hashMat.n_cols);
    hashMat.row(hashMat.n_rows - 1) = allIndRow;

    allKeysPointsMat->cols(i * referenceSet.n_cols, (i + 1) * referenceSet.n_cols - 1)
      = arma::floor(hashMat);

    projections.push_back(projMat);
  } // loop over tables

  return;
}


template<typename SortPolicy, typename MetricType, typename eT>
void LSHSearch<SortPolicy, MetricType, eT>::
BuildSecondLevelHash(MatType& allKeysPointsMat)
{
  // obtain the hash weights for the second hash
  secondHashWeights = arma::floor(arma::randu(numProj) 
                                  * (double) secondHashSize);

  RowType hashVec = secondHashWeights.t() 
    * allKeysPointsMat.rows(0, numProj - 1);

  for (size_t i = 0; i < hashVec.n_elem; i++)
    hashVec[i] = (double)((size_t) hashVec[i] % secondHashSize);

  assert(hashVec.n_elem == referenceSet.n_cols * numTables);

  // start filling up the second hash table;
  secondHashTable.set_size(secondHashSize, bucketSize);
  secondHashTable.fill(referenceSet.n_cols);
  bucketContentSize.zeros(secondHashSize);

  // Initializing to nothing
  bucketRowInHashTable.set_size(secondHashSize);
  bucketRowInHashTable.fill(secondHashSize);

  size_t numRowsInTable = 0;

  for (size_t i = 0; i < hashVec.n_elem; i++)
  {
    size_t hashInd = (size_t) hashVec[i];
    size_t pointInd = (size_t) allKeysPointsMat(numProj, i);

    if (bucketContentSize[hashInd] == 0)
    {
      // start a new row for hash
      bucketRowInHashTable[hashInd] = numRowsInTable;
      secondHashTable(numRowsInTable, 0) = pointInd;

      numRowsInTable++;
    }
    else 
    {
      if (bucketContentSize[hashInd] < bucketSize)
      {
        // continue with an existing row
        size_t tableRow = bucketRowInHashTable[hashInd];
        secondHashTable(tableRow, bucketContentSize[hashInd]) = pointInd;
      }
      // else just ignore as suggested
    }

    if (bucketContentSize[hashInd] < bucketSize)
      bucketContentSize[hashInd]++;
  }

  // condense the second hash table
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
