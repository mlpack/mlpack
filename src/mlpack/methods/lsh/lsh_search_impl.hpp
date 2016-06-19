/**
 * @file lsh_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of the LSHSearch class.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

using std::cout; using std::endl; //TODO: remove

namespace mlpack {
namespace neighbor {

// Construct the object with random tables
template<typename SortPolicy>
LSHSearch<SortPolicy>::
LSHSearch(const arma::mat& referenceSet,
          const size_t numProj,
          const size_t numTables,
          const double hashWidthIn,
          const size_t secondHashSize,
          const size_t bucketSize) :
  referenceSet(NULL), // This will be set in Train().
  ownsSet(false),
  numProj(numProj),
  numTables(numTables),
  hashWidth(hashWidthIn),
  secondHashSize(secondHashSize),
  bucketSize(bucketSize),
  distanceEvaluations(0),
  maxThreads(omp_get_max_threads()),
  numThreadsUsed(1)
{
  // Pass work to training function.
  Train(referenceSet, numProj, numTables, hashWidthIn, secondHashSize,
      bucketSize);
}

// Construct the object with given tables
template<typename SortPolicy>
LSHSearch<SortPolicy>::
LSHSearch(const arma::mat& referenceSet,
          const arma::cube& projections,
          const double hashWidthIn,
          const size_t secondHashSize,
          const size_t bucketSize) :
  referenceSet(NULL), // This will be set in Train().
  ownsSet(false),
  numProj(projections.n_cols),
  numTables(projections.n_slices),
  hashWidth(hashWidthIn),
  secondHashSize(secondHashSize),
  bucketSize(bucketSize),
  distanceEvaluations(0),
  maxThreads(omp_get_max_threads()),
  numThreadsUsed(1)
{
  // Pass work to training function
  Train(referenceSet, numProj, numTables, hashWidthIn, secondHashSize,
      bucketSize, projections);
}

// Empty constructor.
template<typename SortPolicy>
LSHSearch<SortPolicy>::LSHSearch() :
    referenceSet(new arma::mat()), // Use an empty dataset.
    ownsSet(true),
    numProj(0),
    numTables(0),
    hashWidth(0),
    secondHashSize(99901),
    bucketSize(500),
    distanceEvaluations(0),
    maxThreads(omp_get_max_threads()),
    numThreadsUsed(1)
{
  // Nothing to do.
}

// Destructor.
template<typename SortPolicy>
LSHSearch<SortPolicy>::~LSHSearch()
{
  if (ownsSet)
    delete referenceSet;
}

// Train on a new reference set.
template<typename SortPolicy>
void LSHSearch<SortPolicy>::Train(const arma::mat& referenceSet,
                                  const size_t numProj,
                                  const size_t numTables,
                                  const double hashWidthIn,
                                  const size_t secondHashSize,
                                  const size_t bucketSize,
                                  const arma::cube &projection)
{
  // Set new reference set.
  if (this->referenceSet && ownsSet)
    delete this->referenceSet;
  this->referenceSet = &referenceSet;
  this->ownsSet = false;

  // Set new parameters.
  this->numProj = numProj;
  this->numTables = numTables;
  this->hashWidth = hashWidthIn;
  this->secondHashSize = secondHashSize;
  this->bucketSize = bucketSize;

  if (hashWidth == 0.0) // The user has not provided any value.
  {
    // Compute a heuristic hash width from the data.
    for (size_t i = 0; i < 25; i++)
    {
      size_t p1 = (size_t) math::RandInt(referenceSet.n_cols);
      size_t p2 = (size_t) math::RandInt(referenceSet.n_cols);

      hashWidth += std::sqrt(metric::EuclideanDistance::Evaluate(
          referenceSet.unsafe_col(p1), referenceSet.unsafe_col(p2)));
    }

    hashWidth /= 25;
  }

  Log::Info << "Hash width chosen as: " << hashWidth << std::endl;

  // Hash building procedure:
  // The first level hash for a single table outputs a 'numProj'-dimensional
  // integer key for each point in the set -- (key, pointID).  The key creation
  // details are presented below.

  // Step I: Prepare the second level hash.

  // Obtain the weights for the second hash.
  secondHashWeights = arma::floor(arma::randu(numProj) *
                                  (double) secondHashSize);

  // Instead of putting the points in the row corresponding to the bucket, we
  // chose the next empty row and keep track of the row in which the bucket
  // lies. This allows us to stack together and slice out the empty buckets at
  // the end of the hashing.
  bucketRowInHashTable.set_size(secondHashSize);
  bucketRowInHashTable.fill(secondHashSize);

  // Step II: The offsets for all projections in all tables.
  // Since the 'offsets' are in [0, hashWidth], we obtain the 'offsets'
  // as randu(numProj, numTables) * hashWidth.
  offsets.randu(numProj, numTables);
  offsets *= hashWidth;

  // Step III: Obtain the 'numProj' projections for each table.
  projections.clear(); // Reset projections vector.

  if (projection.n_slices == 0) // Randomly generate the tables.
  {
    // For L2 metric, 2-stable distributions are used, and the normal Z ~ N(0,
    // 1) is a 2-stable distribution.

    // Build numTables random tables arranged in a cube.
    projections.randn(referenceSet.n_rows, numProj, numTables);
  }
  else if (projection.n_slices == numTables) // Take user-defined tables.
  {
    projections = projection;
  }
  else // The user gave something wrong.
  {
    throw std::invalid_argument("LSHSearch::Train(): number of projection "
        "tables provided must be equal to numProj");
  }

  // We will store the second hash vectors in this matrix; the second hash
  // vector for table i will be held in row i.
  arma::Mat<size_t> secondHashVectors(numTables, referenceSet.n_cols);

  for (size_t i = 0; i < numTables; i++)
  {
    // Step IV: create the 'numProj'-dimensional key for each point in each
    // table.

    // The following code performs the task of hashing each point to a
    // 'numProj'-dimensional integer key.  Hence you get a ('numProj' x
    // 'referenceSet.n_cols') key matrix.
    //
    // For a single table, let the 'numProj' projections be denoted by 'proj_i'
    // and the corresponding offset be 'offset_i'.  Then the key of a single
    // point is obtained as:
    // key = { floor( (<proj_i, point> + offset_i) / 'hashWidth' ) forall i }
    arma::mat offsetMat = arma::repmat(offsets.unsafe_col(i), 1,
                                       referenceSet.n_cols);
    arma::mat hashMat = projections.slice(i).t() * (referenceSet);
    hashMat += offsetMat;
    hashMat /= hashWidth;

    // Step V: Putting the points in the 'secondHashTable' by hashing the key.
    // Now we hash every key, point ID to its corresponding bucket.
    secondHashVectors.row(i) = arma::conv_to<arma::Row<size_t>>::from(
        secondHashWeights.t() * arma::floor(hashMat));
  }

  // Normalize hashes (take modulus with secondHashSize).
  secondHashVectors.transform([secondHashSize](size_t val)
      { return val % secondHashSize; });

  // Now, using the hash vectors for each table, count the number of rows we
  // have in the second hash table.
  arma::Row<size_t> secondHashBinCounts(secondHashSize, arma::fill::zeros);
  for (size_t i = 0; i < secondHashVectors.n_elem; ++i)
    secondHashBinCounts[secondHashVectors[i]]++;

  // Enforce the maximum bucket size.
  const size_t effectiveBucketSize = (bucketSize == 0) ? SIZE_MAX : bucketSize;
  secondHashBinCounts.transform([effectiveBucketSize](size_t val)
      { return std::min(val, effectiveBucketSize); });

  const size_t numRowsInTable = arma::accu(secondHashBinCounts > 0);
  bucketContentSize.zeros(numRowsInTable);
  secondHashTable.resize(numRowsInTable);

  // Next we must assign each point in each table to the right second hash
  // table.
  size_t currentRow = 0;
  for (size_t i = 0; i < numTables; ++i)
  {
    // Insert the point in the corresponding row to its bucket in the
    // 'secondHashTable'.
    for (size_t j = 0; j < secondHashVectors.n_cols; j++)
    {
      // This is the bucket number.
      size_t hashInd = (size_t) secondHashVectors(i, j);
      // The point ID is 'j'.

      // If this is currently an empty bucket, start a new row keep track of
      // which row corresponds to the bucket.
      const size_t maxSize = secondHashBinCounts[hashInd];
      if (bucketRowInHashTable[hashInd] == secondHashSize)
      {
        bucketRowInHashTable[hashInd] = currentRow;
        secondHashTable[currentRow].set_size(maxSize);
        currentRow++;
      }

      // If this vector in the hash table is not full, add the point.
      const size_t index = bucketRowInHashTable[hashInd];
      if (bucketContentSize[index] < maxSize)
        secondHashTable[index](bucketContentSize[index]++) = j;

    } // Loop over all points in the reference set.
  } // Loop over tables.

  Log::Info << "Final hash table size: " << numRowsInTable << " rows, with a "
            << "maximum length of " << arma::max(secondHashBinCounts) << ", "
            << "totaling " << arma::accu(secondHashBinCounts) << " elements."
            << std::endl;
}

template<typename SortPolicy>
void LSHSearch<SortPolicy>::InsertNeighbor(arma::mat& distances,
                                           arma::Mat<size_t>& neighbors,
                                           const size_t queryIndex,
                                           const size_t pos,
                                           const size_t neighbor,
                                           const double distance) const
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (distances.n_rows - 1))
  {
    const size_t len = (distances.n_rows - 1) - pos;
    memmove(distances.colptr(queryIndex) + (pos + 1),
        distances.colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(neighbors.colptr(queryIndex) + (pos + 1),
        neighbors.colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  distances(pos, queryIndex) = distance;
  neighbors(pos, queryIndex) = neighbor;
}

// Base case where the query set is the reference set.  (So, we can't return
// ourselves as the nearest neighbor.)
template<typename SortPolicy>
inline force_inline
void LSHSearch<SortPolicy>::BaseCase(const size_t queryIndex,
                                     const size_t referenceIndex,
                                     arma::Mat<size_t>& neighbors,
                                     arma::mat& distances) const
{
  // If the points are the same, we can't continue.
  if (queryIndex == referenceIndex)
    return;

  const double distance = metric::EuclideanDistance::Evaluate(
      referenceSet->unsafe_col(queryIndex),
      referenceSet->unsafe_col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distances.unsafe_col(queryIndex);
  arma::Col<size_t> queryIndices = neighbors.unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, queryIndices,
      distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(distances, neighbors, queryIndex, insertPosition,
        referenceIndex, distance);
}

// Base case for bichromatic search.
template<typename SortPolicy>
inline force_inline
void LSHSearch<SortPolicy>::BaseCase(const size_t queryIndex,
                                     const size_t referenceIndex,
                                     const arma::mat& querySet,
                                     arma::Mat<size_t>& neighbors,
                                     arma::mat& distances) const
{
  const double distance = metric::EuclideanDistance::Evaluate(
      querySet.unsafe_col(queryIndex),
      referenceSet->unsafe_col(referenceIndex));

  // If this distance is better than any of the current candidates, the
  // SortDistance() function will give us the position to insert it into.
  arma::vec queryDist = distances.unsafe_col(queryIndex);
  arma::Col<size_t> queryIndices = neighbors.unsafe_col(queryIndex);
  size_t insertPosition = SortPolicy::SortDistance(queryDist, queryIndices,
      distance);

  // SortDistance() returns (size_t() - 1) if we shouldn't add it.
  if (insertPosition != (size_t() - 1))
    InsertNeighbor(distances, neighbors, queryIndex, insertPosition,
        referenceIndex, distance);
}

template<typename SortPolicy>
template<typename VecType>
void LSHSearch<SortPolicy>::ReturnIndicesFromTable(
    const VecType& queryPoint,
    arma::uvec& referenceIndices,
    size_t numTablesToSearch) const
{
  // Decide on the number of tables to look into.
  if (numTablesToSearch == 0) // If no user input is given, search all.
    numTablesToSearch = numTables;

  // Sanity check to make sure that the existing number of tables is not
  // exceeded.
  if (numTablesToSearch > numTables)
    numTablesToSearch = numTables;

  // Hash the query in each of the 'numTablesToSearch' hash tables using the
  // 'numProj' projections for each table. This gives us 'numTablesToSearch'
  // keys for the query where each key is a 'numProj' dimensional integer
  // vector.

  // Compute the projection of the query in each table.
  arma::mat allProjInTables(numProj, numTablesToSearch);
  for (size_t i = 0; i < numTablesToSearch; i++)
    //allProjInTables.unsafe_col(i) = projections[i].t() * queryPoint;
    allProjInTables.unsafe_col(i) = projections.slice(i).t() * queryPoint;
  allProjInTables += offsets.cols(0, numTablesToSearch - 1);
  allProjInTables /= hashWidth;

  // Compute the hash value of each key of the query into a bucket of the
  // 'secondHashTable' using the 'secondHashWeights'.
  arma::rowvec hashVec = secondHashWeights.t() * arma::floor(allProjInTables);

  for (size_t i = 0; i < hashVec.n_elem; i++)
    hashVec[i] = (double) ((size_t) hashVec[i] % secondHashSize);

  Log::Assert(hashVec.n_elem == numTablesToSearch);

  // Count number of points hashed in the same bucket as the query.
  size_t maxNumPoints = 0;
  for (size_t i = 0; i < numTablesToSearch; ++i)
  {
    const size_t hashInd = (size_t) hashVec[i];
    const size_t tableRow = bucketRowInHashTable[hashInd];
    if (tableRow != secondHashSize)
      maxNumPoints += bucketContentSize[tableRow];
  }

  // There are two ways to proceed here:
  // Either allocate a maxNumPoints-size vector, place all candidates, and run
  // unique on the vector to discard duplicates.
  // Or allocate a referenceSet->n_cols size vector (i.e. number of reference
  // points) of zeros, and mark found indices as 1.
  // Option 1 runs faster for small maxNumPoints but worse for larger values, so
  // we choose based on a heuristic.
  const float cutoff = 0.1;
  const float selectivity = static_cast<float>(maxNumPoints) /
      static_cast<float>(referenceSet->n_cols);

  if (selectivity > cutoff)
  {
    // Heuristic: larger maxNumPoints means we should use find() because it
    // should be faster.
    // Reference points hashed in the same bucket as the query are set to >0.
    arma::Col<size_t> refPointsConsidered;
    refPointsConsidered.zeros(referenceSet->n_cols);

    for (size_t i = 0; i < hashVec.n_elem; ++i)
    {
      const size_t hashInd = (size_t) hashVec[i];
      const size_t tableRow = bucketRowInHashTable[hashInd];

      // Pick the indices in the bucket corresponding to 'hashInd'.
      if (tableRow != secondHashSize)
        for (size_t j = 0; j < bucketContentSize[tableRow]; j++)
          refPointsConsidered[secondHashTable[tableRow](j)]++;
    }

    // Only keep reference points found in at least one bucket.
    referenceIndices = arma::find(refPointsConsidered > 0);
    return;
  }
  else
  {
    // Heuristic: smaller maxNumPoints means we should use unique() because it
    // should be faster.
    // Allocate space for the query's potential neighbors.
    arma::uvec refPointsConsideredSmall;
    refPointsConsideredSmall.zeros(maxNumPoints);

    // Retrieve candidates.
    size_t start = 0;
    for (size_t i = 0; i < numTablesToSearch; ++i) // For all tables
    {
      const size_t hashInd = (size_t) hashVec[i]; // Find the query's bucket.
      const size_t tableRow = bucketRowInHashTable[hashInd];

      // Store all secondHashTable points in the candidates set.
      if (tableRow != secondHashSize)
        for (size_t j = 0; j < bucketContentSize[tableRow]; ++j)
          refPointsConsideredSmall(start++) = secondHashTable[tableRow][j];
    }

    // Only keep unique candidates.
    referenceIndices = arma::unique(refPointsConsideredSmall);
    return;
  }
}

// Search for nearest neighbors in a given query set.
template<typename SortPolicy>
void LSHSearch<SortPolicy>::Search(const arma::mat& querySet,
                                   const size_t k,
                                   arma::Mat<size_t>& resultingNeighbors,
                                   arma::mat& distances,
                                   const size_t numTablesToSearch)
{
  // Ensure the dimensionality of the query set is correct.
  if (querySet.n_rows != referenceSet->n_rows)
  {
    std::ostringstream oss;
    oss << "LSHSearch::Search(): dimensionality of query set ("
        << querySet.n_rows << ") is not equal to the dimensionality the model "
        << "was trained on (" << referenceSet->n_rows << ")!" << std::endl;
    throw std::invalid_argument(oss.str());
  }

  if (k > referenceSet->n_cols)
  {
    std::ostringstream oss;
    oss << "LSHSearch::Search(): requested " << k << " approximate nearest "
        << "neighbors, but reference set has " << referenceSet->n_cols
        << " points!" << std::endl;
    throw std::invalid_argument(oss.str());
  }

  // Set the size of the neighbor and distance matrices.
  resultingNeighbors.set_size(k, querySet.n_cols);
  distances.set_size(k, querySet.n_cols);
  distances.fill(SortPolicy::WorstDistance());
  resultingNeighbors.fill(referenceSet->n_cols);

  // If the user asked for 0 nearest neighbors... uh... we're done.
  if (k == 0)
    return;

  size_t avgIndicesReturned = 0;

  Timer::Start("computing_neighbors");

  // Go through every query point sequentially.
  for (size_t i = 0; i < querySet.n_cols; i++)
  {
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(querySet.col(i), refIndices, numTablesToSearch);

    // An informative book-keeping for the number of neighbor candidates
    // returned on average.
    avgIndicesReturned += refIndices.n_elem;

    // Sequentially go through all the candidates and save the best 'k'
    // candidates.
    for (size_t j = 0; j < refIndices.n_elem; j++)
      BaseCase(i, (size_t) refIndices[j], querySet, resultingNeighbors,
          distances);
  }

  Timer::Stop("computing_neighbors");

  distanceEvaluations += avgIndicesReturned;
  avgIndicesReturned /= querySet.n_cols;
  Log::Info << avgIndicesReturned << " distinct indices returned on average." <<
      std::endl;
}

// Search for approximate neighbors of the reference set.
template<typename SortPolicy>
void LSHSearch<SortPolicy>::
Search(const size_t k,
       arma::Mat<size_t>& resultingNeighbors,
       arma::mat& distances,
       const size_t numTablesToSearch)
{
  // This is monochromatic search; the query set is the reference set.
  resultingNeighbors.set_size(k, referenceSet->n_cols);
  distances.set_size(k, referenceSet->n_cols);
  distances.fill(SortPolicy::WorstDistance());
  resultingNeighbors.fill(referenceSet->n_cols);


  size_t avgIndicesReturned = 0;

  Timer::Start("computing_neighbors");

  // Parallelization allows us to process more than one query at a time. To
  // control workload and thread access, we use numThreadsUsed and maxThreads to
  // make sure we only use as many threads as the user specified.
  #pragma omp parallel for \
    if (numThreadsUsed <= maxThreads) \
    num_threads (maxThreads-numThreadsUsed)\
    shared(avgIndicesReturned, resultingNeighbors, distances) \
    schedule(dynamic)
  // Go through every query point.
  for (size_t i = 0; i < referenceSet->n_cols; i++)
  {
    // Master thread updates the number of threads used
    if (i == 0 && omp_get_thread_num() == 0)
    {
      numThreadsUsed+=omp_get_num_threads(); //
      Log::Info 
        << "Using "<< numThreadsUsed << " threads to process queries." << endl;
    }
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(referenceSet->col(i), refIndices, numTablesToSearch);

    // An informative book-keeping for the number of neighbor candidates
    // returned on average.
    // Make atomic to avoid race conditions when multiple threads are running
    #pragma omp atomic
    avgIndicesReturned += refIndices.n_elem;

    // Sequentially go through all the candidates and save the best 'k'
    // candidates.
    for (size_t j = 0; j < refIndices.n_elem; j++)
      BaseCase(i, (size_t) refIndices[j], resultingNeighbors, distances);

  }

  // parallel region over, reset number of threads to 1
  numThreadsUsed = omp_get_num_threads();


  Timer::Stop("computing_neighbors");

  distanceEvaluations += avgIndicesReturned;
  avgIndicesReturned /= referenceSet->n_cols;
  Log::Info << avgIndicesReturned << " distinct indices returned on average." <<
      std::endl;
}

template<typename SortPolicy>
template<typename Archive>
void LSHSearch<SortPolicy>::Serialize(Archive& ar,
                                      const unsigned int version)
{
  using data::CreateNVP;

  // If we are loading, we are going to own the reference set.
  if (Archive::is_loading::value)
  {
    if (ownsSet)
      delete referenceSet;
    ownsSet = true;
  }
  ar & CreateNVP(referenceSet, "referenceSet");

  ar & CreateNVP(numProj, "numProj");
  ar & CreateNVP(numTables, "numTables");

  // Delete existing projections, if necessary.
  if (Archive::is_loading::value)
    projections.reset();

  // Backward compatibility: older versions of LSHSearch stored the projection
  // tables in a std::vector<arma::mat>.
  if (version == 0)
  {
    std::vector<arma::mat> tmpProj;
    ar & CreateNVP(tmpProj, "projections");

    projections.set_size(tmpProj[0].n_rows, tmpProj[0].n_cols, tmpProj.size());
    for (size_t i = 0; i < tmpProj.size(); ++i)
      projections.slice(i) = tmpProj[i];
  }
  else
  {
    ar & CreateNVP(projections, "projections");
  }

  ar & CreateNVP(offsets, "offsets");
  ar & CreateNVP(hashWidth, "hashWidth");
  ar & CreateNVP(secondHashSize, "secondHashSize");
  ar & CreateNVP(secondHashWeights, "secondHashWeights");
  ar & CreateNVP(bucketSize, "bucketSize");
  // needs specific handling for new version

  // Backward compatibility: in older versions of LSHSearch, the secondHashTable
  // was stored as an arma::Mat<size_t>.  So we need to properly load that, then
  // prune it down to size.
  if (version == 0)
  {
    arma::Mat<size_t> tmpSecondHashTable;
    ar & CreateNVP(tmpSecondHashTable, "secondHashTable");

    // The old secondHashTable was stored in row-major format, so we transpose
    // it.
    tmpSecondHashTable = tmpSecondHashTable.t();

    secondHashTable.resize(tmpSecondHashTable.n_cols);
    for (size_t i = 0; i < tmpSecondHashTable.n_cols; ++i)
    {
      // Find length of each column.  We know we are at the end of the list when
      // the value referenceSet->n_cols is seen.

      size_t len = 0;
      for ( ; len < tmpSecondHashTable.n_rows; ++len)
        if (tmpSecondHashTable(len, i) == referenceSet->n_cols)
          break;

      // Set the size of the new column correctly.
      secondHashTable[i].set_size(len);
      for (size_t j = 0; j < len; ++j)
        secondHashTable[i](j) = tmpSecondHashTable(j, i);
    }
  }
  else
  {
    size_t tables;
    if (Archive::is_saving::value)
      tables = secondHashTable.size();
    ar & CreateNVP(tables, "numSecondHashTables");

    // Set size of second hash table if needed.
    if (Archive::is_loading::value)
    {
      secondHashTable.clear();
      secondHashTable.resize(tables);
    }

    for (size_t i = 0; i < secondHashTable.size(); ++i)
    {
      std::ostringstream oss;
      oss << "secondHashTable" << i;
      ar & CreateNVP(secondHashTable[i], oss.str());
    }
  }

  // Backward compatibility: old versions of LSHSearch held bucketContentSize
  // for all possible buckets (of size secondHashSize), but now we hold a
  // compressed representation.
  if (version == 0)
  {
    // The vector was stored in the old uncompressed form.  So we need to shrink
    // it.  But we can't do that until we have bucketRowInHashTable, so we also
    // have to load that.
    arma::Col<size_t> tmpBucketContentSize;
    ar & CreateNVP(tmpBucketContentSize, "bucketContentSize");
    ar & CreateNVP(bucketRowInHashTable, "bucketRowInHashTable");

    // Compress into a smaller vector by just dropping all of the zeros.
    bucketContentSize.set_size(secondHashTable.size());
    for (size_t i = 0; i < tmpBucketContentSize.n_elem; ++i)
      if (tmpBucketContentSize[i] > 0)
        bucketContentSize[bucketRowInHashTable[i]] = tmpBucketContentSize[i];
  }
  else
  {
    ar & CreateNVP(bucketContentSize, "bucketContentSize");
    ar & CreateNVP(bucketRowInHashTable, "bucketRowInHashTable");
  }

  ar & CreateNVP(distanceEvaluations, "distanceEvaluations");
}

} // namespace neighbor
} // namespace mlpack

#endif
