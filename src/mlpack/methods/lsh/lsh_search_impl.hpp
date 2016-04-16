/**
 * @file lsh_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of the LSHSearch class.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

// Construct the object.
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
  distanceEvaluations(0)
{
  // Pass work to training function.
  Train(referenceSet, numProj, numTables, hashWidthIn, secondHashSize,
      bucketSize);
}

// Empty constructor.
template<typename SortPolicy>
LSHSearch<SortPolicy>::LSHSearch() :
    referenceSet(new arma::mat()), // empty dataset
    ownsSet(true),
    numProj(0),
    numTables(0),
    hashWidth(0),
    secondHashSize(99901),
    bucketSize(500),
    distanceEvaluations(0)
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
                                  const size_t bucketSize)
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

  BuildHash();
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
    allProjInTables.unsafe_col(i) = projections[i].t() * queryPoint;
  allProjInTables += offsets.cols(0, numTablesToSearch - 1);
  allProjInTables /= hashWidth;

  // Compute the hash value of each key of the query into a bucket of the
  // 'secondHashTable' using the 'secondHashWeights'.
  arma::rowvec hashVec = secondHashWeights.t() * arma::floor(allProjInTables);

  for (size_t i = 0; i < hashVec.n_elem; i++)
    hashVec[i] = (double) ((size_t) hashVec[i] % secondHashSize);

  Log::Assert(hashVec.n_elem == numTablesToSearch);

  //TODO: The code below is unoptimized with respect to memory, since
  //refPointsConsidered will be sparse. Alternatively, we could count the number
  //of points hashed in the same bucket as the query and allocate only that many
  //spaces

  //Count number of points hashed in the same bucket as the query
  size_t maxNumPoints = 0;
  for (size_t i = 0; i < numTablesToSearch; ++i) //For all tables
  {
    size_t hashInd = (size_t) hashVec[i]; //find query's bucket
    maxNumPoints += bucketContentSize[hashInd]; //count bucket contents
  }

  //Allocate space for query's potential neighbors
  arma::uvec refPointsConsideredSmall;
  refPointsConsideredSmall.zeros(maxNumPoints);

  //Retrieve candidates
  size_t start = 0;
  for (size_t i = 0; i < numTablesToSearch; ++i) //For all tables
  {
    size_t hashInd = (size_t) hashVec[i]; //find query's bucket

    //tableRow hash indices corresponding to query
    size_t tableRow = bucketRowInHashTable[hashInd];
    assert(tableRow < secondHashSize);
    assert(tableRow < secondHashTable.n_rows);

    //this for-loop could be replaced with a vector slice (TODO)
    //store all secondHashTable points in the candidates set
    for (size_t j = 0; j < bucketContentSize[hashInd]; ++j)
      refPointsConsideredSmall(start++) = secondHashTable(tableRow, j);
  }

  //Only keep unique candidates
  referenceIndices = arma::unique(refPointsConsideredSmall);
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

  // Go through every query point sequentially.
  for (size_t i = 0; i < referenceSet->n_cols; i++)
  {
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(referenceSet->col(i), refIndices, numTablesToSearch);

    // An informative book-keeping for the number of neighbor candidates
    // returned on average.
    avgIndicesReturned += refIndices.n_elem;

    // Sequentially go through all the candidates and save the best 'k'
    // candidates.
    for (size_t j = 0; j < refIndices.n_elem; j++)
      BaseCase(i, (size_t) refIndices[j], resultingNeighbors, distances);
  }

  Timer::Stop("computing_neighbors");

  distanceEvaluations += avgIndicesReturned;
  avgIndicesReturned /= referenceSet->n_cols;
  Log::Info << avgIndicesReturned << " distinct indices returned on average." <<
      std::endl;
}

template<typename SortPolicy>
void LSHSearch<SortPolicy>::BuildHash()
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

  // Step I: Prepare the second level hash.

  // Obtain the weights for the second hash.
  secondHashWeights = arma::floor(arma::randu(numProj) *
                                  (double) secondHashSize);

  // The 'secondHashTable' is initially an empty matrix of size
  // ('secondHashSize' x 'bucketSize'). But by only filling the buckets
  // as points land in them allows us to shrink the size of the
  // 'secondHashTable' at the end of the hashing.

  // Fill the second hash table n = referenceSet.n_cols.  This is because no
  // point has index 'n' so the presence of this in the bucket denotes that
  // there are no more points in this bucket.
  secondHashTable.set_size(secondHashSize, bucketSize);
  secondHashTable.fill(referenceSet->n_cols);

  // Keep track of the size of each bucket in the hash.  At the end of hashing
  // most buckets will be empty.
  bucketContentSize.zeros(secondHashSize);

  // Instead of putting the points in the row corresponding to the bucket, we
  // chose the next empty row and keep track of the row in which the bucket
  // lies. This allows us to stack together and slice out the empty buckets at
  // the end of the hashing.
  bucketRowInHashTable.set_size(secondHashSize);
  bucketRowInHashTable.fill(secondHashSize);

  // Keep track of number of non-empty rows in the 'secondHashTable'.
  size_t numRowsInTable = 0;

  // Step II: The offsets for all projections in all tables.
  // Since the 'offsets' are in [0, hashWidth], we obtain the 'offsets'
  // as randu(numProj, numTables) * hashWidth.
  offsets.randu(numProj, numTables);
  offsets *= hashWidth;

  // Step III: Create each hash table in the first level hash one by one and
  // putting them directly into the 'secondHashTable' for memory efficiency.
  projections.clear(); // Reset projections vector.
  for (size_t i = 0; i < numTables; i++)
  {
    // Step IV: Obtain the 'numProj' projections for each table.

    // For L2 metric, 2-stable distributions are used, and
    // the normal Z ~ N(0, 1) is a 2-stable distribution.
    arma::mat projMat;
    projMat.randn(referenceSet->n_rows, numProj);

    // Save the projection matrix for querying.
    projections.push_back(projMat);

    // Step V: create the 'numProj'-dimensional key for each point in each
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
                                       referenceSet->n_cols);
    arma::mat hashMat = projMat.t() * (*referenceSet);
    hashMat += offsetMat;
    hashMat /= hashWidth;

    // Step VI: Putting the points in the 'secondHashTable' by hashing the key.
    // Now we hash every key, point ID to its corresponding bucket.
    arma::rowvec secondHashVec = secondHashWeights.t() * arma::floor(hashMat);

    // This gives us the bucket for the corresponding point ID.
    for (size_t j = 0; j < secondHashVec.n_elem; j++)
      secondHashVec[j] = (double)((size_t) secondHashVec[j] % secondHashSize);

    Log::Assert(secondHashVec.n_elem == referenceSet->n_cols);

    // Insert the point in the corresponding row to its bucket in the
    // 'secondHashTable'.
    for (size_t j = 0; j < secondHashVec.n_elem; j++)
    {
      // This is the bucket number.
      size_t hashInd = (size_t) secondHashVec[j];
      // The point ID is 'j'.

      // If this is currently an empty bucket, start a new row keep track of
      // which row corresponds to the bucket.
      if (bucketContentSize[hashInd] == 0)
      {
        // Start a new row for hash.
        bucketRowInHashTable[hashInd] = numRowsInTable;
        secondHashTable(numRowsInTable, 0) = j;

        numRowsInTable++;
      }

      else
      {
        // If bucket is already present in the 'secondHashTable', find the
        // corresponding row and insert the point ID in this row unless the
        // bucket is full, in which case, do nothing.
        if (bucketContentSize[hashInd] < bucketSize)
          secondHashTable(bucketRowInHashTable[hashInd],
                          bucketContentSize[hashInd]) = j;
      }

      // Increment the count of the points in this bucket.
      if (bucketContentSize[hashInd] < bucketSize)
        bucketContentSize[hashInd]++;
    } // Loop over all points in the reference set.
  } // Loop over tables.

  // Step VII: Condensing the 'secondHashTable'.
  size_t maxBucketSize = 0;
  for (size_t i = 0; i < bucketContentSize.n_elem; i++)
    if (bucketContentSize[i] > maxBucketSize)
      maxBucketSize = bucketContentSize[i];

  Log::Info << "Final hash table size: (" << numRowsInTable << " x "
            << maxBucketSize << ")" << std::endl;
  secondHashTable.resize(numRowsInTable, maxBucketSize);
}

template<typename SortPolicy>
template<typename Archive>
void LSHSearch<SortPolicy>::Serialize(Archive& ar,
                                      const unsigned int /* version */)
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
    projections.clear();

  ar & CreateNVP(projections, "projections");
  ar & CreateNVP(offsets, "offsets");
  ar & CreateNVP(hashWidth, "hashWidth");
  ar & CreateNVP(secondHashSize, "secondHashSize");
  ar & CreateNVP(secondHashWeights, "secondHashWeights");
  ar & CreateNVP(bucketSize, "bucketSize");
  ar & CreateNVP(secondHashTable, "secondHashTable");
  ar & CreateNVP(bucketContentSize, "bucketContentSize");
  ar & CreateNVP(bucketRowInHashTable, "bucketRowInHashTable");
  ar & CreateNVP(distanceEvaluations, "distanceEvaluations");
}

} // namespace neighbor
} // namespace mlpack

#endif
