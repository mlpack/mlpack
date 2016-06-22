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
  distanceEvaluations(0)
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
  distanceEvaluations(0)
{
  // Pass work to training function
  Train(referenceSet, numProj, numTables, hashWidthIn, secondHashSize,
      bucketSize, projections);
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

  // The 'secondHashTable' is initially an empty matrix of size
  // ('secondHashSize' x 'bucketSize'). But by only filling the buckets as
  // points land in them allows us to shrink the size of the 'secondHashTable'
  // at the end of the hashing.

  // Fill the second hash table n = referenceSet.n_cols.  This is because no
  // point has index 'n' so the presence of this in the bucket denotes that
  // there are no more points in this bucket.
  secondHashTable.set_size(secondHashSize, bucketSize);
  secondHashTable.fill(referenceSet.n_cols);

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
    arma::rowvec secondHashVec = secondHashWeights.t() * arma::floor(hashMat);

    // This gives us the bucket for the corresponding point ID.
    for (size_t j = 0; j < secondHashVec.n_elem; j++)
      secondHashVec[j] = (double) ((size_t) secondHashVec[j] % secondHashSize);

    Log::Assert(secondHashVec.n_elem == referenceSet.n_cols);

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

  // Step VI: Condensing the 'secondHashTable'.
  size_t maxBucketSize = 0;
  for (size_t i = 0; i < bucketContentSize.n_elem; i++)
    if (bucketContentSize[i] > maxBucketSize)
      maxBucketSize = bucketContentSize[i];

  Log::Info << "Final hash table size: (" << numRowsInTable << " x "
            << maxBucketSize << ")" << std::endl;
  secondHashTable.resize(numRowsInTable, maxBucketSize);
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

//Returns the score of a perturbation vector generated by perturbation set A.
//The score of a pertubation set (vector) is the sum of scores of the
//participating actions.
inline double perturbationScore(const std::vector<size_t>& A,
                                const arma::vec& scores)
{
  double score = 0.0;
  for (size_t i = 0; i < A.size(); ++i)
    score+=scores[A[i]];
  return score;
}

// Inline function used by GetAdditionalProbingBins. The vector shift operation
// replaces the largest element of a vector A with (largest element) + 1.
inline void perturbationShift(std::vector<size_t>& A)
{
  size_t max_pos = 0;
  size_t max = A[0];
  for (size_t i = 1; i < A.size(); ++i)
  {
    if (A[i] > max)
    {
      max = A[i];
      max_pos = i;
    }
  }
  A[max_pos]++;
}

// Inline function used by GetAdditionalProbingBins. The vector expansion
// operation adds the element [1 + (largest_element)] to a vector A, where
// largest_element is the largest element of A.
inline void perturbationExpand(std::vector<size_t>& A)
{
  size_t max = A[0];
  for (size_t i = 1; i < A.size(); ++i)
    if (A[i] > max)
      max = A[i];
  A.push_back(max + 1);
}

// Return true if perturbation set A is valid. A perturbation set is invalid if
// it contains two (or more) actions for the same dimension or dimensions that
// are larger than the queryCode's dimensions.
inline bool perturbationValid(const std::vector<size_t>& A,
                              const size_t numProj)
{
  // Stack allocation and initialization to 0 (bool check[numProj] = {0}) made
  // some compilers complain, and std::vector might even be compressed (depends
  // on implementation) so this saves some space.
  std::vector<bool> check(numProj);

  for (size_t i = 0; i < A.size(); ++i)
  {
    // Check that we only use valid dimensions. If not, vector is not valid.
    if (A[i] >= 2 * numProj)
      return false;

    // Check that we only see each dimension once. If not, vector is not valid.
    if (check[A[i] % numProj ] == 0)
      check[A[i] % numProj ] = 1;
    else
      return false;
  }
  return true;
}


// Compute additional probing bins for a query
template<typename SortPolicy>
void LSHSearch<SortPolicy>::GetAdditionalProbingBins(
                            const arma::vec& queryCode,
                            const arma::vec& queryCodeNotFloored,
                            const size_t T,
                            arma::mat& additionalProbingBins) const
{

  // No additional bins requested. Our work is done.
  if (T == 0)
    return;

  // Each column of additionalProbingBins is the code of a bin.
  additionalProbingBins.set_size(numProj, T);
  
  // Copy the query's code, then in the end we will  add/subtract according 
  // to perturbations we calculated.
  for (size_t c = 0; c < T; ++c)
    additionalProbingBins.col(c) = queryCode;


  // Calculate query point's projection position
  arma::mat projection = queryCodeNotFloored;

  // Use projection to calculate query's distance from hash limits
  arma::vec limLow = projection - queryCode * hashWidth;
  arma::vec limHigh = hashWidth - limLow;

  // calculate scores = distances^2
  arma::vec scores(2 * numProj);
  scores.rows(0, numProj - 1) = arma::pow(limLow, 2);
  scores.rows(numProj, (2 * numProj) - 1) = arma::pow(limHigh, 2);

  // actions vector shows what transformation to apply to a coordinate
  arma::Col<short int> actions(2 * numProj); // will be [-1 ... 1 ...]

  actions.rows(0, numProj - 1) = // first numProj rows
    -1 * arma::ones< arma::Col<short int> > (numProj); // -1s

  actions.rows(numProj, (2 * numProj) - 1) = // last numProj rows
    arma::ones< arma::Col<short int> > (numProj); // 1s


  // acting dimension vector shows which coordinate to transform according to
  // actions described by actions vector
  arma::Col<size_t> positions(2 * numProj); // will be [0 1 2 ... 0 1 2 ...]
  positions.rows(0, numProj - 1) =
    arma::linspace< arma::Col<size_t> >(0, numProj - 1, numProj);
  positions.rows(numProj, 2 * numProj - 1) =
    arma::linspace< arma::Col<size_t> >(0, numProj - 1, numProj);

  // optimization: no need to create heap for 1 or 2 codes
  if (T <= 2)
  {

    // First, find location of minimum score, generate 1 perturbation vector,
    // and add its code to additionalProbingBins column 0.

    // find location and value of smallest element of scores vector
    double minscore = scores[0];
    size_t minloc = 0;
    for (size_t s = 1; s < (2 * numProj); ++s)
    {
      if (minscore > scores[s])
      {
        minscore = scores[s];
        minloc = s;
      }
    }
    
    // add or subtract 1 to dimension corresponding to minimum score
    additionalProbingBins(positions[minloc], 0) += actions[minloc];
    if (T == 1)
      return; //done if asked for only 1 code

    // Now, find location of second smallest score and generate one more vector.
    // The second perturbation vector still can't comprise of more than one
    // change in the bin codes. This is because of the way perturbation vectors
    // are generated: First we create the one with the smallest score (Ao) and
    // then we either add 1 extra dimension to it (Ae) or shift it by one (As).
    // Since As contains the second smallest score, and Ae contains both the
    // smallest and the second smallest, it's obvious that score(Ae) >
    // score(As). Therefore the second perturbation vector is ALWAYS the vector
    // containing only the second-lowest scoring perturbation.
    
    
    double minscore2 = scores[0];
    size_t minloc2 = 0;
    for (size_t s = 0; s < (2 * numProj); ++s) // here we can't start from 1
    {
      if ( minscore2 > scores[s] && s != minloc) //second smallest
      {
        minscore2 = scores[s];
        minloc2 = s;
      }
    }

    // add or subtract 1 to create second-lowest scoring vector
    additionalProbingBins(positions[minloc2], 1) += actions[minloc2];
    return;
  }
  // General case: more than 2 perturbation vectors require use of minheap.

  // sort everything in increasing order
  arma::Col<long long unsigned int> sortidx = arma::sort_index(scores);
  scores = scores(sortidx);
  actions = actions(sortidx);
  positions = positions(sortidx);


  // Theory:
  // From the paper: This is the part that creates the probing sequence.
  // A probing sequence is a sequence of T probing bins where a query's
  // neighbors are most likely to be. Likelihood is dependent only on a bin's
  // score, which is the sum of scores of all dimension-action pairs, so we
  // need to calculate the T smallest sums of scores that are not conflicting.
  //
  // Method:
  // Store each perturbation set (pair of (dimension, action)) in a
  // std::vector. Create a minheap of scores, with each node pointing to its
  // relevant perturbation set. Each perturbation set popped from the minheap
  // is the next most likely perturbation set.
  // Transform perturbation set to perturbation vector by setting the
  // dimensions specified by the set to queryCode+action (action is {-1, 1}).

  std::vector<size_t> Ao;
  Ao.push_back(0); // initial perturbation holds smallest score (0 if sorted)

  std::vector< std::vector<size_t> > perturbationSets;
  perturbationSets.push_back(Ao); // storage of perturbation sets

  std::priority_queue<
    std::pair<double, size_t>,        // contents: pairs of (score, index)
    std::vector<                      // container: vector of pairs
      std::pair<double, size_t>
      >,
    std::greater< std::pair<double, size_t> > // comparator of pairs(compare scores)
  > minHeap; // our minheap

  // Start by adding the lowest scoring set to the minheap
  // std::pair<double, size_t> pair0( perturbationScore(Ao, scores), 0 );
  // minHeap.push(pair0);
  minHeap.push( std::make_pair(perturbationScore(Ao, scores), 0) );

  // loop invariable: after pvec iterations, additionalProbingBins contains pvec
  // valid codes of the lowest-scoring bins (bins most likely to contain
  // neighbors of the query).
  for (size_t pvec = 0; pvec < T; ++pvec)
  {
    std::vector<size_t> Ai;
    do
    {
      // get the perturbation set corresponding to the minimum score
      Ai = perturbationSets[ minHeap.top().second ];
      minHeap.pop(); // .top() returns, .pop() removes

      // modify Ai (shift)
      std::vector<size_t> As = Ai;
      perturbationShift(As);
      if ( perturbationValid(As, numProj) )
      {
        perturbationSets.push_back(As); // add shifted set to sets
        minHeap.push( 
            std::make_pair(
              perturbationScore(As, scores), 
              perturbationSets.size() - 1)
            );
      }

      // modify Ai (expand)
      std::vector<size_t> Ae = Ai;
      perturbationExpand(Ae);
      if ( perturbationValid(Ae, numProj) )
      {
        perturbationSets.push_back(Ae); // add expanded set to sets
        minHeap.push(
            std::make_pair(
              perturbationScore(Ae, scores),
              perturbationSets.size() - 1)
            );
      }

    }while (! perturbationValid(Ai, numProj)  );//Discard invalid perturbations

    // add perturbation vector to probing sequence if valid
    for (size_t i = 0; i < Ai.size(); ++i)
      additionalProbingBins(positions(Ai[i]), pvec) += actions(Ai[i]);

  }
}


template<typename SortPolicy>
template<typename VecType>
void LSHSearch<SortPolicy>::ReturnIndicesFromTable(
    const VecType& queryPoint,
    arma::uvec& referenceIndices,
    size_t numTablesToSearch,
    const size_t T) const
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
  arma::mat queryCodesNotFloored(numProj, numTablesToSearch);
  for (size_t i = 0; i < numTablesToSearch; i++)
    queryCodesNotFloored.unsafe_col(i) = projections.slice(i).t() * queryPoint;
  queryCodesNotFloored += offsets.cols(0, numTablesToSearch - 1);
  allProjInTables = arma::floor(queryCodesNotFloored/hashWidth);


  // Compute the hash value of each key of the query into a bucket of the
  // 'secondHashTable' using the 'secondHashWeights'.
  arma::Row<size_t> hashVec = 
    arma::conv_to< arma::Row<size_t> >::
    from( secondHashWeights.t() * allProjInTables ); // typecast to floor
  // mod to compute 2nd-level codes
  for (size_t i = 0; i < hashVec.n_elem; i++)
    hashVec[i] = (hashVec[i] % secondHashSize);

  Log::Assert(hashVec.n_elem == numTablesToSearch);

  // Compute hashVectors of additional probing bins
  arma::Mat<size_t> hashMat;
  if (T > 0)
  {
    hashMat.zeros(T, numTablesToSearch);

    for (size_t i = 0; i < numTablesToSearch; ++i)
    {
      // construct this table's probing sequence of length T
      arma::mat additionalProbingBins;
      GetAdditionalProbingBins(allProjInTables.unsafe_col(i),
                                queryCodesNotFloored.unsafe_col(i),
                                T,
                                additionalProbingBins);

      // map the probing bin to second hash table bins
      hashMat.col(i) = 
        arma::conv_to< arma::Col<size_t> >::
        from(additionalProbingBins.t() * secondHashWeights); // typecast floor
      for (size_t p = 0; p < T; ++p)
        hashMat(p, i) = (hashMat(p, i) % secondHashSize);
    }

    // top row of hashMat is primary bins for each table
    hashMat = arma::join_vert(hashVec, hashMat);
  }
  else
  {
    // if not multiprobe, hashMat is only hashVec's elements
    hashMat.set_size(1, numTablesToSearch);
    hashMat.row(0) = hashVec;
  }

  // Count number of points hashed in the same bucket as the query
  size_t maxNumPoints = 0;
  for (size_t i = 0; i < numTablesToSearch; ++i) //For all tables
  {
    for (size_t p = 0; p < T + 1; ++p)
    {
      size_t hashInd = (size_t) hashMat(p, i); //find query's bucket
      maxNumPoints += bucketContentSize[hashInd]; //count bucket contents
    }
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

    for (size_t i = 0; i < numTablesToSearch; ++i) // for all tables
    {
      for (size_t p = 0; p < T + 1; ++p) // for entire probing sequence
      {

        // get the sequence code
        size_t hashInd = (size_t) hashMat(p, i);

        if (bucketContentSize[hashInd] > 0)
        {
          // Pick the indices in the bucket corresponding to hashInd.
          size_t tableRow = bucketRowInHashTable[hashInd];
          assert(tableRow < secondHashSize);
          assert(tableRow < secondHashTable.n_rows);

          for (size_t j = 0; j < bucketContentSize[hashInd]; ++j)
            refPointsConsidered[secondHashTable(tableRow, j)]++;
        }
      }
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
      for (size_t p = 0; p < T + 1; ++p)
      {
        size_t hashInd =  hashMat(p, i); // Find the query's bucket.

        if (bucketContentSize[hashInd] > 0)
        {
          // tableRow hash indices corresponding to query.
          size_t tableRow = bucketRowInHashTable[hashInd];
          assert(tableRow < secondHashSize);
          assert(tableRow < secondHashTable.n_rows);

          // Store all secondHashTable points in the candidates set.
          for (size_t j = 0; j < bucketContentSize[hashInd]; ++j)
            refPointsConsideredSmall(start++) = secondHashTable(tableRow, j);
        }
      }
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
                                   const size_t numTablesToSearch,
                                   const size_t T)
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

  // If the user requested more than the available number of additional probing
  // bins, set Teffective to maximum T. Maximum T is 2^numProj - 1
  size_t Teffective = T;
  if (T > ( (size_t) ( (1 << numProj) - 1) ) )
  {
    Teffective = ( 1 << numProj ) - 1;
    Log::Warn << "Requested " << T << 
      " additional bins are more than theoretical maximum. Using " <<
      Teffective << " instead." << std::endl;
  }

  // If the user set multiprobe, log it
  if (T > 0)
    Log::Info << "Running multiprobe LSH with " << Teffective <<
      " additional probing bins per table per query."<< std::endl;


  size_t avgIndicesReturned = 0;

  Timer::Start("computing_neighbors");

  // Go through every query point sequentially.
  for (size_t i = 0; i < querySet.n_cols; i++)
  {
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(querySet.col(i), refIndices, numTablesToSearch, 
        Teffective);

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
       const size_t numTablesToSearch,
       size_t T)
{
  // This is monochromatic search; the query set is the reference set.
  resultingNeighbors.set_size(k, referenceSet->n_cols);
  distances.set_size(k, referenceSet->n_cols);
  distances.fill(SortPolicy::WorstDistance());
  resultingNeighbors.fill(referenceSet->n_cols);

  // If the user requested more than the available number of additional probing
  // bins, set Teffective to maximum T. Maximum T is 2^numProj - 1
  size_t Teffective = T;
  if (T > ( (size_t) ( (1 << numProj) - 1) ) )
  {
    Teffective = ( 1 << numProj ) - 1;
    Log::Warn << "Requested " << T << 
      " additional bins are more than theoretical maximum. Using " <<
      Teffective << " instead." << std::endl;
  }

  // If the user set multiprobe, log it
  if (T > 0)
    Log::Info << "Running multiprobe LSH with " << Teffective <<
      " additional probing bins per table per query."<< std::endl;
  
  size_t avgIndicesReturned = 0;

  Timer::Start("computing_neighbors");

  // Go through every query point sequentially.
  for (size_t i = 0; i < referenceSet->n_cols; i++)
  {
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(referenceSet->col(i), refIndices, numTablesToSearch,
        Teffective);

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
double LSHSearch<SortPolicy>::ComputeRecall(
    const arma::Mat<size_t>& foundNeighbors,
    const arma::Mat<size_t>& realNeighbors)
{
  if (foundNeighbors.n_rows != realNeighbors.n_rows ||
      foundNeighbors.n_cols != realNeighbors.n_cols)
    throw std::invalid_argument("LSHSearch::ComputeRecall(): matrices provided"
        " must have equal size");

  const size_t queries = foundNeighbors.n_cols;
  const size_t neighbors=  foundNeighbors.n_rows; //k

  // recall is set intersection of found and real neighbors
  double found = 0;
  for (size_t col = 0; col < queries; ++col)
    for (size_t row = 0; row < neighbors; ++row)
      for (size_t nei = 0; nei < realNeighbors.n_rows; ++nei)
        if (realNeighbors(row, col) == foundNeighbors(nei, col))
        {
          found++;
          break;
        }

  return found/realNeighbors.n_elem;

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

  // Backward compatibility: older version of LSHSearch stored the projection
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
  ar & CreateNVP(secondHashTable, "secondHashTable");
  ar & CreateNVP(bucketContentSize, "bucketContentSize");
  ar & CreateNVP(bucketRowInHashTable, "bucketRowInHashTable");
  ar & CreateNVP(distanceEvaluations, "distanceEvaluations");
}

} // namespace neighbor
} // namespace mlpack

#endif
