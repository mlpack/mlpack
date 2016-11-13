/**
 * @file lsh_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of the LSHSearch class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
    referenceSet(new arma::mat()), // Use an empty dataset.
    ownsSet(true),
    numProj(0),
    numTables(0),
    hashWidth(0),
    secondHashSize(99901),
    bucketSize(500),
    distanceEvaluations(0)
{
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
    const size_t numSamples = 25;
    // Compute a heuristic hash width from the data.
    for (size_t i = 0; i < numSamples; i++)
    {
      size_t p1 = (size_t) math::RandInt(referenceSet.n_cols);
      size_t p2 = (size_t) math::RandInt(referenceSet.n_cols);

      hashWidth += std::sqrt(metric::EuclideanDistance::Evaluate(
          referenceSet.unsafe_col(p1), referenceSet.unsafe_col(p2)));
    }

    hashWidth /= numSamples;
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
  // vector for table i will be held in row i.  We have to use int and not
  // size_t, otherwise negative numbers are cast to 0.
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
    // Now we hash every key, point ID to its corresponding bucket.  We must
    // also normalize the hashes to the range [0, secondHashSize).
    arma::rowvec unmodVector = secondHashWeights.t() * arma::floor(hashMat);
    for (size_t j = 0; j < unmodVector.n_elem; ++j)
    {
      double shs = (double) secondHashSize; // Convenience cast.
      if (unmodVector[j] >= 0.0)
      {
        const size_t key = size_t(fmod(unmodVector[j], shs));
        secondHashVectors(i, j) = key;
      }
      else
      {
        const double mod = fmod(-unmodVector[j], shs);
        const size_t key = (mod < 1.0) ? 0 : secondHashSize - size_t(mod);
        secondHashVectors(i, j) = key;
      }
    }
  }

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

// Base case where the query set is the reference set.  (So, we can't return
// ourselves as the nearest neighbor.)
template<typename SortPolicy>
inline force_inline
void LSHSearch<SortPolicy>::BaseCase(const size_t queryIndex,
                                     const arma::uvec& referenceIndices,
                                     const size_t k,
                                     arma::Mat<size_t>& neighbors,
                                     arma::mat& distances) const
{
  // Let's build the list of candidate neighbors for the given query point.
  // It will be initialized with k candidates:
  // (WorstDistance, referenceSet->n_cols)
  const Candidate def = std::make_pair(SortPolicy::WorstDistance(),
      referenceSet->n_cols);
  std::vector<Candidate> vect(k, def);
  CandidateList pqueue(CandidateCmp(), std::move(vect));

  for (size_t j = 0; j < referenceIndices.n_elem; ++j)
  {
    const size_t referenceIndex = referenceIndices[j];
    // If the points are the same, skip this point.
    if (queryIndex == referenceIndex)
      continue;

    const double distance = metric::EuclideanDistance::Evaluate(
        referenceSet->unsafe_col(queryIndex),
        referenceSet->unsafe_col(referenceIndex));

    Candidate c = std::make_pair(distance, referenceIndex);
    // If this distance is better than the worst candidate, let's insert it.
    if (CandidateCmp()(c, pqueue.top()))
    {
      pqueue.pop();
      pqueue.push(c);
    }
  }

  for (size_t j = 1; j <= k; j++)
  {
    neighbors(k - j, queryIndex) = pqueue.top().second;
    distances(k - j, queryIndex) = pqueue.top().first;
    pqueue.pop();
  }
}

// Base case for bichromatic search.
template<typename SortPolicy>
inline force_inline
void LSHSearch<SortPolicy>::BaseCase(const size_t queryIndex,
                                     const arma::uvec& referenceIndices,
                                     const size_t k,
                                     const arma::mat& querySet,
                                     arma::Mat<size_t>& neighbors,
                                     arma::mat& distances) const
{
  // Let's build the list of candidate neighbors for the given query point.
  // It will be initialized with k candidates:
  // (WorstDistance, referenceSet->n_cols)
  const Candidate def = std::make_pair(SortPolicy::WorstDistance(),
      referenceSet->n_cols);
  std::vector<Candidate> vect(k, def);
  CandidateList pqueue(CandidateCmp(), std::move(vect));

  for (size_t j = 0; j < referenceIndices.n_elem; ++j)
  {
    const size_t referenceIndex = referenceIndices[j];
    const double distance = metric::EuclideanDistance::Evaluate(
        querySet.unsafe_col(queryIndex),
        referenceSet->unsafe_col(referenceIndex));

    Candidate c = std::make_pair(distance, referenceIndex);
    // If this distance is better than the worst candidate, let's insert it.
    if (CandidateCmp()(c, pqueue.top()))
    {
      pqueue.pop();
      pqueue.push(c);
    }
  }

  for (size_t j = 1; j <= k; j++)
  {
    neighbors(k - j, queryIndex) = pqueue.top().second;
    distances(k - j, queryIndex) = pqueue.top().first;
    pqueue.pop();
  }
}

template<typename SortPolicy>
inline force_inline
double LSHSearch<SortPolicy>::PerturbationScore(
    const std::vector<bool>& A,
    const arma::vec& scores) const
{
  double score = 0.0;
  for (size_t i = 0; i < A.size(); ++i)
    if (A[i])
      score += scores(i); // add scores of non-zero indices
  return score;
}

template<typename SortPolicy>
inline force_inline
bool LSHSearch<SortPolicy>::PerturbationShift(std::vector<bool>& A) const
{
  size_t maxPos = 0;
  for (size_t i = 0; i < A.size(); ++i)
    if (A[i] == 1) // Marked true.
      maxPos = i;

  if (maxPos + 1 < A.size()) // Otherwise, this is an invalid vector.
  {
    A[maxPos] = 0;
    A[maxPos + 1] = 1;
    return true; // valid
  }
  return false; // invalid
}

template<typename SortPolicy>
inline force_inline
bool LSHSearch<SortPolicy>::PerturbationExpand(std::vector<bool>& A) const
{
  // Find the last '1' in A.
  size_t maxPos = 0;
  for (size_t i = 0; i < A.size(); ++i)
    if (A[i]) // Marked true.
      maxPos = i;

  if (maxPos + 1 < A.size()) // Otherwise, this is an invalid vector.
  {
    A[maxPos + 1] = 1;
    return true;
  }
  return false;
}

template<typename SortPolicy>
inline force_inline
bool LSHSearch<SortPolicy>::PerturbationValid(
    const std::vector<bool>& A) const
{
  // Use check to mark dimensions we have seen before in A. If a dimension is
  // seen twice (or more), A is not a valid perturbation.
  std::vector<bool> check(numProj);

  if (A.size() > 2 * numProj)
    return false; // This should never happen.

  // Check that we only see each dimension once. If not, vector is not valid.
  for (size_t i = 0; i < A.size(); ++i)
  {
    // Only check dimensions that were included.
    if (!A[i])
      continue;

    // If dimesnion is unseen thus far, mark it as seen.
    if (check[i % numProj] == false)
      check[i % numProj] = true;
    else
      return false; // If dimension was seen before, set is not valid.
  }
  // If we didn't fail, set is valid.
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


  // Calculate query point's projection position.
  arma::mat projection = queryCodeNotFloored;

  // Use projection to calculate query's distance from hash limits.
  arma::vec limLow = projection - queryCode * hashWidth;
  arma::vec limHigh = hashWidth - limLow;

  // Calculate scores. score = distance^2.
  arma::vec scores(2 * numProj);
  scores.rows(0, numProj - 1) = arma::pow(limLow, 2);
  scores.rows(numProj, (2 * numProj) - 1) = arma::pow(limHigh, 2);

  // Actions vector describes what perturbation (-1/+1) corresponds to a score.
  arma::Col<short int> actions(2 * numProj); // will be [-1 ... 1 ...]
  actions.rows(0, numProj - 1) = // First numProj rows.
    -1 * arma::ones< arma::Col<short int> > (numProj); // -1s
  actions.rows(numProj, (2 * numProj) - 1) = // Last numProj rows.
    arma::ones< arma::Col<short int> > (numProj); // 1s


  // Acting dimension vector shows which coordinate to transform according to
  // actions (actions are described by actions vector above).
  arma::Col<size_t> positions(2 * numProj); // Will be [0 1 2 ... 0 1 2 ...].
  positions.rows(0, numProj - 1) =
    arma::linspace< arma::Col<size_t> >(0, numProj - 1, numProj);
  positions.rows(numProj, 2 * numProj - 1) =
    arma::linspace< arma::Col<size_t> >(0, numProj - 1, numProj);

  // Special case: No need to create heap for 1 or 2 codes.
  if (T <= 2)
  {
    // First, find location of minimum score, generate 1 perturbation vector,
    // and add its code to additionalProbingBins column 0.

    // Find location and value of smallest element of scores vector.
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

    // Add or subtract 1 to dimension corresponding to minimum score.
    additionalProbingBins(positions[minloc], 0) += actions[minloc];
    if (T == 1)
      return; // Done if asked for only 1 code.

    // Now, find location of second smallest score and generate one more vector.
    // The second perturbation vector still can't comprise of more than one
    // change in the bin codes, because of the way perturbation vectors
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
      if (minscore2 > scores[s] && s != minloc) //second smallest
      {
        minscore2 = scores[s];
        minloc2 = s;
      }
    }

    // Add or subtract 1 to create second-lowest scoring vector.
    additionalProbingBins(positions[minloc2], 1) += actions[minloc2];
    return;
  }

  // General case: more than 2 perturbation vectors require use of minheap.

  // Sort everything in increasing order.
  arma::uvec sortidx = arma::sort_index(scores);
  scores = scores(sortidx);
  actions = actions(sortidx);
  positions = positions(sortidx);


  // Theory:
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

  // Perturbation sets (A) mark with 1 the (score, action, dimension) positions
  // included in a given perturbation vector. Other spaces are 0.
  std::vector<bool> Ao(2 * numProj);
  Ao[0] = 1; // Smallest vector includes only smallest score.

  std::vector< std::vector<bool> > perturbationSets;
  perturbationSets.push_back(Ao); // Storage of perturbation sets.

  std::priority_queue<
    std::pair<double, size_t>,        // contents: pairs of (score, index)
    std::vector<                      // container: vector of pairs
      std::pair<double, size_t>
      >,
    std::greater< std::pair<double, size_t> > // comparator of pairs
  > minHeap; // our minheap

  // Start by adding the lowest scoring set to the minheap.
  minHeap.push( std::make_pair(PerturbationScore(Ao, scores), 0) );

  // Loop invariable: after pvec iterations, additionalProbingBins contains pvec
  // valid codes of the lowest-scoring bins (bins most likely to contain
  // neighbors of the query).
  for (size_t pvec = 0; pvec < T; ++pvec)
  {
    std::vector<bool> Ai;
    do
    {
      // Get the perturbation set corresponding to the minimum score.
      Ai = perturbationSets[ minHeap.top().second ];
      minHeap.pop(); // .top() returns, .pop() removes

      // Shift operation on Ai (replace max with max+1).
      std::vector<bool> As = Ai;
      if (PerturbationShift(As) && PerturbationValid(As))
        // Don't add invalid sets.
      {
        perturbationSets.push_back(As); // add shifted set to sets
        minHeap.push(
            std::make_pair(PerturbationScore(As, scores),
            perturbationSets.size() - 1));
      }

      // Expand operation on Ai (add max+1 to set).
      std::vector<bool> Ae = Ai;
      if (PerturbationExpand(Ae) && PerturbationValid(Ae))
        // Don't add invalid sets.
      {
        perturbationSets.push_back(Ae); // add expanded set to sets
        minHeap.push(
            std::make_pair(PerturbationScore(Ae, scores),
            perturbationSets.size() - 1));
      }

    } while (!PerturbationValid(Ai));//Discard invalid perturbations

    // Found valid perturbation set Ai. Construct perturbation vector from set.
    for (size_t pos = 0; pos < Ai.size(); ++pos)
      // If Ai[pos] is marked, add action to probing vector.
      additionalProbingBins(positions(pos), pvec)
          += Ai[pos] ? actions(pos) : 0;
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
  allProjInTables = arma::floor(queryCodesNotFloored / hashWidth);

  // Use hashMat to store the primary probing codes and any additional codes
  // from multiprobe LSH.
  arma::Mat<size_t> hashMat;
  hashMat.set_size(T + 1, numTablesToSearch);

  // Compute the primary hash value of each key of the query into a bucket of
  // the secondHashTable using the secondHashWeights.
  hashMat.row(0) = arma::conv_to<arma::Row<size_t>> // Floor by typecasting
      ::from(secondHashWeights.t() * allProjInTables);
  // Mod to compute 2nd-level codes.
  for (size_t i = 0; i < numTablesToSearch; i++)
    hashMat(0, i) = (hashMat(0, i) % secondHashSize);

  // Compute hash codes of additional probing bins.
  if (T > 0)
  {
    for (size_t i = 0; i < numTablesToSearch; ++i)
    {
      // Construct this table's probing sequence of length T.
      arma::mat additionalProbingBins;
      GetAdditionalProbingBins(allProjInTables.unsafe_col(i),
                                queryCodesNotFloored.unsafe_col(i),
                                T,
                                additionalProbingBins);

      // Map each probing bin to a bin in secondHashTable (just like we did for
      // the primary hash table).
      hashMat(arma::span(1, T), i) = // Compute code of rows 1:end of column i
        arma::conv_to< arma::Col<size_t> >:: // floor by typecasting to size_t
        from( secondHashWeights.t() * additionalProbingBins );
      for (size_t p = 1; p < T + 1; ++p)
        hashMat(p, i) = (hashMat(p, i) % secondHashSize);
    }
  }


  // Count number of points hashed in the same bucket as the query.
  size_t maxNumPoints = 0;
  for (size_t i = 0; i < numTablesToSearch; ++i)
  {
    for (size_t p = 0; p < T + 1; ++p)
    {
      const size_t hashInd = hashMat(p, i); // find query's bucket
      const size_t tableRow = bucketRowInHashTable[hashInd];
      if (tableRow < secondHashSize)
        maxNumPoints += bucketContentSize[tableRow]; // count bucket contents
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
      for (size_t p = 0; p < T + 1; ++p) // For entire probing sequence.
      {
        // get the sequence code
        size_t hashInd = hashMat(p, i);
        size_t tableRow = bucketRowInHashTable[hashInd];

        if (tableRow < secondHashSize && bucketContentSize[tableRow] > 0)
          // Pick the indices in the bucket corresponding to hashInd.
          for (size_t j = 0; j < bucketContentSize[tableRow]; ++j)
            refPointsConsidered[ secondHashTable[tableRow](j) ]++;
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
        const size_t hashInd =  hashMat(p, i); // Find the query's bucket.
        const size_t tableRow = bucketRowInHashTable[hashInd];

        if (tableRow < secondHashSize)
         // Store all secondHashTable points in the candidates set.
         for (size_t j = 0; j < bucketContentSize[tableRow]; ++j)
           refPointsConsideredSmall(start++) = secondHashTable[tableRow](j);
      }
    }

    // Keep only one copy of each candidate.
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

  // If the user asked for 0 nearest neighbors... uh... we're done.
  if (k == 0)
    return;

  // If the user requested more than the available number of additional probing
  // bins, set Teffective to maximum T. Maximum T is 2^numProj - 1
  size_t Teffective = T;
  if (T > ((size_t) ((1 << numProj) - 1)))
  {
    Teffective = (1 << numProj) - 1;
    Log::Warn << "Requested " << T << " additional bins are more than "
        << "theoretical maximum. Using " << Teffective << " instead."
        << std::endl;
  }

  // If the user set multiprobe, log it
  if (Teffective > 0)
    Log::Info << "Running multiprobe LSH with " << Teffective
        <<" additional probing bins per table per query." << std::endl;

  size_t avgIndicesReturned = 0;

  Timer::Start("computing_neighbors");

  // Parallelization to process more than one query at a time.
#ifdef _WIN32
  // Tiny workaround: Visual Studio only implements OpenMP 2.0, which doesn't
  // support unsigned loop variables. If we're building for Visual Studio, use
  // the intmax_t type instead.
  #pragma omp parallel for \
      shared(resultingNeighbors, distances) \
      schedule(dynamic)\
      reduction(+:avgIndicesReturned)
  for (intmax_t i = 0; i < (intmax_t) querySet.n_cols; ++i)
#else
  #pragma omp parallel for \
      shared(resultingNeighbors, distances) \
      schedule(dynamic)\
      reduction(+:avgIndicesReturned)
  for (size_t i = 0; i < querySet.n_cols; ++i)
#endif
  {
    // Go through every query point.
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(querySet.col(i), refIndices, numTablesToSearch,
        Teffective);

    // An informative book-keeping for the number of neighbor candidates
    // returned on average.
    // Make atomic to avoid race conditions when multiple threads are running
    // #pragma omp atomic
    avgIndicesReturned = avgIndicesReturned + refIndices.n_elem;

    // Sequentially go through all the candidates and save the best 'k'
    // candidates.
    BaseCase(i, refIndices, k, querySet, resultingNeighbors, distances);
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

  // If the user requested more than the available number of additional probing
  // bins, set Teffective to maximum T. Maximum T is 2^numProj - 1
  size_t Teffective = T;
  if (T > ((size_t) ((1 << numProj) - 1)))
  {
    Teffective = (1 << numProj) - 1;
    Log::Warn << "Requested " << T << " additional bins are more than "
        << "theoretical maximum. Using " << Teffective << " instead."
        << std::endl;
  }

  // If the user set multiprobe, log it
  if (T > 0)
    Log::Info << "Running multiprobe LSH with " << Teffective <<
      " additional probing bins per table per query."<< std::endl;

  size_t avgIndicesReturned = 0;

  Timer::Start("computing_neighbors");

  // Parallelization to process more than one query at a time.
#ifdef _WIN32
  // Tiny workaround: Visual Studio only implements OpenMP 2.0, which doesn't
  // support unsigned loop variables. If we're building for Visual Studio, use
  // the intmax_t type instead.
  #pragma omp parallel for \
      shared(resultingNeighbors, distances) \
      schedule(dynamic)\
      reduction(+:avgIndicesReturned)
  for (intmax_t i = 0; i < (intmax_t) referenceSet->n_cols; ++i)
#else
  #pragma omp parallel for \
      shared(resultingNeighbors, distances) \
      schedule(dynamic)\
      reduction(+:avgIndicesReturned)
  for (size_t i = 0; i < referenceSet->n_cols; ++i)
#endif
  {
    // Go through every query point.
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(referenceSet->col(i), refIndices, numTablesToSearch,
        Teffective);

    // An informative book-keeping for the number of neighbor candidates
    // returned on average.
    // Make atomic to avoid race conditions when multiple threads are running.
    // #pragma omp atomic
    avgIndicesReturned += refIndices.n_elem;

    // Sequentially go through all the candidates and save the best 'k'
    // candidates.
    BaseCase(i, refIndices, k, resultingNeighbors, distances);
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
  const size_t neighbors = foundNeighbors.n_rows; // Should be equal to k.

  // The recall is the set intersection of found and real neighbors.
  size_t found = 0;
  for (size_t col = 0; col < queries; ++col)
    for (size_t row = 0; row < neighbors; ++row)
      for (size_t nei = 0; nei < realNeighbors.n_rows; ++nei)
        if (realNeighbors(row, col) == foundNeighbors(nei, col))
        {
          found++;
          break;
        }

  return ((double) found) / realNeighbors.n_elem;
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
