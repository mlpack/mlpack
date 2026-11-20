/**
 * @file methods/lsh/lsh_search_impl.hpp
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

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {

// Construct the object with random tables
template<typename SortPolicy, typename MatType>
LSHSearch<SortPolicy, MatType>::
LSHSearch(MatType referenceSet,
          const size_t numProj,
          const size_t numTables,
          const double hashWidthIn,
          const size_t secondHashSize,
          const size_t bucketSize) :
  numProj(numProj),
  numTables(numTables),
  hashWidth(hashWidthIn),
  secondHashSize(secondHashSize),
  bucketSize(bucketSize),
  distanceEvaluations(0)
{
  // Pass work to training function.
  Train(std::move(referenceSet), numProj, numTables, hashWidthIn,
      secondHashSize, bucketSize);
}

// Construct the object with given tables
template<typename SortPolicy, typename MatType>
LSHSearch<SortPolicy, MatType>::
LSHSearch(MatType referenceSet,
          const arma::cube& projections,
          const double hashWidthIn,
          const size_t secondHashSize,
          const size_t bucketSize) :
  numProj(projections.n_cols),
  numTables(projections.n_slices),
  hashWidth(hashWidthIn),
  secondHashSize(secondHashSize),
  bucketSize(bucketSize),
  distanceEvaluations(0)
{
  // Pass work to training function.
  Train(std::move(referenceSet), numProj, numTables, hashWidthIn,
      secondHashSize, bucketSize, projections);
}

// Empty constructor.
template<typename SortPolicy, typename MatType>
LSHSearch<SortPolicy, MatType>::LSHSearch() :
    numProj(0),
    numTables(0),
    hashWidth(0),
    secondHashSize(99901),
    bucketSize(500),
    distanceEvaluations(0)
{
}

// Copy constructor.
template<typename SortPolicy, typename MatType>
LSHSearch<SortPolicy, MatType>::LSHSearch(const LSHSearch& other) :
    referenceSet(other.referenceSet), // Copy the other set.
    numProj(other.numProj),
    numTables(other.numTables),
    projections(other.projections),
    offsets(other.offsets),
    hashWidth(other.hashWidth),
    secondHashSize(other.secondHashSize),
    secondHashWeights(other.secondHashWeights),
    bucketSize(other.bucketSize),
    secondHashTable(other.secondHashTable),
    bucketContentSize(other.bucketContentSize),
    bucketRowInHashTable(other.bucketRowInHashTable),
    distanceEvaluations(other.distanceEvaluations)
{
  // Nothing to do.
}

// Move constructor.
template<typename SortPolicy, typename MatType>
LSHSearch<SortPolicy, MatType>::LSHSearch(LSHSearch&& other) :
    referenceSet(std::move(other.referenceSet)),
    numProj(other.numProj),
    numTables(other.numTables),
    projections(std::move(other.projections)),
    offsets(std::move(other.offsets)),
    hashWidth(other.hashWidth),
    secondHashSize(other.secondHashSize),
    secondHashWeights(std::move(other.secondHashWeights)),
    bucketSize(other.bucketSize),
    secondHashTable(std::move(other.secondHashTable)),
    bucketContentSize(std::move(other.bucketContentSize)),
    bucketRowInHashTable(std::move(other.bucketRowInHashTable)),
    distanceEvaluations(other.distanceEvaluations)
{
  // Reset other model to defaults.
  other.numProj = 0;
  other.numTables = 0;
  other.hashWidth = 0;
  other.secondHashSize = 99901;
  other.bucketSize = 500;
  other.distanceEvaluations = 0;
}

// Copy operator.
template<typename SortPolicy, typename MatType>
LSHSearch<SortPolicy, MatType>& LSHSearch<SortPolicy, MatType>::operator=(
    const LSHSearch& other)
{
  referenceSet = other.referenceSet;
  numProj = other.numProj;
  numTables = other.numTables;
  projections = other.projections;
  offsets = other.offsets;
  hashWidth = other.hashWidth;
  secondHashSize = other.secondHashSize;
  secondHashWeights = other.secondHashWeights;
  bucketSize = other.bucketSize;
  secondHashTable = other.secondHashTable;
  bucketContentSize = other.bucketContentSize;
  bucketRowInHashTable = other.bucketRowInHashTable;
  distanceEvaluations = other.distanceEvaluations;

  return *this;
}

// Move operator.
template<typename SortPolicy, typename MatType>
LSHSearch<SortPolicy, MatType>& LSHSearch<SortPolicy, MatType>::operator=(
    LSHSearch&& other)
{
  referenceSet = std::move(other.referenceSet);
  numProj = other.numProj;
  numTables = other.numTables;
  projections = std::move(other.projections);
  offsets = std::move(other.offsets);
  hashWidth = other.hashWidth;
  secondHashSize = other.secondHashSize;
  secondHashWeights = std::move(other.secondHashWeights);
  bucketSize = other.bucketSize;
  secondHashTable = std::move(other.secondHashTable);
  bucketContentSize = std::move(other.bucketContentSize);
  bucketRowInHashTable = std::move(other.bucketRowInHashTable);
  distanceEvaluations = other.distanceEvaluations;

  // Reset other model to defaults.
  other.numProj = 0;
  other.numTables = 0;
  other.hashWidth = 0;
  other.secondHashSize = 99901;
  other.bucketSize = 500;
  other.distanceEvaluations = 0;

  return *this;
}

// Train on a new reference set.
template<typename SortPolicy, typename MatType>
void LSHSearch<SortPolicy, MatType>::Train(MatType referenceSet,
                                           const size_t numProj,
                                           const size_t numTables,
                                           const double hashWidthIn,
                                           const size_t secondHashSize,
                                           const size_t bucketSize,
                                           const arma::cube& projection)
{
  // Set new reference set.
  this->referenceSet = std::move(referenceSet);

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
    for (size_t i = 0; i < numSamples; ++i)
    {
      size_t p1 = (size_t) RandInt(this->referenceSet.n_cols);
      size_t p2 = (size_t) RandInt(this->referenceSet.n_cols);

      hashWidth += std::sqrt(EuclideanDistance::Evaluate(
          this->referenceSet.col(p1),
          this->referenceSet.col(p2)));
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
    projections.randn(this->referenceSet.n_rows, numProj, numTables);
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
  arma::Mat<size_t> secondHashVectors(numTables, this->referenceSet.n_cols);

  for (size_t i = 0; i < numTables; ++i)
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
    // key = { floor((<proj_i, point> + offset_i) / 'hashWidth') forall i }
    arma::mat offsetMat = repmat(offsets.unsafe_col(i), 1,
                                 this->referenceSet.n_cols);
    arma::mat hashMat = projections.slice(i).t() * (this->referenceSet);
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
  arma::Row<size_t> secondHashBinCounts(secondHashSize);
  for (size_t i = 0; i < secondHashVectors.n_elem; ++i)
    secondHashBinCounts[secondHashVectors[i]]++;

  // Enforce the maximum bucket size.
  const size_t effectiveBucketSize = (bucketSize == 0) ? SIZE_MAX : bucketSize;
  secondHashBinCounts.transform([effectiveBucketSize](size_t val)
      { return std::min(val, effectiveBucketSize); });

  const size_t numRowsInTable = accu(secondHashBinCounts > 0);
  bucketContentSize.zeros(numRowsInTable);
  secondHashTable.resize(numRowsInTable);

  // Next we must assign each point in each table to the right second hash
  // table.
  size_t currentRow = 0;
  for (size_t i = 0; i < numTables; ++i)
  {
    // Insert the point in the corresponding row to its bucket in the
    // 'secondHashTable'.
    for (size_t j = 0; j < secondHashVectors.n_cols; ++j)
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
            << "maximum length of " << max(secondHashBinCounts) << ", "
            << "totaling " << accu(secondHashBinCounts) << " elements."
            << std::endl;
}

// Base case where the query set is the reference set.  (So, we can't return
// ourselves as the nearest neighbor.)
template<typename SortPolicy, typename MatType>
inline mlpack_force_inline
void LSHSearch<SortPolicy, MatType>::BaseCase(
    const size_t queryIndex,
    const arma::uvec& referenceIndices,
    const size_t k,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances) const
{
  // Let's build the list of candidate neighbors for the given query point.
  // It will be initialized with k candidates:
  // (WorstDistance, referenceSet.n_cols)
  const Candidate def = std::make_pair(SortPolicy::WorstDistance(),
      referenceSet.n_cols);
  std::vector<Candidate> vect(k, def);
  CandidateList pqueue(CandidateCmp(), std::move(vect));

  for (size_t j = 0; j < referenceIndices.n_elem; ++j)
  {
    const size_t referenceIndex = referenceIndices[j];
    // If the points are the same, skip this point.
    if (queryIndex == referenceIndex)
      continue;

    const double distance = EuclideanDistance::Evaluate(
        referenceSet.col(queryIndex),
        referenceSet.col(referenceIndex));

    Candidate c = std::make_pair(distance, referenceIndex);
    // If this distance is better than the worst candidate, let's insert it.
    if (CandidateCmp()(c, pqueue.top()))
    {
      pqueue.pop();
      pqueue.push(c);
    }
  }

  for (size_t j = 1; j <= k; ++j)
  {
    neighbors(k - j, queryIndex) = pqueue.top().second;
    distances(k - j, queryIndex) = pqueue.top().first;
    pqueue.pop();
  }
}

// Base case for bichromatic search.
template<typename SortPolicy, typename MatType>
inline mlpack_force_inline
void LSHSearch<SortPolicy, MatType>::BaseCase(
    const size_t queryIndex,
    const arma::uvec& referenceIndices,
    const size_t k,
    const MatType& querySet,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances) const
{
  // Let's build the list of candidate neighbors for the given query point.
  // It will be initialized with k candidates:
  // (WorstDistance, referenceSet.n_cols)
  const Candidate def = std::make_pair(SortPolicy::WorstDistance(),
      referenceSet.n_cols);
  std::vector<Candidate> vect(k, def);
  CandidateList pqueue(CandidateCmp(), std::move(vect));

  for (size_t j = 0; j < referenceIndices.n_elem; ++j)
  {
    const size_t referenceIndex = referenceIndices[j];
    const double distance = EuclideanDistance::Evaluate(
        querySet.col(queryIndex),
        referenceSet.col(referenceIndex));

    Candidate c = std::make_pair(distance, referenceIndex);
    // If this distance is better than the worst candidate, let's insert it.
    if (CandidateCmp()(c, pqueue.top()))
    {
      pqueue.pop();
      pqueue.push(c);
    }
  }

  for (size_t j = 1; j <= k; ++j)
  {
    neighbors(k - j, queryIndex) = pqueue.top().second;
    distances(k - j, queryIndex) = pqueue.top().first;
    pqueue.pop();
  }
}

template<typename SortPolicy, typename MatType>
inline mlpack_force_inline
double LSHSearch<SortPolicy, MatType>::PerturbationScore(
    const std::vector<bool>& A,
    const arma::vec& scores) const
{
  double score = 0.0;
  for (size_t i = 0; i < A.size(); ++i)
    if (A[i])
      score += scores(i); // add scores of non-zero indices
  return score;
}

template<typename SortPolicy, typename MatType>
inline mlpack_force_inline
bool LSHSearch<SortPolicy, MatType>::PerturbationShift(
    std::vector<bool>& A) const
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

template<typename SortPolicy, typename MatType>
inline mlpack_force_inline
bool LSHSearch<SortPolicy, MatType>::PerturbationExpand(
    std::vector<bool>& A) const
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

template<typename SortPolicy, typename MatType>
inline mlpack_force_inline
bool LSHSearch<SortPolicy, MatType>::PerturbationValid(
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
template<typename SortPolicy, typename MatType>
void LSHSearch<SortPolicy, MatType>::GetAdditionalProbingBins(
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
  scores.rows(0, numProj - 1) = pow(limLow, 2);
  scores.rows(numProj, (2 * numProj) - 1) = pow(limHigh, 2);

  // Actions vector describes what perturbation (-1/+1) corresponds to a score.
  arma::Col<short int> actions(2 * numProj); // will be [-1 ... 1 ...]
  actions.rows(0, numProj - 1) = // First numProj rows.
    -1 * ones<arma::Col<short int>> (numProj); // -1s
  actions.rows(numProj, (2 * numProj) - 1) = // Last numProj rows.
    ones<arma::Col<short int>> (numProj); // 1s


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
    for (size_t s = 0; s < (2 * numProj); ++s) // Here we can't start from 1.
    {
      if (minscore2 > scores[s] && s != minloc) // Second smallest.
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
      std::pair<double, size_t>>,
    std::greater< std::pair<double, size_t> > // comparator of pairs
  > minHeap; // our minheap

  // Start by adding the lowest scoring set to the minheap.
  minHeap.push(std::make_pair(PerturbationScore(Ao, scores), 0));

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

      // Don't add invalid sets.
      if (PerturbationShift(As) && PerturbationValid(As))
      {
        perturbationSets.push_back(As); // add shifted set to sets
        minHeap.push(
            std::make_pair(PerturbationScore(As, scores),
            perturbationSets.size() - 1));
      }

      // Expand operation on Ai (add max+1 to set).
      std::vector<bool> Ae = Ai;

      // Don't add invalid sets.
      if (PerturbationExpand(Ae) && PerturbationValid(Ae))
      {
        perturbationSets.push_back(Ae); // add expanded set to sets
        minHeap.push(
            std::make_pair(PerturbationScore(Ae, scores),
            perturbationSets.size() - 1));
      }
    } while (!PerturbationValid(Ai)); // Discard invalid perturbations

    // Found valid perturbation set Ai. Construct perturbation vector from set.
    for (size_t pos = 0; pos < Ai.size(); ++pos)
    {
      // If Ai[pos] is marked, add action to probing vector.
      additionalProbingBins(positions(pos), pvec) += Ai[pos] ? actions(pos) : 0;
    }
  }
}

template<typename SortPolicy, typename MatType>
template<typename VecType>
void LSHSearch<SortPolicy, MatType>::ReturnIndicesFromTable(
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
  for (size_t i = 0; i < numTablesToSearch; ++i)
    queryCodesNotFloored.unsafe_col(i) = projections.slice(i).t() * queryPoint;

  queryCodesNotFloored += offsets.cols(0, numTablesToSearch - 1);
  allProjInTables = arma::floor(queryCodesNotFloored / hashWidth);

  // Use hashMat to store the primary probing codes and any additional codes
  // from multiprobe LSH.
  arma::Mat<size_t> hashMat;
  hashMat.set_size(T + 1, numTablesToSearch);

  // Compute the primary hash value of each key of the query into a bucket of
  // the secondHashTable using the secondHashWeights.
  hashMat.row(0) = ConvTo<arma::Row<size_t>> // Floor by typecasting
      ::From(secondHashWeights.t() * allProjInTables);
  // Mod to compute 2nd-level codes.
  for (size_t i = 0; i < numTablesToSearch; ++i)
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
        ConvTo<arma::Col<size_t>>:: // floor by typecasting to size_t
        From(secondHashWeights.t() * additionalProbingBins);
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
  // Or allocate a referenceSet.n_cols size vector (i.e. number of reference
  // points) of zeros, and mark found indices as 1.
  // Option 1 runs faster for small maxNumPoints but worse for larger values, so
  // we choose based on a heuristic.
  const float cutoff = 0.1;
  const float selectivity = static_cast<float>(maxNumPoints) /
      static_cast<float>(referenceSet.n_cols);

  if (selectivity > cutoff)
  {
    // Heuristic: larger maxNumPoints means we should use find() because it
    // should be faster.
    // Reference points hashed in the same bucket as the query are set to >0.
    arma::Col<size_t> refPointsConsidered;
    refPointsConsidered.zeros(referenceSet.n_cols);

    for (size_t i = 0; i < numTablesToSearch; ++i) // for all tables
    {
      for (size_t p = 0; p < T + 1; ++p) // For entire probing sequence.
      {
        // get the sequence code
        size_t hashInd = hashMat(p, i);
        size_t tableRow = bucketRowInHashTable[hashInd];

        if (tableRow < secondHashSize && bucketContentSize[tableRow] > 0)
        {
          // Pick the indices in the bucket corresponding to hashInd.
          for (size_t j = 0; j < bucketContentSize[tableRow]; ++j)
            refPointsConsidered[ secondHashTable[tableRow](j) ]++;
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
        const size_t hashInd = hashMat(p, i); // Find the query's bucket.
        const size_t tableRow = bucketRowInHashTable[hashInd];

        if (tableRow < secondHashSize)
        {
          // Store all secondHashTable points in the candidates set.
          for (size_t j = 0; j < bucketContentSize[tableRow]; ++j)
            refPointsConsideredSmall(start++) = secondHashTable[tableRow](j);
       }
      }
    }

    // Keep only one copy of each candidate.
    referenceIndices = arma::unique(refPointsConsideredSmall);
    return;
  }
}

// Search for nearest neighbors in a given query set.
template<typename SortPolicy, typename MatType>
void LSHSearch<SortPolicy, MatType>::Search(
    const MatType& querySet,
    const size_t k,
    arma::Mat<size_t>& resultingNeighbors,
    arma::mat& distances,
    const size_t numTablesToSearch,
    const size_t T)
{
  // Ensure the dimensionality of the query set is correct.
  util::CheckSameDimensionality(querySet, referenceSet, "LSHSearch::Search()",
      "query set");

  if (k > referenceSet.n_cols)
  {
    std::ostringstream oss;
    oss << "LSHSearch::Search(): requested " << k << " approximate nearest "
        << "neighbors, but reference set has " << referenceSet.n_cols
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

  // Parallelization to process more than one query at a time.
  #pragma omp parallel for \
      shared(resultingNeighbors, distances) \
      schedule(dynamic)\
      reduction(+:avgIndicesReturned)
  for (size_t i = 0; i < (size_t) querySet.n_cols; ++i)
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

  distanceEvaluations += avgIndicesReturned;
  avgIndicesReturned /= querySet.n_cols;
  Log::Info << avgIndicesReturned << " distinct indices returned on average." <<
      std::endl;
}

// Search for approximate neighbors of the reference set.
template<typename SortPolicy, typename MatType>
void LSHSearch<SortPolicy, MatType>::
Search(const size_t k,
       arma::Mat<size_t>& resultingNeighbors,
       arma::mat& distances,
       const size_t numTablesToSearch,
       size_t T)
{
  // This is monochromatic search; the query set is the reference set.
  resultingNeighbors.set_size(k, referenceSet.n_cols);
  distances.set_size(k, referenceSet.n_cols);

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

  // Parallelization to process more than one query at a time.
  #pragma omp parallel for \
      shared(resultingNeighbors, distances) \
      schedule(dynamic)\
      reduction(+:avgIndicesReturned)
  for (size_t i = 0; i < (size_t) referenceSet.n_cols; ++i)
  {
    // Go through every query point.
    // Hash every query into every hash table and eventually into the
    // 'secondHashTable' to obtain the neighbor candidates.
    arma::uvec refIndices;
    ReturnIndicesFromTable(referenceSet.col(i), refIndices, numTablesToSearch,
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

  distanceEvaluations += avgIndicesReturned;
  avgIndicesReturned /= referenceSet.n_cols;
  Log::Info << avgIndicesReturned << " distinct indices returned on average." <<
      std::endl;
}

template<typename SortPolicy, typename MatType>
double LSHSearch<SortPolicy, MatType>::ComputeRecall(
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

template<typename SortPolicy, typename MatType>
template<typename Archive>
void LSHSearch<SortPolicy, MatType>::serialize(Archive& ar,
                                               const uint32_t /* version */)
{
  ar(CEREAL_NVP(referenceSet));
  ar(CEREAL_NVP(numProj));
  ar(CEREAL_NVP(numTables));

  // Delete existing projections, if necessary.
  if (cereal::is_loading<Archive>())
    projections.reset();

  ar(CEREAL_NVP(projections));
  ar(CEREAL_NVP(offsets));
  ar(CEREAL_NVP(hashWidth));
  ar(CEREAL_NVP(secondHashSize));
  ar(CEREAL_NVP(secondHashWeights));
  ar(CEREAL_NVP(bucketSize));
  ar(CEREAL_NVP(secondHashTable));
  ar(CEREAL_NVP(bucketContentSize));
  ar(CEREAL_NVP(bucketRowInHashTable));
  ar(CEREAL_NVP(distanceEvaluations));
}

} // namespace mlpack

#endif
