/**
 * @file methods/lsh/lsh_search.hpp
 * @author Parikshit Ram
 *
 * Defines the LSHSearch class, which performs an approximate
 * nearest neighbor search for a queries in a query set
 * over a given dataset using Locality-sensitive hashing
 * with 2-stable distributions.
 *
 * The details of this method can be found in the following paper:
 *
 * @code
 * @inproceedings{datar2004locality,
 *  title={Locality-sensitive hashing scheme based on p-stable distributions},
 *  author={Datar, M. and Immorlica, N. and Indyk, P. and Mirrokni, V.S.},
 *  booktitle=
 *      {Proceedings of the 12th Annual Symposium on Computational Geometry},
 *  pages={253--262},
 *  year={2004},
 *  organization={ACM}
 * }
 * @endcode
 *
 * Additionally, the class implements Multiprobe LSH, which improves
 * approximation results during the search for approximate nearest neighbors.
 * The Multiprobe LSH algorithm was presented in the paper:
 *
 * @code
 * @inproceedings{Lv2007multiprobe,
 *  tile={Multi-probe LSH: efficient indexing for high-dimensional similarity
 *  search},
 *  author={Lv, Qin and Josephson, William and Wang, Zhe and Charikar, Moses and
 *  Li, Kai},
 *  booktitle={Proceedings of the 33rd international conference on Very large
 *  data bases},
 *  year={2007},
 *  pages={950--961}
 * }
 * @endcode
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>

namespace mlpack {

/**
 * The LSHSearch class; this class builds a hash on the reference set and uses
 * this hash to compute the distance-approximate nearest-neighbors of the given
 * queries.
 *
 * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
 * @tparam MatType Type of matrix to use to store the data.
 */
template<
    typename SortPolicy = NearestNeighborSort,
    typename MatType = arma::mat
>
class LSHSearch
{
 public:
  /**
   * This function initializes the LSH class. It builds the hash on the
   * reference set with 2-stable distributions. See the individual functions
   * performing the hashing for details on how the hashing is done.  In order to
   * avoid copying the reference set, it is suggested to pass that parameter
   * with std::move().
   *
   * @param referenceSet Set of reference points and the set of queries.
   * @param projections Cube of projection tables. For a cube of size (a, b, c)
   *     we set numProj = a, numTables = c. b is the reference set
   *     dimensionality.
   * @param hashWidth The width of hash for every table. If 0 (the default) is
   *     provided, then the hash width is automatically obtained by computing
   *     the average pairwise distance of 25 pairs.  This should be a reasonable
   *     upper bound on the nearest-neighbor distance in general.
   * @param secondHashSize The size of the second hash table. This should be a
   *     large prime number.
   * @param bucketSize The size of the bucket in the second hash table. This is
   *     the maximum number of points that can be hashed into single bucket.  A
   *     value of 0 indicates that there is no limit (so the second hash table
   *     can be arbitrarily large---be careful!).
   */
  LSHSearch(MatType referenceSet,
            const arma::cube& projections,
            const double hashWidth = 0.0,
            const size_t secondHashSize = 99901,
            const size_t bucketSize = 500);

  /**
   * This function initializes the LSH class. It builds the hash one the
   * reference set using the provided projections. See the individual functions
   * performing the hashing for details on how the hashing is done.  In order to
   * avoid copying the reference set, consider passing the set with std::move().
   *
   * @param referenceSet Set of reference points and the set of queries.
   * @param numProj Number of projections in each hash table (anything between
   *     10-50 might be a decent choice).
   * @param numTables Total number of hash tables (anything between 10-20
   *     should suffice).
   * @param hashWidth The width of hash for every table. If 0 (the default) is
   *     provided, then the hash width is automatically obtained by computing
   *     the average pairwise distance of 25 pairs.  This should be a reasonable
   *     upper bound on the nearest-neighbor distance in general.
   * @param secondHashSize The size of the second hash table. This should be a
   *     large prime number.
   * @param bucketSize The size of the bucket in the second hash table. This is
   *     the maximum number of points that can be hashed into single bucket.  A
   *     value of 0 indicates that there is no limit (so the second hash table
   *     can be arbitrarily large---be careful!).
   */
  LSHSearch(MatType referenceSet,
            const size_t numProj,
            const size_t numTables,
            const double hashWidth = 0.0,
            const size_t secondHashSize = 99901,
            const size_t bucketSize = 500);

  /**
   * Create an untrained LSH model.  Be sure to call Train() before calling
   * Search(); otherwise, an exception will be thrown when Search() is called.
   */
  LSHSearch();

  /**
   * Copy the given LSH model.
   *
   * @param other Other LSH model to copy.
   */
  LSHSearch(const LSHSearch& other);

  /**
   * Take ownership of the given LSH model.
   *
   * @param other Other LSH model to take ownership of.
   */
  LSHSearch(LSHSearch&& other);

  /**
   * Copy the given LSH model.
   *
   * @param other Other LSH model to copy.
   */
  LSHSearch& operator=(const LSHSearch& other);

  /**
   * Take ownership of the given LSH model.
   *
   * @param other Other LSH model to take ownership of.
   */
  LSHSearch& operator=(LSHSearch&& other);

  /**
   * Train the LSH model on the given dataset.  If a correctly-sized projection
   * cube is not provided, this means building new hash tables. Otherwise, we
   * use the projections provided by the user.  In order to avoid copying the
   * reference set, consider passing that parameter with std::move().
   *
   * @param referenceSet Set of reference points and the set of queries.
   * @param numProj Number of projections in each hash table (anything between
   *     10-50 might be a decent choice).
   * @param numTables Total number of hash tables (anything between 10-20
   *     should suffice).
   * @param hashWidth The width of hash for every table. If 0 (the default) is
   *     provided, then the hash width is automatically obtained by computing
   *     the average pairwise distance of 25 pairs.  This should be a reasonable
   *     upper bound on the nearest-neighbor distance in general.
   * @param secondHashSize The size of the second hash table. This should be a
   *     large prime number.
   * @param bucketSize The size of the bucket in the second hash table. This is
   *     the maximum number of points that can be hashed into single bucket.  A
   *     value of 0 indicates that there is no limit (so the second hash table
   *     can be arbitrarily large---be careful!).
   * @param projection Cube of projection tables. For a cube of size (a, b, c)
   *     we set numProj = a, numTables = c. b is the reference set
   *     dimensionality.
   */
  void Train(MatType referenceSet,
             const size_t numProj,
             const size_t numTables,
             const double hashWidth = 0.0,
             const size_t secondHashSize = 99901,
             const size_t bucketSize = 500,
             const arma::cube& projection = arma::cube());

  /**
   * Compute the nearest neighbors of the points in the given query set and
   * store the output in the given matrices.  The matrices will be set to the
   * size of n columns by k rows, where n is the number of points in the query
   * dataset and k is the number of neighbors being searched for.
   *
   * @param querySet Set of query points.
   * @param k Number of neighbors to search for.
   * @param resultingNeighbors Matrix storing lists of neighbors for each query
   *     point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   * @param numTablesToSearch This parameter allows the user to have control
   *     over the number of hash tables to be searched. This allows
   *     the user to pick the number of tables it can afford for the time
   *     available without having to build hashing for every table size.
   *     By default, this is set to zero in which case all tables are
   *     considered.
   * @param T The number of additional probing bins to examine with multiprobe
   *     LSH. If T = 0, classic single-probe LSH is run (default).
   */
  void Search(const MatType& querySet,
              const size_t k,
              arma::Mat<size_t>& resultingNeighbors,
              arma::mat& distances,
              const size_t numTablesToSearch = 0,
              const size_t T = 0);

  /**
   * Compute the nearest neighbors and store the output in the given matrices.
   * The matrices will be set to the size of n columns by k rows, where n is
   * the number of points in the query dataset and k is the number of neighbors
   * being searched for.
   *
   * @param k Number of neighbors to search for.
   * @param resultingNeighbors Matrix storing lists of neighbors for each query
   *     point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   * @param numTablesToSearch This parameter allows the user to have control
   *     over the number of hash tables to be searched. This allows
   *     the user to pick the number of tables it can afford for the time
   *     available without having to build hashing for every table size.
   *     By default, this is set to zero in which case all tables are
   *     considered.
   * @param T Number of probing bins.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& resultingNeighbors,
              arma::mat& distances,
              const size_t numTablesToSearch = 0,
              size_t T = 0);

  /**
   * Compute the recall (% of neighbors found) given the neighbors returned by
   * LSHSearch::Search and a "ground truth" set of neighbors.  The recall
   * returned will be in the range [0, 1].
   *
   * @param foundNeighbors Set of neighbors to compute recall of.
   * @param realNeighbors Set of "ground truth" neighbors to compute recall
   *     against.
   */
  static double ComputeRecall(const arma::Mat<size_t>& foundNeighbors,
                              const arma::Mat<size_t>& realNeighbors);

  /**
   * Serialize the LSH model.
   *
   * @param ar Archive to serialize to.
   * @param version serialize class version to provide backward compatibility
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

  //! Return the number of distance evaluations performed.
  size_t DistanceEvaluations() const { return distanceEvaluations; }
  //! Modify the number of distance evaluations performed.
  size_t& DistanceEvaluations() { return distanceEvaluations; }

  //! Return the reference dataset.
  const MatType& ReferenceSet() const { return referenceSet; }

  //! Get the number of projections.
  size_t NumProjections() const { return projections.n_slices; }

  //! Get the offsets 'b' for each of the projections.  (One 'b' per column.)
  const arma::mat& Offsets() const { return offsets; }

  //! Get the weights of the second hash.
  const arma::vec& SecondHashWeights() const { return secondHashWeights; }

  //! Get the bucket size of the second hash.
  size_t BucketSize() const { return bucketSize; }

  //! Get the second hash table.
  const std::vector<arma::Col<size_t>>& SecondHashTable() const
      { return secondHashTable; }

  //! Get the projection tables.
  const arma::cube& Projections() { return projections; }

  //! Change the projection tables (this retrains the LSH model).
  void Projections(const arma::cube& projTables)
  {
    // Simply call Train() with the given projection tables.
    Train(referenceSet, numProj, numTables, hashWidth, secondHashSize,
        bucketSize, projTables);
  }

 private:
  /**
   * This function takes a query and hashes it into each of the hash tables to
   * get keys for the query and then the key is hashed to a bucket of the second
   * hash table and all the points (if any) in those buckets are collected as
   * the potential neighbor candidates.
   *
   * @param queryPoint The query point currently being processed.
   * @param referenceIndices The list of neighbor candidates obtained from
   *    hashing the query into all the hash tables and eventually into
   *    multiple buckets of the second hash table.
   * @param numTablesToSearch The number of tables to perform the search in. If
   *    0, all tables are searched.
   * @param T The number of additional probing bins for multiprobe LSH. If 0,
   *    single-probe is used.
   */
  template<typename VecType>
  void ReturnIndicesFromTable(const VecType& queryPoint,
                              arma::uvec& referenceIndices,
                              size_t numTablesToSearch,
                              const size_t T) const;

  /**
   * This is a helper function that computes the distance of the query to the
   * neighbor candidates and appropriately stores the best 'k' candidates.  This
   * is specific to the monochromatic search case, where the query set is the
   * reference set.
   *
   * @param queryIndex The index of the query in question
   * @param referenceIndices The vector of indices of candidate neighbors for
   *    the query.
   * @param k Number of neighbors to search for.
   * @param neighbors Matrix holding output neighbors.
   * @param distances Matrix holding output distances.
   */
  void BaseCase(const size_t queryIndex,
                const arma::uvec& referenceIndices,
                const size_t k,
                arma::Mat<size_t>& neighbors,
                arma::mat& distances) const;

  /**
   * This is a helper function that computes the distance of the query to the
   * neighbor candidates and appropriately stores the best 'k' candidates.  This
   * is specific to bichromatic search, where the query set is not the same as
   * the reference set.
   *
   * @param queryIndex The index of the query in question
   * @param referenceIndices The vector of indices of candidate neighbors for
   *    the query.
   * @param k Number of neighbors to search for.
   * @param querySet Set of query points.
   * @param neighbors Matrix holding output neighbors.
   * @param distances Matrix holding output distances.
   */
  void BaseCase(const size_t queryIndex,
                const arma::uvec& referenceIndices,
                const size_t k,
                const MatType& querySet,
                arma::Mat<size_t>& neighbors,
                arma::mat& distances) const;

  /**
   * This function implements the core idea behind Multiprobe LSH. It is called
   * by ReturnIndicesFromTables when T > 0. Given a query's code and its
   * projection location, GetAdditionalProbingBins will calculate the T most
   * likely alternative bin codes (other than queryCode) where a query's
   * neighbors might be found in.
   *
   * @param queryCode vector containing the numProj-dimensional query code.
   * @param queryCodeNotFloored vector containing the projection location of the
   *    query.
   * @param T number of additional probing bins.
   * @param additionalProbingBins matrix. Each column will hold one additional
   *    bin.
  */
  void GetAdditionalProbingBins(const arma::vec& queryCode,
                                const arma::vec& queryCodeNotFloored,
                                const size_t T,
                                arma::mat& additionalProbingBins) const;

  /**
   * Returns the score of a perturbation vector generated by perturbation set A.
   * The score of a pertubation set (vector) is the sum of scores of the
   * participating actions.
   * @param A perturbation set to compute the score of.
   * @param scores vector containing score of each perturbation.
  */
  double PerturbationScore(const std::vector<bool>& A,
                           const arma::vec& scores) const;

  /**
   * Inline function used by GetAdditionalProbingBins. The vector shift
   * operation replaces the largest element of a vector A with (largest element)
   * + 1.  Returns true if resulting vector is valid, otherwise false.
   *
   * @param A perturbation set to shift.
   */
  bool PerturbationShift(std::vector<bool>& A) const;

  /**
   * Inline function used by GetAdditionalProbingBins. The vector expansion
   * operation adds the element [1 + (largest_element)] to a vector A, where
   * largest_element is the largest element of A. Returns true if resulting
   * vector is valid, otherwise false.
   *
   * @param A perturbation set to expand.
   */
  bool PerturbationExpand(std::vector<bool>& A) const;

  /**
   * Return true if perturbation set A is valid. A perturbation set is invalid
   * if it contains two (or more) actions for the same dimension or dimensions
   * that are larger than the queryCode's dimensions.
   *
   * @param A perturbation set to validate.
   */
  bool PerturbationValid(const std::vector<bool>& A) const;

  //! Reference dataset.
  MatType referenceSet;

  //! The number of projections.
  size_t numProj;
  //! The number of hash tables.
  size_t numTables;

  //! The arma::cube containing the projection matrix of each table.
  arma::cube projections; // should be [numProj x dims] x numTables slices

  //! The list of the offsets 'b' for each of the projection for each table.
  arma::mat offsets; // should be numProj x numTables

  //! The hash width.
  double hashWidth;

  //! The big prime representing the size of the second hash.
  size_t secondHashSize;

  //! The weights of the second hash.
  arma::vec secondHashWeights;

  //! The bucket size of the second hash.
  size_t bucketSize;

  //! The final hash table; should be (< secondHashSize) vectors each with
  //! (<= bucketSize) elements.
  std::vector<arma::Col<size_t>> secondHashTable;

  //! The number of elements present in each hash bucket; should be
  //! secondHashSize.
  arma::Col<size_t> bucketContentSize;

  //! For a particular hash value, points to the row in secondHashTable
  //! corresponding to this value. Length secondHashSize.
  arma::Col<size_t> bucketRowInHashTable;

  //! The number of distance evaluations.
  size_t distanceEvaluations;

  //! Candidate represents a possible candidate neighbor (distance, index).
  using Candidate = std::pair<double, size_t>;

  //! Compare two candidates based on the distance.
  struct CandidateCmp {
    bool operator()(const Candidate& c1, const Candidate& c2)
    {
      return !SortPolicy::IsBetter(c2.first, c1.first);
    };
  };

  //! Use a priority queue to represent the list of candidate neighbors.
  using CandidateList = std::priority_queue<Candidate, std::vector<Candidate>,
      CandidateCmp>;
}; // class LSHSearch

} // namespace mlpack

// Include implementation.
#include "lsh_search_impl.hpp"

#endif
