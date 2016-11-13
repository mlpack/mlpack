/**
 * @file drusilla_select.hpp
 * @author Ryan Curtin
 *
 * An implementation of the approximate furthest neighbor algorithm specified in
 * the following paper:
 *
 * @code
 * @incollection{curtin2016fast,
 *   title={Fast approximate furthest neighbors with data-dependent candidate
 *          selection},
 *   author={Curtin, R.R., and Gardner, A.B.},
 *   booktitle={Similarity Search and Applications},
 *   pages={221--235},
 *   year={2016},
 *   publisher={Springer}
 * }
 * @endcode
 *
 * This algorithm, called DrusillaSelect, constructs a candidate set of points
 * to query to find an approximate furthest neighbor.  The strange name is a
 * result of the algorithm being named after a cat.  The cat in question may be
 * viewed at http://www.ratml.org/misc_img/drusilla_fence.png.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_APPROX_KFN_DRUSILLA_SELECT_HPP
#define MLPACK_METHODS_APPROX_KFN_DRUSILLA_SELECT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

template<typename MatType = arma::mat>
class DrusillaSelect
{
 public:
  /**
   * Construct the DrusillaSelect object with the given reference set (this is
   * the set that will be searched).  The resulting set of candidate points that
   * will be searched at query time will have size l*m.
   *
   * @param referenceSet Set of reference data.
   * @param l Number of projections.
   * @param m Number of elements to store for each projection.
   */
  DrusillaSelect(const MatType& referenceSet,
                 const size_t l,
                 const size_t m);

  /**
   * Construct the DrusillaSelect object with no given reference set.  Be sure
   * to call Train() before calling Search()!
   *
   * @param l Number of projections.
   * @param m Number of elements to store for each projection.
   */
  DrusillaSelect(const size_t l, const size_t m);

  /**
   * Build the set of candidate points on the given reference set.  If l and m
   * are left unspecified, then the values set in the constructor will be used
   * instead.
   *
   * @param referenceSet Set to extract candidate points from.
   * @param l Number of projections.
   * @param m Number of elements to store for each projection.
   */
  void Train(const MatType& referenceSet,
             const size_t l = 0,
             const size_t m = 0);

  /**
   * Search for the k furthest neighbors of the given query set.  (The query set
   * can contain just one point: that is okay.)  The results will be stored in
   * the given neighbors and distances matrices, in the same format as the
   * NeighborSearch and LSHSearch classes.  That is, each column in the
   * neighbors and distances matrices will refer to a single query point, and
   * the k'th row in that column will refer to the k'th candidate neighbor or
   * distance for that query point.
   *
   * @param querySet Set of query points to search.
   * @param k Number of furthest neighbors to search for.
   * @param neighbors Matrix to store resulting neighbors in.
   * @param distances Matrix to store resulting distances in.
   */
  void Search(const MatType& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Access the candidate set.
  const MatType& CandidateSet() const { return candidateSet; }
  //! Modify the candidate set.  Be careful!
  MatType& CandidateSet() { return candidateSet; }

  //! Access the indices of points in the candidate set.
  const arma::Col<size_t>& CandidateIndices() const { return candidateIndices; }
  //! Modify the indices of points in the candidate set.  Be careful!
  arma::Col<size_t>& CandidateIndices() { return candidateIndices; }

 private:
  //! The reference set.
  MatType candidateSet;
  //! Indices of each point in the reference set.
  arma::Col<size_t> candidateIndices;

  //! The number of projections.
  size_t l;
  //! The number of points in each projection.
  size_t m;
};

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "drusilla_select_impl.hpp"

#endif
