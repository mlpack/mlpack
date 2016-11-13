/**
 * @file qdafn.hpp
 * @author Ryan Curtin
 *
 * An implementation of the query-dependent approximate furthest neighbor
 * algorithm specified in the following paper:
 *
 * @code
 * @incollection{pagh2015approximate,
 *   title={Approximate furthest neighbor in high dimensions},
 *   author={Pagh, R. and Silvestri, F. and Sivertsen, J. and Skala, M.},
 *   booktitle={Similarity Search and Applications},
 *   pages={3--14},
 *   year={2015},
 *   publisher={Springer}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_APPROX_KFN_QDAFN_HPP
#define MLPACK_METHODS_APPROX_KFN_QDAFN_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

template<typename MatType = arma::mat>
class QDAFN
{
 public:
  /**
   * Construct the QDAFN object but do not train it.  Be sure to call Train()
   * before calling Search().
   *
   * @param l Number of projections.
   * @param m Number of elements to store for each projection.
   */
  QDAFN(const size_t l, const size_t m);

  /**
   * Construct the QDAFN object with the given reference set (this is the set
   * that will be searched).
   *
   * @param referenceSet Set of reference data.
   * @param l Number of projections.
   * @param m Number of elements to store for each projection.
   */
  QDAFN(const MatType& referenceSet,
        const size_t l,
        const size_t m);

  /**
   * Train the QDAFN model on the given reference set, optionally setting new
   * parameters for the number of projections/tables (l) and the number of
   * elements stored for each projection/table (m).
   *
   * @param referenceSet Reference set to train on.
   * @param l Number of projections.
   * @param m Number of elements to store for each projection.
   */
  void Train(const MatType& referenceSet,
             const size_t l = 0,
             const size_t m = 0);

  /**
   * Search for the k furthest neighbors of the given query set.  (The query set
   * can contain just one point, that is okay.)  The results will be stored in
   * the given neighbors and distances matrices, in the same format as the
   * mlpack NeighborSearch and LSHSearch classes.
   */
  void Search(const MatType& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Get the number of projections.
  size_t NumProjections() const { return candidateSet.size(); }

  //! Get the candidate set for the given projection table.
  const MatType& CandidateSet(const size_t t) const { return candidateSet[t]; }
  //! Modify the candidate set for the given projection table.  Careful!
  MatType& CandidateSet(const size_t t) { return candidateSet[t]; }

 private:
  //! The number of projections.
  size_t l;
  //! The number of elements to store for each projection.
  size_t m;
  //! The random lines we are projecting onto.  Has l columns.
  arma::mat lines;
  //! Projections of each point onto each random line.
  arma::mat projections;

  //! Indices of the points for each S.
  arma::Mat<size_t> sIndices;
  //! Values of a_i * x for each point in S.
  arma::mat sValues;

  // Candidate sets; one element in the vector for each table.
  std::vector<MatType> candidateSet;
};

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "qdafn_impl.hpp"

#endif
