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
   * Search for the k furthest neighbors of the given query set.  (The query set
   * can contain just one point, that is okay.)  The results will be stored in
   * the given neighbors and distances matrices, in the same format as the
   * mlpack NeighborSearch and LSHSearch classes.
   */
  void Search(const MatType& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

 private:
  //! The reference set.
  const MatType& referenceSet;

  //! The number of projections.
  const size_t l;
  //! The number of elements to store for each projection.
  const size_t m;
  //! The random lines we are projecting onto.  Has l columns.
  arma::mat lines;
  //! Projections of each point onto each random line.
  arma::mat projections;

  //! Indices of the points for each S.
  arma::Mat<size_t> sIndices;
  //! Values of a_i * x for each point in S.
  arma::mat sValues;

  //! Insert a neighbor into a set of results for a given query point.
  void InsertNeighbor(arma::mat& distances,
                      arma::Mat<size_t>& neighbors,
                      const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance) const;
};

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "qdafn_impl.hpp"

#endif
