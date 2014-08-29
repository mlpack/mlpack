/**
 * @file fastmks.hpp
 * @author Ryan Curtin
 *
 * Definition of the FastMKS class, which implements fast exact max-kernel
 * search.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/ip_metric.hpp>
#include "fastmks_stat.hpp"
#include <mlpack/core/tree/cover_tree.hpp>

namespace mlpack {
namespace fastmks /** Fast max-kernel search. */ {

/**
 * An implementation of fast exact max-kernel search.  Given a query dataset and
 * a reference dataset (or optionally just a reference dataset which is also
 * used as the query dataset), fast exact max-kernel search finds, for each
 * point in the query dataset, the k points in the reference set with maximum
 * kernel value K(p_q, p_r), where k is a specified parameter and K() is a
 * Mercer kernel.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{curtin2013fast,
 *   title={Fast Exact Max-Kernel Search},
 *   author={Curtin, Ryan R. and Ram, Parikshit and Gray, Alexander G.},
 *   booktitle={Proceedings of the 2013 SIAM International Conference on Data
 *       Mining (SDM 13)},
 *   year={2013}
 * }
 * @endcode
 *
 * This class allows specification of the type of kernel and also of the type of
 * tree.  FastMKS can be run on kernels that work on arbitrary objects --
 * however, this only works with cover trees and other trees that are built only
 * on points in the dataset (and not centroids of regions or anything like
 * that).
 *
 * @tparam KernelType Type of kernel to run FastMKS with.
 * @tparam TreeType Type of tree to run FastMKS with; it must have metric
 *     IPMetric<KernelType>.
 */
template<
    typename KernelType,
    typename TreeType = tree::CoverTree<metric::IPMetric<KernelType>,
        tree::FirstPointIsRoot, FastMKSStat>
>
class FastMKS
{
 public:
  /**
   * Create the FastMKS object using the reference set as the query set.
   * Optionally, specify whether or not single-tree search or naive
   * (brute-force) search should be used.
   *
   * @param referenceSet Set of data to run FastMKS on.
   * @param single Whether or not to run single-tree search.
   * @param naive Whether or not to run brute-force (naive) search.
   */
  FastMKS(const arma::mat& referenceSet,
          const bool single = false,
          const bool naive = false);

  /**
   * Create the FastMKS object using separate reference and query sets.
   * Optionally, specify whether or not single-tree search or naive
   * (brute-force) search should be used.
   *
   * @param referenceSet Reference set of data for FastMKS.
   * @param querySet Set of query points for FastMKS.
   * @param single Whether or not to run single-tree search.
   * @param naive Whether or not to run brute-force (naive) search.
   */
  FastMKS(const arma::mat& referenceSet,
          const arma::mat& querySet,
          const bool single = false,
          const bool naive = false);

  /**
   * Create the FastMKS object using the reference set as the query set, and
   * with an initialized kernel.  This is useful for when the kernel stores
   * state.  Optionally, specify whether or not single-tree search or naive
   * (brute-force) search should be used.
   *
   * @param referenceSet Reference set of data for FastMKS.
   * @param kernel Initialized kernel.
   * @param single Whether or not to run single-tree search.
   * @param naive Whether or not to run brute-force (naive) search.
   */
  FastMKS(const arma::mat& referenceSet,
          KernelType& kernel,
          const bool single = false,
          const bool naive = false);

  /**
   * Create the FastMKS object using separate reference and query sets, and with
   * an initialized kernel.  This is useful for when the kernel stores state.
   * Optionally, specify whether or not single-tree search or naive
   * (brute-force) search should be used.
   *
   * @param referenceSet Reference set of data for FastMKS.
   * @param querySet Set of query points for FastMKS.
   * @param kernel Initialized kernel.
   * @param single Whether or not to run single-tree search.
   * @param naive Whether or not to run brute-force (naive) search.
   */
  FastMKS(const arma::mat& referenceSet,
          const arma::mat& querySet,
          KernelType& kernel,
          const bool single = false,
          const bool naive = false);

  /**
   * Create the FastMKS object with an already-initialized tree built on the
   * reference points.  Be sure that the tree is built with the metric type
   * IPMetric<KernelType>.  For this constructor, the reference set and the
   * query set are the same points.  Optionally, whether or not to run
   * single-tree search or brute-force (naive) search can be specified.
   *
   * @param referenceSet Reference set of data for FastMKS.
   * @param referenceTree Tree built on reference data.
   * @param single Whether or not to run single-tree search.
   * @param naive Whether or not to run brute-force (naive) search.
   */
  FastMKS(const arma::mat& referenceSet,
          TreeType* referenceTree,
          const bool single = false,
          const bool naive = false);

  /**
   * Create the FastMKS object with already-initialized trees built on the
   * reference and query points.  Be sure that the trees are built with the
   * metric type IPMetric<KernelType>.  Optionally, whether or not to run
   * single-tree search or naive (brute-force) search can be specified.
   *
   * @param referenceSet Reference set of data for FastMKS.
   * @param referenceTree Tree built on reference data.
   * @param querySet Set of query points for FastMKS.
   * @param queryTree Tree built on query data.
   * @param single Whether or not to use single-tree search.
   * @param naive Whether or not to use naive (brute-force) search.
   */
  FastMKS(const arma::mat& referenceSet,
          TreeType* referenceTree,
          const arma::mat& querySet,
          TreeType* queryTree,
          const bool single = false,
          const bool naive = false);

  //! Destructor for the FastMKS object.
  ~FastMKS();

  /**
   * Search for the maximum inner products of the query set (or if no query set
   * was passed, the reference set is used).  The resulting maximum inner
   * products are stored in the products matrix and the corresponding point
   * indices are stores in the indices matrix.  The results for each point in
   * the query set are stored in the corresponding column of the indices and
   * products matrices; for instance, the index of the point with maximum inner
   * product to point 4 in the query set will be stored in row 0 and column 4 of
   * the indices matrix.
   *
   * @param k The number of maximum kernels to find.
   * @param indices Matrix to store resulting indices of max-kernel search in.
   * @param products Matrix to store resulting max-kernel values in.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& indices,
              arma::mat& products);

  //! Get the inner-product metric induced by the given kernel.
  const metric::IPMetric<KernelType>& Metric() const { return metric; }
  //! Modify the inner-product metric induced by the given kernel.
  metric::IPMetric<KernelType>& Metric() { return metric; }

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

 private:
  //! The reference dataset.
  const arma::mat& referenceSet;
  //! The query dataset.
  const arma::mat& querySet;

  //! The tree built on the reference dataset.
  TreeType* referenceTree;
  //! The tree built on the query dataset.  This is NULL if there is no query
  //! set.
  TreeType* queryTree;

  //! If true, this object created the trees and is responsible for them.
  bool treeOwner;

  //! If true, single-tree search is used.
  bool single;
  //! If true, naive (brute-force) search is used.
  bool naive;

  //! The instantiated inner-product metric induced by the given kernel.
  metric::IPMetric<KernelType> metric;

  //! Utility function.  Copied too many times from too many places.
  void InsertNeighbor(arma::Mat<size_t>& indices,
                      arma::mat& products,
                      const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "fastmks_impl.hpp"

#endif
