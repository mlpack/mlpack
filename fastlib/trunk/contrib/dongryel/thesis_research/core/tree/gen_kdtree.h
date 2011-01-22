/** @file tree/kdtree.h
 *
 *  The generic kd-tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_KDTREE_H
#define CORE_TREE_GEN_KDTREE_H

#include <boost/mpi.hpp>
#include "core/tree/general_spacetree.h"
#include "core/tree/hrect_bound.h"

namespace core {
namespace parallel {
class HrectBoundCombine:
  public std::binary_function <
    core::tree::HrectBound, core::tree::HrectBound, core::tree::HrectBound > {
  public:
    const core::tree::HrectBound operator()(
      const core::tree::HrectBound &a, const core::tree::HrectBound &b) const {
      core::tree::HrectBound combined_box;
      combined_box.Init(a.dim());
      combined_box |= a;
      combined_box |= b;
      return combined_box;
    }
};
}
}

namespace boost {
namespace mpi {
template<>
class is_commutative <
  core::parallel::HrectBoundCombine, core::tree::HrectBound > :
  public boost::mpl::true_ {

};
}
}

namespace core {
namespace tree {

/** @brief The generic midpoint splitting specification for kd-tree.
 */
class GenKdTreeMidpointSplitter {
  public:
    template<typename TKdTree>
    static double ChooseKdTreeSplitValue(
      const core::table::DenseMatrix &matrix,
      TKdTree *node, int split_dim) {
      return node->bound().get(split_dim).mid();
    }
};

/** @brief The specification of the kd-tree.
 */
template< typename IncomingStatisticType >
class GenKdTree {
  public:

    typedef core::tree::HrectBound BoundType;

    typedef IncomingStatisticType StatisticType;

    template<typename MetricType>
    static void FindBoundFromMatrix(
      const MetricType &metric_in,
      const core::table::DenseMatrix &matrix,
      int first, int count, BoundType *bounds) {

      int end = first + count;
      for(int i = first; i < end; i++) {
        core::table::DensePoint col;
        matrix.MakeColumnVector(i, &col);
        *bounds |= col;
      }
    }

    /** @brief The parallel MPI version of finding the bound for which
     *         the reduction is done over a MPI communicator.
     */
    template<typename MetricType>
    static void FindBoundFromMatrix(
      boost::mpi::communicator &comm,
      const MetricType &metric_in,
      const core::table::DenseMatrix &matrix,
      BoundType *combined_bound) {

      // Each MPI process finds a local bound.
      BoundType local_bound;
      FindBoundFromMatrix(
        metric_in, matrix, 0, matrix.n_cols(), &local_bound);

      // Call reduction.
      boost::mpi::all_reduce(
        comm, local_bound, *combined_bound,
        core::parallel::HrectBoundCombine());
    }

    template<typename MetricType>
    static void MakeLeafNode(
      const MetricType &metric_in,
      const core::table::DenseMatrix& matrix,
      int begin, int count, BoundType *bounds) {

      FindBoundFromMatrix(metric_in, matrix, begin, count, bounds);
    }

    template<typename MetricType, typename TreeType>
    static void CombineBounds(
      const MetricType &metric_in,
      core::table::DenseMatrix &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

    }

    template<typename MetricType>
    static void ComputeMemberships(
      const MetricType &metric_in,
      const core::table::DenseMatrix &matrix,
      int first, int end,
      BoundType &left_bound, BoundType &right_bound,
      int *left_count, std::deque<bool> *left_membership) {

      // Get the split dimension and the split value.
      int split_dim = static_cast<int>(left_bound.get(0).lo);
      double split_val = left_bound.get(0).hi;

      // Reset the left bound.
      left_bound.Reset();

      // Build the bounds for the kd-tree.
      for(int left = first; left < end; left++) {

        // Make alias of the current point.
        core::table::DensePoint point;
        matrix.MakeColumnVector(left, &point);

        // We swap if the point is further away from the left pivot.
        if(point[split_dim] > split_val) {
          (*left_membership)[left - first] = false;
          right_bound |= point;
        }
        else {
          (*left_membership)[left - first] = true;
          left_bound |= point;
          (*left_count)++;
        }
      }
    }

    template<typename MetricType, typename TreeType, typename IndexType>
    static bool AttemptSplitting(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix, TreeType *node, TreeType **left,
      TreeType **right, int leaf_size,
      IndexType *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      *left = NULL;
      *right = NULL;

      if(node->count() > leaf_size) {
        int split_dim = -1;
        double max_width = -1;

        for(int d = 0; d < matrix.n_rows(); d++) {
          double w = node->bound().get(d).width();

          if(w > max_width) {
            max_width = w;
            split_dim = d;
          }
        }

        // Choose the split value along the dimension to be splitted.
        double split_val =
          core::tree::GenKdTreeMidpointSplitter::ChooseKdTreeSplitValue(
            matrix, node, split_dim);

        if(max_width < std::numeric_limits<double>::epsilon()) {
          return false;
        }
        else {
          *left = (m_file_in) ?
                  m_file_in->Construct<TreeType>() : new TreeType();
          *right = (m_file_in) ?
                   m_file_in->Construct<TreeType>() : new TreeType();

          // Copy the split dimension and split value.
          (*left)->bound().Init(matrix.n_rows());
          (*right)->bound().Init(matrix.n_rows());
          (*left)->bound().get(0).lo = split_dim;
          (*left)->bound().get(0).hi = split_val;

          int left_count = TreeType::MatrixPartition(
                             metric_in, matrix, node->begin(), node->count(),
                             (*left)->bound(), (*right)->bound(), old_from_new);

          (*left)->Init(node->begin(), left_count);
          (*right)->Init(
            node->begin() + left_count, node->count() - left_count);
        }
        return true;
      }
      return false;
    }
};
}
}

#endif
