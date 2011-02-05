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

/** @brief A class that combines two bounding boxes and produces the
 *         tightest bounding box that contains both.
 */
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

/** @brief HrectBoundCombine function is a commutative reduction
 *         operator.
 */
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

    /** @brief Computes the widest dimension and its width of a
     *         bounding box.
     */
    template<typename BoundType>
    static void ComputeWidestDimension(
      const BoundType &bound, int *split_dim, double *max_width) {

      *split_dim = -1;
      *max_width = -1.0;
      for(int d = 0; d < bound.dim(); d++) {
        double w = bound.get(d).width();
        if(w > *max_width) {
          *max_width = w;
          *split_dim = d;
        }
      }
    }

    /** @brief The splitter that simply returns the mid point of the
     *         splitting dimension.
     */
    template<typename BoundType>
    static double ChooseKdTreeSplitValue(
      const BoundType &bound, int split_dim) {
      return bound.get(split_dim).mid();
    }
};

/** @brief The specification of the kd-tree.
 */
template< typename IncomingStatisticType >
class GenKdTree {
  public:

    /** @brief The bounding primitive used in kd-tree.
     */
    typedef core::tree::HrectBound BoundType;

    /** @brief The statistics type used in the tree.
     */
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
      local_bound.Init(matrix.n_rows());
      FindBoundFromMatrix(
        metric_in, matrix, 0, matrix.n_cols(), &local_bound);

      // Call reduction.
      boost::mpi::all_reduce(
        comm, local_bound, *combined_bound,
        core::parallel::HrectBoundCombine());
    }

    /** @brief Makes a leaf node by constructing its bound.
     */
    template<typename MetricType>
    static void MakeLeafNode(
      const MetricType &metric_in,
      const core::table::DenseMatrix& matrix,
      int begin, int count, BoundType *bounds) {

      FindBoundFromMatrix(metric_in, matrix, begin, count, bounds);
    }

    /** @brief Combines the bounding primitives of the children node
     *         to form the bound for the self.
     */
    template<typename MetricType, typename TreeType>
    static void CombineBounds(
      const MetricType &metric_in,
      core::table::DenseMatrix &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

      // Do nothing.
    }

    /** @brief Computes two bounding primitives and membership vectors
     *         for a given consecutive column points in the data
     *         matrix.
     */
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

      // Reset the left bound and the right bound.
      left_bound.Reset();
      right_bound.Reset();
      *left_count = 0;
      left_membership->resize(end - first);

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

    template<typename MetricType>
    static bool AttemptSplitting(
      boost::mpi::communicator &comm,
      const MetricType &metric_in,
      const BoundType &bound,
      const core::table::DenseMatrix &matrix_in,
      std::vector< std::vector<int> > *assigned_point_indices,
      std::vector<int> *membership_counts_per_process) {

      // Splitting dimension/widest dimension info.
      int split_dim = -1;
      double max_width = -1;

      // Find the splitting dimension.
      core::tree::GenKdTreeMidpointSplitter::ComputeWidestDimension(
        bound, &split_dim, &max_width);

      // Choose the split value along the dimension to be splitted.
      double split_val =
        core::tree::GenKdTreeMidpointSplitter::ChooseKdTreeSplitValue(
          bound, split_dim);

      if(max_width < std::numeric_limits<double>::epsilon()) {
        return false;
      }

      // Copy the split dimension and split value.
      BoundType left_bound;
      left_bound.Init(bound.dim());
      left_bound.get(0).lo = split_dim;
      left_bound.get(0).hi = split_val;
      BoundType right_bound;
      right_bound.Init(bound.dim());

      // Assign the point on the local process using the splitting
      // value.
      int left_count;
      std::deque<bool> left_membership;
      ComputeMemberships(
        metric_in, matrix_in, 0, matrix_in.n_cols(), left_bound, right_bound,
        &left_count, &left_membership);

      // The assigned point indices per process and per-process counts
      // will be outputted.
      assigned_point_indices->resize(comm.size());
      membership_counts_per_process->resize(comm.size());

      // Loop through the membership vectors and assign to the right
      // process partner.
      int left_destination =
        (comm.rank() % 2 == 0) ? comm.rank() : comm.rank() - 1;
      int right_destination = (comm.rank() % 2 == 0) ?
                              comm.rank() + 1 : comm.rank();
      right_destination = right_destination % comm.size();
      for(unsigned int i = 0; i < left_membership.size(); i++) {
        if(left_membership[i]) {
          (*assigned_point_indices)[left_destination].push_back(i);
          (*membership_counts_per_process)[left_destination]++;
        }
        else {
          (*assigned_point_indices)[right_destination].push_back(i);
          (*membership_counts_per_process)[right_destination]++;
        }
      }
      return true;
    }

    /** @brief Attempts to split a kd-tree node and reshuffles the
     *         data accordingly and creates two child nodes.
     */
    template<typename MetricType, typename TreeType, typename IndexType>
    static bool AttemptSplitting(
      const MetricType &metric_in,
      core::table::DenseMatrix& matrix, TreeType *node, TreeType **left,
      TreeType **right, int leaf_size,
      IndexType *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      int split_dim = -1;
      double max_width = -1;

      // Find the splitting dimension.
      core::tree::GenKdTreeMidpointSplitter::ComputeWidestDimension(
        node->bound(), &split_dim, &max_width);

      // Choose the split value along the dimension to be splitted.
      double split_val =
        core::tree::GenKdTreeMidpointSplitter::ChooseKdTreeSplitValue(
          node->bound(), split_dim);

      // Allocate the children and its bound.
      *left = (m_file_in) ?
              m_file_in->Construct<TreeType>() : new TreeType();
      *right = (m_file_in) ?
               m_file_in->Construct<TreeType>() : new TreeType();
      (*left)->bound().Init(matrix.n_rows());
      (*right)->bound().Init(matrix.n_rows());

      int left_count = 0;
      if(max_width < std::numeric_limits<double>::epsilon()) {

        // Give the first half to the left and second half to the
        // right.
        left_count = node->count() / 2;
        FindBoundFromMatrix(
          metric_in, matrix, node->begin(), left_count, & ((*left)->bound()));
        FindBoundFromMatrix(
          metric_in, matrix, node->begin() + left_count,
          node->count() - left_count, & ((*right)->bound()));
      }
      else {

        // Copy the split dimension and split value.
        (*left)->bound().get(0).lo = split_dim;
        (*left)->bound().get(0).hi = split_val;

        left_count = TreeType::MatrixPartition(
                       metric_in, matrix, node->begin(), node->count(),
                       (*left)->bound(), (*right)->bound(), old_from_new);
      }
      (*left)->Init(node->begin(), left_count);
      (*right)->Init(
        node->begin() + left_count, node->count() - left_count);

      return true;
    }
};
}
}

#endif
