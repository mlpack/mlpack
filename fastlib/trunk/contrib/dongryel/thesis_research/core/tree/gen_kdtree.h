/** @file tree/kdtree.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_KDTREE_H
#define CORE_TREE_GEN_KDTREE_H

#include "core/tree/general_spacetree.h"
#include "core/tree/hrect_bound.h"


namespace core {
namespace tree {

class GenKdTreeMidpointSplitter {
  public:
    template<typename TKdTree>
    static double ChooseKdTreeSplitValue(
      const core::table::DenseMatrix &matrix,
      TKdTree *node, int split_dim) {
      return node->bound().get(split_dim).mid();
    }
};

template< typename IncomingStatisticType >
class GenKdTree {
  public:

    typedef core::tree::HrectBound BoundType;

    typedef IncomingStatisticType StatisticType;

    static void FindBoundFromMatrix(
      const core::table::DenseMatrix &matrix,
      int first, int count, BoundType *bounds) {

      int end = first + count;
      for(int i = first; i < end; i++) {
        core::table::DensePoint col;
        matrix.MakeColumnVector(i, &col);
        *bounds |= col;
      }
    }

    static void MakeLeafNode(
      const core::metric_kernels::AbstractMetric &metric_in,
      const core::table::DenseMatrix& matrix,
      int begin, int count, BoundType *bounds) {

      FindBoundFromMatrix(matrix, begin, count, bounds);
    }

    template<typename TreeType>
    static void CombineBounds(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

    }

    static void ComputeMemberships(
      const core::metric_kernels::AbstractMetric &metric_in,
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

    template<typename TreeType>
    static bool AttemptSplitting(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix& matrix, TreeType *node, TreeType **left,
      TreeType **right, int leaf_size,
      int *old_from_new,
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
};
};

#endif
