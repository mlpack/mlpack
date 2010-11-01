/** @file gen_metric.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_METRIC_TREE_H
#define CORE_TREE_GEN_METRIC_TREE_H

#include <vector>
#include "ball_bound.h"
#include "general_spacetree.h"
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace tree {
template<typename PointType>
class GenMetricTree {
  public:

    typedef core::tree::BallBound<PointType> BoundType;

  private:
    static int FurthestColumnIndex_(
      const core::metric_kernels::AbstractMetric &metric_in,
      const PointType &pivot,
      const core::table::DenseMatrix &matrix,
      int begin, int count,
      double *furthest_distance) {

      int furthest_index = -1;
      int end = begin + count;
      *furthest_distance = -1.0;

      for(int i = begin; i < end; i++) {
        core::table::DenseConstPoint point;
        matrix.MakeColumnVector(i, &point);
        double distance_between_center_and_point =
          metric_in.Distance(pivot, point);

        if((*furthest_distance) < distance_between_center_and_point) {
          *furthest_distance = distance_between_center_and_point;
          furthest_index = i;
        }
      }

      return furthest_index;
    }

  public:
    template<typename TBound>
    static void MakeLeafNode(
      const core::metric_kernels::AbstractMetric &metric_in,
      const core::table::DenseMatrix& matrix,
      int begin, int count, TBound *bounds) {

      bounds->center().SetZero();

      int end = begin + count;
      core::table::DenseConstPoint col_point;
      for(int i = begin; i < end; i++) {
        matrix.MakeColumnVector(i, &col_point);
        bounds->center() += col_point;
      }
      bounds->center() /= ((double) count);

      double furthest_distance;
      FurthestColumnIndex_(
        metric_in, bounds->center(), matrix, begin, count, &furthest_distance);
      bounds->set_radius(furthest_distance);
    }

    template<typename TreeType>
    static void CombineBounds(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

      // Compute the weighted sum of the two pivots
      node->bound().center().CopyValues(left->bound().center());
      node->bound().center() *= left->count();
      node->bound().center().Add(right->count(), right->bound().center());
      node->bound().center() /= ((double) node->count());

      double left_max_dist, right_max_dist;
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, left->begin(),
        left->count(), &left_max_dist);
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, right->begin(),
        right->count(), &right_max_dist);
      node->bound().set_radius(std::max(left_max_dist, right_max_dist));
    }

    template<typename TreeType>
    static bool AttemptSplitting(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix& matrix, TreeType *node, TreeType **left,
      TreeType **right, int leaf_size,
      std::vector<int> *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      // Pick a random row.
      int random_row = core::math::RandInt(
                         node->begin(), node->begin() + node->count());
      core::table::DensePoint random_row_vec;
      matrix.MakeColumnVector(random_row, & random_row_vec);

      // Now figure out the furthest point from the random row picked
      // above.
      double furthest_distance;
      int furthest_from_random_row =
        FurthestColumnIndex_(
          metric_in, random_row_vec, matrix, node->begin(), node->count(),
          &furthest_distance);
      core::table::DensePoint furthest_from_random_row_vec;
      matrix.MakeColumnVector(
        furthest_from_random_row, &furthest_from_random_row_vec);

      // Then figure out the furthest point from the furthest point.
      double furthest_from_furthest_distance;
      int furthest_from_furthest_random_row =
        FurthestColumnIndex_(
          metric_in, furthest_from_random_row_vec, matrix, node->begin(),
          node->count(), &furthest_from_furthest_distance);
      core::table::DensePoint furthest_from_furthest_random_row_vec;
      matrix.MakeColumnVector(
        furthest_from_furthest_random_row,
        &furthest_from_furthest_random_row_vec);

      if(furthest_from_furthest_distance <
          std::numeric_limits<double>::epsilon()) {
        return false;
      }
      else {
        *left = (m_file_in) ? (TreeType *)
                m_file_in->Allocate(sizeof(TreeType)) : new TreeType();
        *right = (m_file_in) ? (TreeType *)
                 m_file_in->Allocate(sizeof(TreeType)) : new TreeType();

        ((*left)->bound().center()).Copy(furthest_from_random_row_vec);
        ((*right)->bound().center()).Copy(
          furthest_from_furthest_random_row_vec);

        int left_count = TreeType::MatrixPartition(
                           metric_in, matrix, node->begin(), node->count(),
                           (*left)->bound(), (*right)->bound(), old_from_new);

        (*left)->Init(node->begin(), left_count);
        (*right)->Init(node->begin() + left_count, node->count() - left_count);
      }

      return true;
    }
};
};
};

#endif
