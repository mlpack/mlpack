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
    template<typename T, typename TKdTree>
    static double ChooseKdTreeSplitValue(const GenMatrix<T>& matrix,
                                         TKdTree *node, int split_dim) {
      return node->bound().get(split_dim).mid();
    }

    template<typename T, typename TKdTree>
    static double ChooseKdTreeSplitValue
    (const GenMatrix<T>& lower_limit_matrix,
     const GenMatrix<T>& upper_limit_matrix, TKdTree *node, int split_dim) {
      return node->bound().get(split_dim).mid();
    }
};

class GenKdTreeMedianSplitter {

  private:
    template<typename T>
    static int qsort_compar(const void *a, const void *b) {

      T *a_dbl = (T *) a;
      T *b_dbl = (T *) b;

      if(*a_dbl < *b_dbl) {
        return -1;
      }
      else if(*a_dbl > *b_dbl) {
        return 1;
      }
      else {
        return 0;
      }
    }

  public:
    template<typename T, typename TKdTree>
    static double ChooseKdTreeSplitValue(const GenMatrix<T>& matrix,
                                         TKdTree *node, int split_dim) {
      GenVector<T> coordinate_vals;
      coordinate_vals.Init(node->count());
      for(int i = node->begin(); i < node->end(); i++) {
        coordinate_vals[i - node->begin()] = matrix.get(split_dim, i);
      }

      // sort coordinate value
      qsort(coordinate_vals.ptr(), node->count(), sizeof(T),
            &GenKdTreeMedianSplitter::qsort_compar<T>);

      double split_val = (double) coordinate_vals[node->count() / 2];
      if(split_val == coordinate_vals[0] ||
          split_val == coordinate_vals[node->count() - 1]) {
        split_val = 0.5 * (coordinate_vals[0] +
                           coordinate_vals[node->count() - 1]);
      }

      return split_val;
    }

    template<typename T, typename TKdTree>
    static double ChooseKdTreeSplitValue
    (const GenMatrix<T>& lower_limit_matrix,
     const GenMatrix<T>& upper_limit_matrix, TKdTree *node, int split_dim) {

      GenVector<T> coordinate_vals;
      coordinate_vals.Init(node->count());
      for(int i = node->begin(); i < node->end(); i++) {
        coordinate_vals[i - node->begin()] =
          lower_limit_matrix.get(split_dim, i);
      }

      // sort coordinate value
      qsort(coordinate_vals.ptr(), node->count(), sizeof(T),
            &GenKdTreeMedianSplitter::qsort_compar<T>);

      double split_val = (double) coordinate_vals[node->count() / 2];
      if(split_val == coordinate_vals[0] ||
          split_val == coordinate_vals[node->count() - 1]) {
        split_val = 0.5 * (coordinate_vals[0] +
                           coordinate_vals[node->count() - 1]);
      }
      return split_val;
    }
};

class GenKdTree {
  public:

    typedef core::tree::HrectBound BoundType;

    void FindBoundFromMatrix(
      const core::table::DenseMatrix &matrix,
      int first, int count, BoundType *bounds) {

      int end = first + count;
      for(int i = first; i < end; i++) {
        GenVector<T> col;

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

      for(int left = first; left < end; left++) {

        // Make alias of the current point.
        core::table::DenseConstPoint point;
        matrix.MakeColumnVector(left, &point);

        // We swap if the point is further away from the left pivot.
        if(point[split_dim] > split_val) {
          left_membership[left - first] = false;
        }
        else {
          left_membership[left - first] = true;
          left_count++;
        }
      }
    }

    template<typename TreeType>
    static bool AttemptSplitting(
      const core::metric_kernels::AbstractMetric &metric_in,
      core::table::DenseMatrix& matrix, TreeType *node, TreeType **left,
      TreeType **right, int leaf_size,
      std::vector<int> *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      TreeType *left = NULL;
      TreeType *right = NULL;

      if(node->count() > leaf_size) {
        index_t split_dim = BIG_BAD_NUMBER;
        T max_width = -1;

        for(index_t d = 0; d < matrix.n_rows(); d++) {
          T w = node->bound().get(d).width();

          if(unlikely(w > max_width)) {
            max_width = w;
            split_dim = d;
          }
        }

        // choose the split value along the dimension to be splitted
        double split_val =
          core::tree::GenKdTreeMidpointSplitter::ChooseKdTreeSplitValue(
            matrix, node, split_dim);

        if(max_width < std::numeric_limits<double>::epsilon()) {
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
          (*right)->Init(
            node->begin() + left_count, node->count() - left_count);
        }

        return true;
      }
    };
};
};

#endif
