/* Implementation for the regular pointer-style spill kd-tree builder. */

#include <mlpack/core.h>

#include "mlpack/core/tree/hrectbound.h"

#define BIG_BAD_NUMBER (size_t) (-1)

namespace tree_gen_kdtree_private {

  template<typename T, typename TBound>
  void FindBoundFromMatrix(const arma::Mat<T>& matrix,
			   size_t first, size_t count, TBound *bounds) {

    size_t end = first + count;
    for (size_t i = first; i < end; i++)
    {
      arma::vec col = arma::vec (matrix.n_rows);
      for (size_t d = 0; d < matrix.n_rows; ++d)
      {
        col(d) = (matrix (d, i));
      }
      *bounds |= col;
    }
  }

  template<typename T, typename TBound>
  size_t MatrixPartition(arma::Mat<T>& matrix, size_t dim, double splitvalue,
			  size_t first, size_t count, TBound* left_bound,
			  TBound* right_bound, size_t *old_from_new) {

    size_t left = first;
    size_t right = first + count - 1;

    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (matrix(dim, left) < splitvalue && left <= right)
      {
        arma::vec left_vector;
        left_vector = matrix.col(left);
        *left_bound |= left_vector;
        left++;
      }

      while (matrix(dim, right) >= splitvalue && left <= right)
      {
        arma::vec right_vector;
        right_vector = matrix.col(right);
        *right_bound |= right_vector;
        right--;
      }

      if (left > right) {
        /* left == right + 1 */
        break;
      }

      arma::vec left_vector = matrix.col(left);
      arma::vec right_vector = matrix.col(right);

      // TODO TODO TODO: determine what this is doing... left_vector.SwapValues(&right_vector);

      *left_bound |= left_vector;
      *right_bound |= right_vector;

      if (old_from_new) {
        size_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }

      Log::Assert(left <= right, "Left is greater than right!");
      right--;
    }

    Log::Assert(left == right + 1, "Left is NOT equal to right plus one!");

    return left;
  }

  template<typename T, typename TKdTree, typename TKdTreeSplitter>
  void SplitGenKdTree(arma::Mat<T>& matrix, TKdTree *node,
		      size_t leaf_size, size_t *old_from_new)
  {

    TKdTree *left = NULL;
    TKdTree *right = NULL;

    if (node->count() > leaf_size)
    {
      size_t split_dim = BIG_BAD_NUMBER;
      T max_width = -1;

      for (size_t d = 0; d < matrix.n_rows; d++)
      {
        T w = node->bound()[d].width();

        if (w > max_width)
        {
          max_width = w;
          split_dim = d;
        }
      }

      // choose the split value along the dimension to be splitted
      double split_val =
	TKdTreeSplitter::ChooseKdTreeSplitValue(matrix, node, split_dim);

      if (max_width < DBL_EPSILON)
      {
        // Okay, we can't do any splitting, because all these points are the
        // same.  We have to give up.
      }
      else
      {
        mlpack::bound::HRectBound<2> lBound = mlpack::bound::HRectBound<2>(matrix.n_rows);
        mlpack::bound::HRectBound<2> rBound = mlpack::bound::HRectBound<2>(matrix.n_rows);

        left = new TKdTree();
        left->setBound(lBound);

        right = new TKdTree();
        right->setBound (rBound);

        size_t split_col = MatrixPartition(matrix, split_dim, split_val,
					    node->begin(), node->count(),
					    &left->bound(), &right->bound(),
					    old_from_new);

  Log::Info << 3.0 << "split (" << node->begin() << ",[" << split_col << "]," <<
        node->begin() + node->count() << ") dim " << split_dim << " on " <<
        split_val << " (between " << node->bound()[split_dim].lo << ", " <<
        node->bound()[split_dim].hi << ")" << std::endl;

	left->Init(node->begin(), split_col - node->begin());
	right->Init(split_col, node->begin() + node->count() - split_col);

	SplitGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, left, leaf_size, old_from_new);
	SplitGenKdTree<T, TKdTree, TKdTreeSplitter>
	  (matrix, right, leaf_size, old_from_new);
      }
    }

    node->set_children(matrix, left, right);
  }
};
