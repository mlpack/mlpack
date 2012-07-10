 /**
 * @file dtree_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * Implementations of some declared functions in
 * the Density Estimation Tree class.
 *
 */

#ifndef __MLPACK_METHODS_DET_DTREE_IMPL_HPP
#define __MLPACK_METHODS_DET_DTREE_IMPL_HPP

#include "dtree.hpp"

namespace mlpack{
namespace det {

// This function computes the log-l2-negative-error of a given node from the
// formula R(t) = log(|t|^2 / (N^2 V_t)).
template<typename eT, typename cT>
double DTree<eT, cT>::LogNegativeError(size_t total_points)
{
  // log(-|t|^2 / (N^2 V_t)) = log(-1) + 2 log(|t|) - 2 log(N) - log(V_t).
  return 2 * std::log((double) (end_ - start_)) -
         2 * std::log((double) total_points) -
         arma::accu(arma::log((*max_vals_) - (*min_vals_)));
}

// This function finds the best split with respect to the L2-error,
// but trying all possible splits.  The dataset is the full data set but the
// start_ and end_ are used to obtain the point in this node.
template<typename eT, typename cT>
bool DTree<eT, cT>::FindSplit_(MatType* data,
                               size_t *split_dim,
                               size_t *split_ind,
                               cT *left_error,
                               cT *right_error,
                               size_t maxLeafSize,
                               size_t minLeafSize)
{
  assert(data->n_rows == max_vals_->n_elem);
  assert(data->n_rows == min_vals_->n_elem);

  size_t total_n = data->n_cols, n_t = end_ - start_;

  cT min_error = error_;
  bool some_split_found = false;
  size_t point_mass_in_dim = 0;

  // Loop through each dimension.
  for (size_t dim = 0; dim < max_vals_->n_elem; dim++)
  {
    // Have to deal with REAL, INTEGER, NOMINAL data differently, so we have to
    // think of how to do that...
    eT min = (*min_vals_)[dim], max = (*max_vals_)[dim];

    // Check if there is any scope of splitting in this dimension.
    if (max - min > 0.0) {
      // Initializing all the stuff for this dimension.
      bool dim_split_found = false;
      cT min_dim_error = min_error, temp_lval = 0.0, temp_rval = 0.0;
      size_t dim_split_ind = -1;

      cT log_range_all_not_dim = 0;
      for (size_t i = 0; i < max_vals_->n_elem; i++)
      {
        if ((*max_vals_)[i] -(*min_vals_)[i] > 0.0 && i != dim)
        {
          log_range_all_not_dim +=
              (cT) std::log((*max_vals_)[i] - (*min_vals_)[i]);
        }
      }

      assert(std::exp(log_range_all_not_dim) > 0);

      // Get the values for the dimension.
      RowVecType dim_val_vec = data->row(dim).subvec(start_, end_ - 1);

      // Sort the values in ascending order.
      dim_val_vec = arma::sort(dim_val_vec);

      // Get ready to go through the sorted list and compute error.
      assert(dim_val_vec.n_elem > maxLeafSize);

      // Enforce that the leaves have a minimum number of points to avoid
      // spikes.  One way of doing this is to only consider splits resulting in
      // sizes > some constant (minLeafSize).
      size_t left_child_size = minLeafSize - 1, right_child_size;

      // Find the best split for this dimension.  We need to figure out why
      // there are spikes if this min_leaf_size is enforced here...
      for (size_t i = minLeafSize -1; i < dim_val_vec.n_elem - minLeafSize;
          ++i, ++left_child_size)
      {
        eT split, lsplit = dim_val_vec[i], rsplit = dim_val_vec[i + 1];

        if (lsplit < rsplit)
        {
          // This makes sense for real continuous data.  This kinda corrupts the
          // data and estimation if the data is ordinal
          split = (lsplit + rsplit) / 2;

          // Another way of picking split is using this:
          //   split = left_split;

          if (split - min > 0.0 && max - split > 0.0)
          {
            assert(std::exp(log_range_all_not_dim +
                (cT) std::log(split - min)) > 0);
            assert(std::exp(log_range_all_not_dim +
                (cT) std::log(max - split)) > 0);

            cT temp_log_neg_l_error = 2 * std::log((cT) (i + 1) / (cT) total_n)
                - (log_range_all_not_dim + (cT) std::log(split - min));

            assert(std::exp(temp_log_neg_l_error) > 0.0);

            cT temp_l_error = -1.0 * std::exp(temp_log_neg_l_error);

            assert(std::abs(temp_l_error) < std::numeric_limits<cT>::max());

            cT temp_log_neg_r_error = 2 * std::log((cT) (n_t - i - 1) /
                (cT) total_n) - (log_range_all_not_dim +
                (cT) std::log(max - split));

            assert(std::exp(temp_log_neg_r_error) > 0.0);

            right_child_size = n_t - i - 1;
            assert(right_child_size >= minLeafSize);

            cT temp_r_error = -1.0 * std::exp(temp_log_neg_r_error);

            assert(std::abs(temp_r_error) < std::numeric_limits<cT>::max());

            //if (temp_l + temp_r <= min_dim_error) {
            // Why not just less than?
            if (temp_l_error + temp_r_error < min_dim_error)
            {
              min_dim_error = temp_l_error + temp_r_error;
              temp_lval = temp_l_error;
              temp_rval = temp_r_error;
              dim_split_ind = i;
              dim_split_found = true;
            } // end if improvement.
          } // end if split - min > 0 & max - split > 0.
        } // end if lsplit < rsplit instead of being equal.
      } // end for loop over all splits in this dimension.

      dim_val_vec.clear();

      if ((min_dim_error < min_error) && dim_split_found)
      {
        min_error = min_dim_error;
        *split_dim = dim;
        *split_ind = dim_split_ind;
        *left_error = temp_lval;
        *right_error = temp_rval;
        some_split_found = true;
      } // end if better split found in this dimension.
    }
    else
    {
      point_mass_in_dim++;
    }
  }

  return some_split_found;
} // end FindSplit_


template<typename eT, typename cT>
void DTree<eT, cT>::SplitData_(MatType* data,
                               size_t split_dim,
                               size_t split_ind,
                               arma::Col<size_t> *old_from_new,
                               eT *split_val,
                               eT *lsplit_val,
                               eT *rsplit_val)
{
  // Get the values for the split dimension.
  RowVecType dim_val_vec = data->row(split_dim).subvec(start_, end_ - 1);

  // Sort the values.
  dim_val_vec = arma::sort(dim_val_vec);

  *lsplit_val = dim_val_vec[split_ind];
  *rsplit_val = dim_val_vec[split_ind + 1];
  *split_val = (*lsplit_val + *rsplit_val) / 2 ;

  std::vector<bool> left_membership;
  left_membership.reserve(end_ - start_);

  for (size_t i = start_; i < end_; ++i)
  {
    if ((*data)(split_dim, i) > *split_val)
      left_membership[i - start_] = false;
    else
      left_membership[i - start_] = true;
  }

  size_t left_ind = start_, right_ind = end_ - 1;
  for (;;)
  {
    while (left_membership[left_ind - start_] && (left_ind <= right_ind))
      left_ind++;

    while (!left_membership[right_ind - start_] && (left_ind <= right_ind))
      right_ind--;

    if (left_ind > right_ind)
      break;


    data->swap_cols(left_ind, right_ind);
    bool tmp = left_membership[left_ind - start_];
    left_membership[left_ind - start_] = left_membership[right_ind - start_];
    left_membership[right_ind - start_] = tmp;

    size_t t = (*old_from_new)[left_ind];
    (*old_from_new)[left_ind] = (*old_from_new)[right_ind];
    (*old_from_new)[right_ind] = t;
  }

  assert(left_ind == right_ind + 1);
}


template<typename eT, typename cT>
void DTree<eT, cT>::GetMaxMinVals_(MatType* data,
                                   VecType *max_vals,
                                   VecType *min_vals)
{
  max_vals->set_size(data->n_rows);
  min_vals->set_size(data->n_rows);

  MatType temp_d = arma::trans(*data);

  for (size_t i = 0; i < temp_d.n_cols; ++i)
  {
    VecType dim_vals = arma::sort(temp_d.col(i));
    (*min_vals)[i] = dim_vals[0];
    (*max_vals)[i] = dim_vals[dim_vals.n_elem - 1];
  }
}


template<typename eT, typename cT>
DTree<eT, cT>::DTree() :
    start_(0),
    end_(0),
    max_vals_(NULL),
    min_vals_(NULL),
    left_(NULL),
    right_(NULL)
{ /* Nothing to do. */ }


// Root node initializers
template<typename eT, typename cT>
DTree<eT, cT>::DTree(VecType* max_vals,
                     VecType* min_vals,
                     size_t total_points) :
    start_(0),
    end_(total_points),
    max_vals_(max_vals),
    min_vals_(min_vals),
    left_(NULL),
    right_(NULL)
{
  error_ = -std::exp(LogNegativeError(total_points));

  bucket_tag_ = -1;
  root_ = true;
}


template<typename eT, typename cT>
DTree<eT, cT>::DTree(MatType* data) :
    start_(0),
    end_(data->n_cols),
    left_(NULL),
    right_(NULL)
{
  max_vals_ = new VecType();
  min_vals_ = new VecType();

  GetMaxMinVals_(data, max_vals_, min_vals_);

  error_ = -std::exp(LogNegativeError(data->n_cols));

  bucket_tag_ = -1;
  root_ = true;
}


// Non-root node initializers
template<typename eT, typename cT>
DTree<eT, cT>::DTree(VecType* max_vals,
                     VecType* min_vals,
                     size_t start,
                     size_t end,
                     cT error) :
    start_(start),
    end_(end),
    error_(error),
    max_vals_(max_vals),
    min_vals_(min_vals),
    left_(NULL),
    right_(NULL)
{
  bucket_tag_ = -1;
  root_ = false;
}


template<typename eT, typename cT>
DTree<eT, cT>::DTree(VecType* max_vals,
                     VecType* min_vals,
                     size_t total_points,
                     size_t start,
                     size_t end) :
    start_(start),
    end_(end),
    max_vals_(max_vals),
    min_vals_(min_vals),
    left_(NULL),
    right_(NULL)
{
  error_ = -std::exp(LogNegativeError(total_points));

  bucket_tag_ = -1;
  root_ = false;
}


template<typename eT, typename cT>
DTree<eT, cT>::~DTree()
{
  if (left_ != NULL)
    delete left_;

  if (right_ != NULL)
    delete right_;

  if (min_vals_ != NULL)
    delete min_vals_;

  if (max_vals_ != NULL)
    delete max_vals_;
}


// Greedily expand the tree
template<typename eT, typename cT>
cT DTree<eT, cT>::Grow(MatType* data,
                       arma::Col<size_t> *old_from_new,
                       bool useVolReg,
                       size_t maxLeafSize,
                       size_t minLeafSize)
{
  assert(data->n_rows == max_vals_->n_elem);
  assert(data->n_rows == min_vals_->n_elem);

  cT left_g, right_g;

  // Compute points ratio.
  ratio_ = (cT) (end_ - start_) / (cT) old_from_new->n_elem;

  // Compute the v_t_inv: the inverse of the volume of the node.
  cT log_vol_t = 0;
  for (size_t i = 0; i < max_vals_->n_elem; ++i)
    if ((*max_vals_)[i] - (*min_vals_)[i] > 0.0)
      // Use log to prevent overflow.
      log_vol_t += (cT) std::log((*max_vals_)[i] - (*min_vals_)[i]);

  // Check for overflow.
  assert(std::exp(log_vol_t) > 0.0);
  v_t_inv_ = 1.0 / std::exp(log_vol_t);

  // Check if node is large enough.
  if ((size_t) (end_ - start_) > maxLeafSize) {

    // Find the split.
    size_t dim, split_ind;
    cT left_error, right_error;
    if (FindSplit_(data, &dim, &split_ind, &left_error, &right_error,
        maxLeafSize, minLeafSize))
    {
      // Move the data around for the children to have points in a node lie
      // contiguously (to increase efficiency during the training).
      eT split_val, lsplit_val, rsplit_val;
      SplitData_(data, dim, split_ind, old_from_new, &split_val, &lsplit_val,
          &rsplit_val);

      // Make max and min vals for the children.
      VecType* max_vals_l = new VecType(*max_vals_);
      VecType* max_vals_r = new VecType(*max_vals_);
      VecType* min_vals_l = new VecType(*min_vals_);
      VecType* min_vals_r = new VecType(*min_vals_);

      (*max_vals_l)[dim] = split_val;
      (*min_vals_r)[dim] = split_val;

      // Store split dim and split val in the node.
      split_value_ = split_val;
      split_dim_ = dim;

      // Recursively grow the children.
      left_ = new DTree(max_vals_l, min_vals_l, start_, start_ + split_ind + 1,
          left_error);
      right_ = new DTree(max_vals_r, min_vals_r, start_ + split_ind + 1, end_,
          right_error);

      left_g = left_->Grow(data, old_from_new, useVolReg, maxLeafSize,
          minLeafSize);
      right_g = right_->Grow(data, old_from_new, useVolReg, maxLeafSize,
          minLeafSize);

      // Store values of R(T~) and |T~|.
      subtree_leaves_ = left_->subtree_leaves() + right_->subtree_leaves();
      subtree_leaves_error_ = left_->subtree_leaves_error() +
          right_->subtree_leaves_error();

      // Store the subtree_leaves_v_t_inv.
      subtree_leaves_v_t_inv_ = left_->subtree_leaves_v_t_inv() +
          right_->subtree_leaves_v_t_inv();

      // Form T1 by removing leaves for which R(t) = R(t_L) + R(t_R).
      if ((left_->subtree_leaves() == 1) && (right_->subtree_leaves() == 1))
      {
        if (left_->error() + right_->error() == error_)
        {
          delete left_;
          left_ = NULL;

          delete right_;
          right_ = NULL;

          subtree_leaves_ = 1;
          subtree_leaves_error_ = error_;
          subtree_leaves_v_t_inv_ = v_t_inv_;
        }
      }
    }
    else
    {
      // No split found so make a leaf out of it.
      subtree_leaves_ = 1;
      subtree_leaves_error_ = error_;
      subtree_leaves_v_t_inv_ = v_t_inv_;
    }
  }
  else
  {
    // We can make this a leaf node.
    assert((size_t) (end_ - start_) >= minLeafSize);
    subtree_leaves_ = 1;
    subtree_leaves_error_ = error_;
    subtree_leaves_v_t_inv_ = v_t_inv_;
  }

  // If this is a leaf, do not compute g_k(t); otherwise compute, store, and
  // propagate min(g_k(t_L),g_k(t_R),g_k(t)), unless t_L and/or t_R are leaves.
  if (subtree_leaves_ == 1)
  {
    return std::numeric_limits<cT>::max();
  }
  else
  {
    cT g_t;
    if (useVolReg)
      g_t = (error_ - subtree_leaves_error_) /
          (subtree_leaves_v_t_inv_ - v_t_inv_);
    else
      g_t = (error_ - subtree_leaves_error_) / (subtree_leaves_ - 1);

    assert(g_t > 0.0);
    return min(g_t, min(left_g, right_g));
  }

  // We need to compute (c_t^2)*r_t for all subtree leaves; this is equal to
  // n_t ^ 2 / r_t * n ^ 2 = -error_.  Therefore the value we need is actually
  // -1.0 * subtree_leaves_error_.
}


template<typename eT, typename cT>
cT DTree<eT, cT>::PruneAndUpdate(cT old_alpha, bool useVolReg)
{
  // Compute g_t.
  if (subtree_leaves_ == 1) // If we are a leaf...
  {
    return std::numeric_limits<cT>::max();
  }
  else
  {
    // Compute g_t value for node t.
    cT g_t;
    if (useVolReg)
      g_t = (error_ - subtree_leaves_error_) /
          (subtree_leaves_v_t_inv_ - v_t_inv_);
    else
      g_t = (error_ - subtree_leaves_error_) / (subtree_leaves_ - 1);

    if (g_t > old_alpha)
    {
      // Go down the tree and update accordingly.  Traverse the children.
      cT left_g = left_->PruneAndUpdate(old_alpha, useVolReg);
      cT right_g = right_->PruneAndUpdate(old_alpha, useVolReg);

      // Update values.
      subtree_leaves_ = left_->subtree_leaves() + right_->subtree_leaves();
      subtree_leaves_error_ = left_->subtree_leaves_error() +
          right_->subtree_leaves_error();
      subtree_leaves_v_t_inv_ = left_->subtree_leaves_v_t_inv() + right_->subtree_leaves_v_t_inv();

      // Update g_t value.
      if (useVolReg)
        g_t = (error_ - subtree_leaves_error_)
            / (subtree_leaves_v_t_inv_ - v_t_inv_);
      else
        g_t = (error_ - subtree_leaves_error_) / (subtree_leaves_ - 1);

      assert(g_t < std::numeric_limits<cT>::max());

      if (left_->subtree_leaves() == 1 && right_->subtree_leaves() == 1)
        return g_t;
      else if (left_->subtree_leaves() == 1)
        return min(g_t, right_g);
      else if (right_->subtree_leaves() == 1)
        return min(g_t, left_g);
      else
        return min(g_t, min(left_g, right_g));

    }
    else
    {
      // Prune this subtree.
      // First, make this node a leaf node.
      subtree_leaves_ = 1;
      subtree_leaves_error_ = error_;
      subtree_leaves_v_t_inv_ = v_t_inv_;

      delete left_;
      left_ = NULL;
      delete right_;
      right_ = NULL;

      // Pass information upward.
      return std::numeric_limits<cT>::max();

    }
  }
}

// Check whether a given point is within the bounding box of this node (check
// generally done at the root, so its the bounding box of the data).
//
// Future improvement: Open up the range with epsilons on both sides where
// epsilon depends on the density near the boundary.
template<typename eT, typename cT>
bool DTree<eT, cT>::WithinRange_(VecType* query)
{
  for (size_t i = 0; i < query->n_elem; ++i)
    if (((*query)[i] < (*min_vals_)[i]) || ((*query)[i] > (*max_vals_)[i]))
      return false;

  return true;
}


template<typename eT, typename cT>
cT DTree<eT, cT>::ComputeValue(VecType* query)
{
  assert(query->n_elem == max_vals_->n_elem);

  if (root_ == 1) // If we are the root...
    // Check if the query is within range.
    if (!WithinRange_(query))
      return 0.0;

  if (subtree_leaves_ == 1)  // If we are a leaf...
    return ratio_ * v_t_inv_;
  else
  {
    if ((*query)[split_dim_] <= split_value_)
      // If left subtree, go to left child.
      return left_->ComputeValue(query);
    else  // If right subtree, go to right child
      return right_->ComputeValue(query);
  }
}


template<typename eT, typename cT>
void DTree<eT, cT>::WriteTree(size_t level, FILE *fp)
{
  if (subtree_leaves_ > 1)
  {
    fprintf(fp, "\n");
    for (size_t i = 0; i < level; ++i)
      fprintf(fp, "|\t");
    fprintf(fp, "Var. %zu > %lg", split_dim_, split_value_);

    right_->WriteTree(level + 1, fp);

    fprintf(fp, "\n");
    for (size_t i = 0; i < level; ++i)
      fprintf(fp, "|\t");
    fprintf(fp, "Var. %zu <= %lg ", split_dim_, split_value_);

    left_->WriteTree(level + 1, fp);
  }
  else // If we are a leaf...
  {
    fprintf(fp, ": f(x)=%Lg", (cT) ratio_ * v_t_inv_);
    if (bucket_tag_ != -1)
      fprintf(fp, " BT:%d", bucket_tag_);
  }
}


// Index the buckets for possible usage later.
template<typename eT, typename cT>
int DTree<eT, cT>::TagTree(int tag)
{
  if (subtree_leaves_ == 1)
  {
    bucket_tag_ = tag;
    return (tag + 1);
  }
  else
  {
    return right_->TagTree(left_->TagTree(tag));
  }
} // TagTree


template<typename eT, typename cT>
int DTree<eT, cT>::FindBucket(VecType* query)
{
  assert(query->n_elem == max_vals_->n_elem);

  if (subtree_leaves_ == 1) // If we are a leaf...
  {
    return bucket_tag_;
  }
  else if ((*query)[split_dim_] <= split_value_)
  {
    // If left subtree, go to left child.
    return left_->FindBucket(query);
  }
  else // If right subtree, go to right child.
  {
    return right_->FindBucket(query);
  }
}


template<typename eT, typename cT>
void DTree<eT, cT>::ComputeVariableImportance(arma::Col<double> *imps)
{
  if (subtree_leaves_ == 1)
  {
    // If we are a leaf, do nothing.
    return;
  }
  else
  {
    // Compute the improvement in error because of the split.
    double error_improv = (double)
        (error_ - (left_->error() + right_->error()));
    (*imps)[split_dim_] += error_improv;
    left_->ComputeVariableImportance(imps);
    right_->ComputeVariableImportance(imps);
    return;
  }
}

}; // namespace det
}; // namespace mlpack

#endif // __MLPACK_METHODS_DET_DTREE_IMPL_HPP
