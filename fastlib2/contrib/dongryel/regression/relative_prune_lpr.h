#ifndef RELATIVE_PRUNE_LPR_H
#define RELATIVE_PRUNE_LPR_H

#include "fastlib/fastlib.h"

class RelativePruneLpr {
  
  public:
  
    template<typename QueryTree, typename ReferenceTree>
    static bool Prunable(double relative_error,
			 double numerator_total_alloc_error,
			 double denominator_total_alloc_error,
			 QueryTree *qnode, ReferenceTree *rnode,
			 const DRange &dsqd_range, 
			 const DRange &kernel_value_range,
			 Vector &numerator_dl, Vector &numerator_de, 
			 double &numerator_used_error,
			 double &numerator_n_pruned,
			 Matrix &denominator_dl, Matrix &denominator_de, 
			 double &denominator_used_error, 
			 double &denominator_n_pruned) {
      
      Vector tmp_numerator_dl;
      tmp_numerator_dl.Init(numerator_dl.length());
      Matrix tmp_denominator_dl;
      tmp_denominator_dl.Init(denominator_dl.n_rows(), 
			      denominator_dl.n_cols());
      
      // The lower bound contribution and the finite difference
      // estimate for the numerator.
      la::ScaleOverwrite(kernel_value_range.lo,
			 rnode->stat().sum_target_weighted_data_,
			 &numerator_dl);
      la::ScaleOverwrite(kernel_value_range.mid(),
			 rnode->stat().sum_target_weighted_data_,
			 &numerator_de);
      
      // The upper bound contribution and the finite difference
      // estimate for the denominator.
      la::ScaleOverwrite(kernel_value_range.lo,
			 rnode->stat().sum_data_outer_products_,
			 &denominator_dl);
      la::ScaleOverwrite(kernel_value_range.mid(),
			 rnode->stat().sum_data_outer_products_,
			 &denominator_de);
            
      // Refine the lower bound norm using the new lower bound info
      // for the numerator B^T W(q) Y.
      la::AddOverwrite(qnode->stat().postponed_numerator_l_, numerator_dl,
		       &tmp_numerator_dl);
      double new_numerator_norm_l = qnode->stat().numerator_norm_l_ + 
	MatrixUtil::EntrywiseLpNorm(tmp_numerator_dl, 2);
      double numerator_allowed_err = 
	(relative_error * new_numerator_norm_l - 
	 qnode->stat().numerator_used_error_) /
	(numerator_total_alloc_error - qnode->stat().numerator_n_pruned_);

      // Refine the lower bound norm using the new lower bound info
      // for the denominator B^T W(q) B.
      la::AddOverwrite(qnode->stat().postponed_denominator_l_, denominator_dl,
		       &tmp_denominator_dl);
      double new_denominator_norm_l = qnode->stat().denominator_norm_l_ + 
	MatrixUtil::EntrywiseLpNorm(tmp_denominator_dl, 2);
      double denominator_allowed_err = 
	(relative_error * new_denominator_norm_l - 
	 qnode->stat().denominator_used_error_) /
	(denominator_total_alloc_error - qnode->stat().denominator_n_pruned_);
           
      // this is error per each query/reference pair for a fixed query
      double kernel_error = 0.5 * kernel_value_range.width();
      
      // This is total norm error for each query point for
      // approximating the B^T W(q) Y vector.
      numerator_used_error = kernel_error * 
	(rnode->stat().sum_target_weighted_data_error_norm_);
      numerator_n_pruned = rnode->stat().sum_target_weighted_data_alloc_norm_;
      
      // The total norm error for each query point for approximating
      // the B^T W(q) B matrix.
      denominator_used_error = kernel_error *
	(rnode->stat().sum_data_outer_products_error_norm_);
      denominator_n_pruned = rnode->stat().sum_data_outer_products_alloc_norm_;
      
      // Check pruning condition.
      return (numerator_used_error <= numerator_allowed_err &&
	      denominator_used_error <= denominator_allowed_err);
    }
};

#endif
