#ifndef QUICK_PRUNE_LPR_H
#define QUICK_PRUNE_LPR_H

#include "fastlib/fastlib.h"

class QuickPruneLpr {

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

      // Refine the lower bound norm using the new lower bound info
      // for the denominator B^T W(q) B.
      la::AddOverwrite(qnode->stat().postponed_denominator_l_, denominator_dl,
		       &tmp_denominator_dl);
           
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
      return (kernel_error <= relative_error *
	      (qnode->stat().kernel_sum_l_ + tmp_denominator_dl.get(0, 0)));
    }
      
};

#endif
