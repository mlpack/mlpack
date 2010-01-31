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
			 double &denominator_n_pruned,
			 Matrix &weight_diagram_numerator_dl,
			 Matrix &weight_diagram_numerator_de,
			 double &weight_diagram_numerator_used_error) {
      
      Vector tmp_numerator_dl;
      tmp_numerator_dl.Init(numerator_dl.length());
      Matrix tmp_denominator_dl, tmp_weight_diagram_numerator_dl;
      tmp_denominator_dl.Init(denominator_dl.n_rows(), 
			      denominator_dl.n_cols());
      tmp_weight_diagram_numerator_dl.Init
	(weight_diagram_numerator_dl.n_rows(),
	 weight_diagram_numerator_dl.n_cols());
      
      // Form the squared kernel value range based on the kernel value
      // bound.
      DRange squared_kernel_value_range;
      squared_kernel_value_range.lo = kernel_value_range.lo *
	kernel_value_range.lo;
      squared_kernel_value_range.hi = kernel_value_range.hi *
	kernel_value_range.hi;

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
      
      // The finite difference estimate for the weight diagram matrix.
      la::ScaleOverwrite(squared_kernel_value_range.lo,
			 rnode->stat().sum_data_outer_products_,
			 &weight_diagram_numerator_dl);
      la::ScaleOverwrite(squared_kernel_value_range.mid(),
			 rnode->stat().sum_data_outer_products_,
			 &weight_diagram_numerator_de);

      // Refine the lower bound norm using the new lower bound info
      // for the numerator B^T W(q) Y.
      la::AddOverwrite(qnode->stat().postponed_numerator_l_, numerator_dl,
		       &tmp_numerator_dl);
      double new_numerator_norm_l = qnode->stat().numerator_norm_l_ + 
	MatrixUtil::EntrywiseLpNorm(tmp_numerator_dl, 1);
      double new_numerator_used_error = qnode->stat().numerator_used_error_ +
	qnode->stat().postponed_numerator_used_error_;
      double new_numerator_n_pruned = qnode->stat().numerator_n_pruned_ +
	qnode->stat().postponed_numerator_n_pruned_;

      double numerator_allowed_err = 
	(relative_error * new_numerator_norm_l - new_numerator_used_error) /
	(numerator_total_alloc_error - new_numerator_n_pruned);

      // Refine the lower bound norm using the new lower bound info
      // for the denominator B^T W(q) B.
      la::AddOverwrite(qnode->stat().postponed_denominator_l_, denominator_dl,
		       &tmp_denominator_dl);
      double new_denominator_norm_l = qnode->stat().denominator_norm_l_ + 
	MatrixUtil::EntrywiseLpNorm(tmp_denominator_dl, 1);
      double new_denominator_used_error = 
	qnode->stat().denominator_used_error_ +
	qnode->stat().postponed_denominator_used_error_;
      double new_denominator_n_pruned =	qnode->stat().denominator_n_pruned_ +
	qnode->stat().postponed_denominator_n_pruned_;

      double denominator_allowed_err = 
	(relative_error * new_denominator_norm_l - 
	 new_denominator_used_error) /
	(denominator_total_alloc_error - new_denominator_n_pruned);
        
      // Refine the bound using the new info for the weight diagram
      // numerator matrix.
      la::AddOverwrite(qnode->stat().postponed_weight_diagram_numerator_l_, 
		       weight_diagram_numerator_dl,
		       &tmp_weight_diagram_numerator_dl);
      double new_weight_diagram_numerator_norm_l = 
	qnode->stat().weight_diagram_numerator_norm_l_ + 
	MatrixUtil::EntrywiseLpNorm(tmp_weight_diagram_numerator_dl, 1);
      double new_weight_diagram_numerator_used_error =
	qnode->stat().weight_diagram_numerator_used_error_ +
	qnode->stat().postponed_weight_diagram_numerator_used_error_;

      double weight_diagram_numerator_allowed_err = 
	(relative_error * new_weight_diagram_numerator_norm_l - 
	 new_weight_diagram_numerator_used_error) /
	(denominator_total_alloc_error - new_denominator_n_pruned);

      // this is error per each query/reference pair for a fixed query
      // for the numerator and the denominator used for computing the
      // regression estimates.
      double kernel_error = 0.5 * kernel_value_range.width();
      double squared_kernel_error = 0.5 * squared_kernel_value_range.width();

      // This is the error per each query/reference pair for a fixed
      // query for the weight diagram numerator matrix approximation.
      //double squared_kernel_error = 0.5 * squared_kernel_value_range.width();

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

      // The total norm error for each query point for approximating
      // the B^T W(q)^2 B matrix.
      weight_diagram_numerator_used_error = squared_kernel_error *
	(rnode->stat().sum_data_outer_products_error_norm_);

      // Check pruning condition. Note that this pruning criterion
      // does not enforce error directly on the weight diagram
      // computation.
      return (numerator_used_error <= numerator_allowed_err &&
	      denominator_used_error <= denominator_allowed_err &&
	      weight_diagram_numerator_used_error <= 
	      weight_diagram_numerator_allowed_err);
    }

    template<typename QueryTree, typename ReferenceTree>
    static bool PrunableKrylovRightHandSides
    (double relative_error, double total_alloc_error, QueryTree *qnode, 
     ReferenceTree *rnode, const DRange &dsqd_range, 
     const DRange &kernel_value_range, Vector &delta_l, Vector &delta_e, 
     double &delta_used_error, double &delta_n_pruned) {
      
      // Compute the vector component lower and upper bound changes. This
      // assumes that the maximum kernel value is 1.
      la::ScaleOverwrite(kernel_value_range.lo,
			 rnode->stat().sum_target_weighted_data_, &delta_l);
      la::ScaleOverwrite(kernel_value_range.mid(),
			 rnode->stat().sum_target_weighted_data_, &delta_e);
      
      // Compute the L1 norm of the most refined lower bound.
      double new_ll_vector_norm_l = 
	qnode->stat().ll_vector_norm_l_ +
	MatrixUtil::EntrywiseLpNorm(qnode->stat().postponed_ll_vector_l_, 1) +
	MatrixUtil::EntrywiseLpNorm(delta_l, 1);
      double new_ll_vector_used_error = qnode->stat().ll_vector_used_error_ +
	qnode->stat().postponed_ll_vector_used_error_;
      double new_ll_vector_n_pruned = qnode->stat().ll_vector_n_pruned_ +
	qnode->stat().postponed_ll_vector_n_pruned_;

      // Compute the allowed amount of error for pruning the given query
      // and reference pair.
      double allowed_err = 
	(relative_error * new_ll_vector_norm_l - new_ll_vector_used_error) /
	(total_alloc_error - new_ll_vector_n_pruned);

      // Record how much error and pruned portion will be if pruning
      // were to succeed.
      delta_used_error = 0.5 * kernel_value_range.width() * 
	(rnode->stat().sum_target_weighted_data_error_norm_);
      delta_n_pruned = 
	rnode->stat().sum_target_weighted_data_alloc_norm_;
      
      // check pruning condition  
      return (delta_used_error <= allowed_err);
    }
  
    template<typename QueryTree, typename ReferenceTree>
    static bool PrunableKrylovSolver
    (double relative_error, double total_alloc_error, QueryTree *qnode, 
     ReferenceTree *rnode, const DRange &dsqd_range, 
     const DRange &kernel_value_range, 
     const DRange &negative_dot_product_range,
     const DRange &positive_dot_product_range,
     Vector &delta_l, Vector &delta_e, double &delta_used_error, 
     double &delta_n_pruned, Vector &delta_neg_u, Vector &delta_neg_e,
     double &delta_neg_used_error, double &delta_neg_n_pruned) {

      // Compute the vector component lower and upper bound changes. This
      // assumes that the maximum kernel value is 1.
      la::ScaleOverwrite(positive_dot_product_range.lo * kernel_value_range.lo,
			 rnode->stat().sum_reference_point_expansion_,
			 &delta_l);
      la::ScaleOverwrite(0.5 * (positive_dot_product_range.lo *
				kernel_value_range.lo +
				positive_dot_product_range.hi *
				kernel_value_range.hi),
			 rnode->stat().sum_reference_point_expansion_,
			 &delta_e);
      
      la::ScaleOverwrite(0.5 * (negative_dot_product_range.lo *
				kernel_value_range.hi +
				negative_dot_product_range.hi *
				kernel_value_range.lo),
			 rnode->stat().sum_reference_point_expansion_, 
			 &delta_neg_e);
      la::ScaleOverwrite(negative_dot_product_range.hi * kernel_value_range.lo,
			 rnode->stat().sum_reference_point_expansion_,
			 &delta_neg_u);

      // Compute the L1 norm of the most refined lower bound.
      double new_ll_vector_norm_l = 
	qnode->stat().ll_vector_norm_l_ +
	MatrixUtil::EntrywiseLpNorm(qnode->stat().postponed_ll_vector_l_, 1) +
	MatrixUtil::EntrywiseLpNorm(delta_l, 1);
      double new_ll_vector_used_error = qnode->stat().ll_vector_used_error_ +
	qnode->stat().postponed_ll_vector_used_error_;
      double new_ll_vector_n_pruned = qnode->stat().ll_vector_n_pruned_ +
	qnode->stat().postponed_ll_vector_n_pruned_;

      double new_neg_ll_vector_norm_l = 
	qnode->stat().neg_ll_vector_norm_l_ +
	MatrixUtil::EntrywiseLpNorm
	(qnode->stat().postponed_neg_ll_vector_u_, 1) +
	MatrixUtil::EntrywiseLpNorm(delta_neg_u, 1);
      double new_neg_ll_vector_used_error = 
	qnode->stat().neg_ll_vector_used_error_ +
	qnode->stat().postponed_neg_ll_vector_used_error_;
      double new_neg_ll_vector_n_pruned = 
	qnode->stat().neg_ll_vector_n_pruned_ + 
	qnode->stat().postponed_neg_ll_vector_n_pruned_;

      // Compute the allowed amount of error for pruning the given query
      // and reference pair.
      double allowed_err = 
	(relative_error * new_ll_vector_norm_l - new_ll_vector_used_error) /
	(total_alloc_error - new_ll_vector_n_pruned);
      double neg_allowed_err = 
	(relative_error * new_neg_ll_vector_norm_l - 
	 new_neg_ll_vector_used_error) /
	(total_alloc_error - new_neg_ll_vector_n_pruned);

      // Record how much error and pruned portion will be if pruning
      // were to succeed.
      delta_used_error = 
	0.5 * (positive_dot_product_range.hi * kernel_value_range.hi -
	       positive_dot_product_range.lo * kernel_value_range.lo) *
	(rnode->stat().sum_reference_point_expansion_norm_);
      delta_n_pruned = 
	rnode->stat().sum_reference_point_expansion_norm_;
      
      delta_neg_used_error =
	0.5 * (negative_dot_product_range.hi * kernel_value_range.lo -
	       negative_dot_product_range.lo * kernel_value_range.hi) * 
	(rnode->stat().sum_reference_point_expansion_norm_);
      delta_neg_n_pruned = 
	rnode->stat().sum_reference_point_expansion_norm_;

      // check pruning condition
      return (delta_used_error <= allowed_err &&
	      delta_neg_used_error <= neg_allowed_err);
    }
};

#endif
