#ifndef INSIDE_DUALTREE_VKDE_H
#error "This is not a public header file!"
#endif

#include "inverse_normal_cdf.h"

template<typename TKernel>
void DualtreeVKde<TKernel>::DualtreeVKdeBase_(Tree *qnode, Tree *rnode,
					      double probability) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().ResetBoundStatistics();

  // Compute unnormalized sum for each query point.
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {

    // Incorporate the postponed information.
    DualtreeKdeCommon::AddPostponed(qnode, q, this);

    // Get the query point.
    const double *q_col = qset_.GetColumnPtr(q);
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // pairwise distance and kernel value
      double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
      double kernel_value = kernels_[r].EvalUnnormOnSq(dsqd);
      double weighted_kernel_value = rset_weights_[r] * kernel_value;
      
      densities_l_[q] += weighted_kernel_value;
      densities_e_[q] += weighted_kernel_value;
      densities_u_[q] += weighted_kernel_value;

    } // end of iterating over each reference point.
    
    // Each query point has taken care of all reference points.
    n_pruned_[q] += rnode->stat().weight_sum_;
    
    // Subtract the number of reference points to undo the assumption
    // made in the function PreProcess.
    densities_u_[q] -= rnode->stat().weight_sum_;
    
    // Refine min and max summary statistics.
    DualtreeKdeCommon::RefineBoundStatistics(q, qnode, this);

  } // end of looping over each query point.

  // Clear postponed information.
  qnode->stat().ClearPostponed();
}

template<typename TKernel>
double DualtreeVKde<TKernel>::EvalUnnormOnSq_(index_t reference_point_index,
					      double squared_distance) {
  return kernels_[reference_point_index].
    EvalUnnormOnSq(squared_distance);
}

template<typename TKernel>
bool DualtreeVKde<TKernel>::MonteCarloPrunable_
(Tree *qnode, Tree *rnode, double probability, DRange &dsqd_range,
 DRange &kernel_value_range, double &dl, double &de, double &du, 
 double &used_error, double &n_pruned) {

  // If the reference node contains too few points, then return.
  if(qnode->count() * rnode->count() < num_initial_samples_per_query_) {
    return false;
  }

  // Refine the lower bound using the new lower bound info.
  VKdeStat<TKernel> &stat = qnode->stat();
  double max_used_error = 0;

  // Take random query/reference pair samples and determine how many
  // more samples are needed.
  bool flag = true;
  
  // Reset the current position of the scratch space to zero.
  double kernel_sums = 0;
  double squared_kernel_sums = 0;

  // Commence sampling...
  {
    double standard_score = 
      InverseNormalCDF::Compute(probability + 0.5 * (1 - probability));

    // The initial number of samples is equal to the default.
    int num_samples = num_initial_samples_per_query_;
    int total_samples = 0;

    do {
      for(index_t s = 0; s < num_samples; s++) {
	
	index_t random_query_point_index =
	  math::RandInt(qnode->begin(), qnode->end());
	index_t random_reference_point_index = 
	  math::RandInt(rnode->begin(), rnode->end());

	// Get the pointer to the current query point.
	const double *query_point = 
	  qset_.GetColumnPtr(random_query_point_index);
	
	// Get the pointer to the current reference point.
	const double *reference_point = 
	  rset_.GetColumnPtr(random_reference_point_index);
	
	// Compute the pairwise distance and kernel value.
	double squared_distance = la::DistanceSqEuclidean(rset_.n_rows(), 
							  query_point,
							  reference_point);

	double weighted_kernel_value = kernels_[random_reference_point_index].
	  EvalUnnormOnSq(squared_distance);
	kernel_sums += weighted_kernel_value;
	squared_kernel_sums += weighted_kernel_value * weighted_kernel_value;

      } // end of taking samples for this roune...

      // Increment total number of samples.
      total_samples += num_samples;

      // Compute the current estimate of the sample mean and the
      // sample variance.
      double sample_mean = kernel_sums / ((double) total_samples);
      double sample_variance =
	(squared_kernel_sums - total_samples * sample_mean * sample_mean) / 
	((double) total_samples - 1);

      // Compute the current threshold for guaranteeing the relative
      // error bound.
      double new_used_error = stat.used_error_ +
	stat.postponed_used_error_;
      double new_n_pruned = stat.n_pruned_ + stat.postponed_n_pruned_;

      // The currently proven lower bound.
      double new_mass_l = stat.mass_l_ + stat.postponed_l_ + dl;
      double new_max_kernel_value_l = threshold_;
      double right_hand_side = 
	(std::max(relative_error_ * new_mass_l, new_max_kernel_value_l) - 
	 new_used_error) / 
	(rroot_->stat().farfield_expansion_.get_weight_sum() - new_n_pruned);
      
      // NOTE: It is very important that the following pruning rule is
      // a strict inequality!
      if(sqrt(sample_variance) * standard_score < right_hand_side) {
	kernel_sums = kernel_sums / ((double) total_samples) * 
	  rnode->stat().farfield_expansion_.get_weight_sum();
	max_used_error = rnode->stat().farfield_expansion_.get_weight_sum() * 
	  standard_score * sqrt(sample_variance);
	break;
      }
      else {
	flag = false;
	break;
      }

    } while(true);

  } // end of sampling...

  // If all queries can be pruned, then add the approximations.
  if(flag) {
    de = kernel_sums;
    used_error = max_used_error;
    return true;
  }
  return false;
}

template<typename TKernel>
bool DualtreeVKde<TKernel>::DualtreeVKdeCanonical_
(Tree *qnode, Tree *rnode, double probability) {

  // temporary variable for storing lower bound change.
  double dl = 0, de = 0, du = 0;
  
  // temporary variables for holding used error for pruning.
  double used_error = 0, n_pruned = 0;
  
  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;
  
  // First compute distance/kernel value bounds.
  dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
  dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
  kernel_value_range.lo = rnode->stat().min_bandwidth_kernel_.
    EvalUnnormOnSq(dsqd_range.hi);
  kernel_value_range.hi = rnode->stat().max_bandwidth_kernel_.
    EvalUnnormOnSq(dsqd_range.lo);

  // Try finite difference pruning first.
  if(DualtreeKdeCommon::Prunable(qnode, rnode, probability, dsqd_range, 
				 kernel_value_range, dl, 
				 de, du, used_error, n_pruned, this)) {
    qnode->stat().postponed_l_ += dl;
    qnode->stat().postponed_e_ += de;
    qnode->stat().postponed_u_ += du;
    qnode->stat().postponed_used_error_ += used_error;
    qnode->stat().postponed_n_pruned_ += n_pruned;
    num_finite_difference_prunes_++;
    return true;
  }

  // Then Monte Carlo-based pruning.
  else if(probability < 1 &&
	  DualtreeKdeCommon::MonteCarloPrunableByOrderStatistics_
	  (qnode, rnode, probability, dsqd_range, 
	   kernel_value_range, dl, de, du, used_error, n_pruned, this)) {
    qnode->stat().postponed_l_ += dl;
    qnode->stat().postponed_e_ += de;
    qnode->stat().postponed_u_ += du;
    qnode->stat().postponed_used_error_ += used_error;
    qnode->stat().postponed_n_pruned_ += n_pruned;
    num_monte_carlo_prunes_++;
    return false;
  }
  
  // For a leaf query node,
  if(qnode->is_leaf()) {
    
    // For leaf pairs, do exhaustive computations.
    if(rnode->is_leaf()) {
      DualtreeVKdeBase_(qnode, rnode, probability);
      return true;
    }
    
    // For a non-leaf reference, expand reference node,
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
      DualtreeKdeCommon::BestNodePartners
	(qnode, rnode->left(), rnode->right(), probability,
	 &rnode_first, &probability_first, &rnode_second, &probability_second);
      
      bool first_result = 
	DualtreeVKdeCanonical_(qnode, rnode_first, probability_first);

      // If the first recursion is computed exactly, then increment
      // the probability tolerance for the second recursion.
      if(first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool second_result =
	DualtreeVKdeCanonical_(qnode, rnode_second, probability_second);
      return first_result && second_result;
    }
  }
  
  // For a non-leaf query node,
  else {

    // The boolean flag that states that the contribution of the
    // current reference node is computed exactly for the current
    // query node.
    bool result = true;

    // Push down postponed bound changes owned by the current query
    // node to the children of the query node and clear them.
    qnode->left()->stat().AddPostponed(qnode->stat());
    qnode->right()->stat().AddPostponed(qnode->stat());
    
    // Clear out the postponed info after being passed down.
    qnode->stat().ClearPostponed();
    
    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      Tree *qnode_first = NULL, *qnode_second = NULL;
      double probability_first = 0, probability_second = 0;

      DualtreeKdeCommon::BestNodePartners
	(rnode, qnode->left(), qnode->right(), probability,
	 &qnode_first, &probability_first, &qnode_second, &probability_second);
      bool first_result =
	DualtreeVKdeCanonical_(qnode_first, rnode, probability);
      bool second_result =
	DualtreeVKdeCanonical_(qnode_second, rnode, probability);
      result = first_result && second_result;
    }
    
    // For a non-leaf reference node, expand both query and reference
    // nodes.
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
  
      // Fix the query node to be the left child, and recurse.
      DualtreeKdeCommon::BestNodePartners
	(qnode->left(), rnode->left(), rnode->right(), 
	 probability, &rnode_first, &probability_first,
	 &rnode_second, &probability_second);
      bool left_first_result =
	DualtreeVKdeCanonical_(qnode->left(), rnode_first, probability_first);

      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(left_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool left_second_result = 
	DualtreeVKdeCanonical_(qnode->left(), rnode_second, 
			       probability_second);
      
      // Fix the query node to be the right child, and recurse.
      DualtreeKdeCommon::BestNodePartners
	(qnode->right(), rnode->left(), rnode->right(), 
	 probability, &rnode_first, &probability_first,
	 &rnode_second, &probability_second);
      bool right_first_result =
	DualtreeVKdeCanonical_(qnode->right(), rnode_first, probability_first);


      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(right_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool right_second_result =
	DualtreeVKdeCanonical_(qnode->right(), rnode_second,
			       probability_second);

      result = left_first_result && left_second_result &&
	right_first_result && right_second_result;
    }
    
    // Reaccumulate the summary statistics.
    qnode->stat().RefineBoundStatistics(qnode->left()->stat(),
					qnode->right()->stat());
    return result;
  } // end of the case: non-leaf query node.

} // end of DualtreeVKdeCanonical_

template<typename TKernel>
void DualtreeVKde<TKernel>::PreProcess(Tree *node, bool reference_side) {

  // Initialize lower bound to 0.
  node->stat().mass_l_ = 0;
  
  // Set the upper bound to the sum of the reference weights.
  node->stat().mass_u_ = rset_weight_sum_;
  
  node->stat().used_error_ = 0;
  node->stat().n_pruned_ = 0;
  
  // Postponed lower and upper bound density changes to 0.
  node->stat().postponed_l_ = node->stat().postponed_u_ = 0;
  
  // Set the finite difference approximated amounts to 0.
  node->stat().postponed_e_ = 0;
  
  // Set the error incurred to 0.
  node->stat().postponed_used_error_ = 0;
  
  // set the number of pruned reference points to 0
  node->stat().postponed_n_pruned_ = 0;
  
  // for non-leaf node, recurse
  if(!node->is_leaf()) {    
    PreProcess(node->left(), reference_side);
    PreProcess(node->right(), reference_side);

    if(reference_side) {
      // After recursing set the max/min bandwidth and the weight sum
      // approproiately.
      node->stat().min_bandwidth_kernel_.Init
	(std::min
	 (sqrt(node->left()->stat().min_bandwidth_kernel_.bandwidth_sq()),
	  sqrt(node->right()->stat().min_bandwidth_kernel_.bandwidth_sq())));
      node->stat().max_bandwidth_kernel_.Init
	(std::max
	 (sqrt(node->left()->stat().max_bandwidth_kernel_.bandwidth_sq()),
	  sqrt(node->right()->stat().max_bandwidth_kernel_.bandwidth_sq())));
      node->stat().weight_sum_ =
	node->left()->stat().weight_sum_ + node->right()->stat().weight_sum_;
    }
  }
  else {
    
    if(reference_side) {
      node->stat().min_bandwidth_kernel_.Init(sqrt(DBL_MAX));
      node->stat().max_bandwidth_kernel_.Init(0);
      node->stat().weight_sum_ = 0;

      // Reset the minimum/maximum bandwidths owned by the node.
      for(index_t i = node->begin(); i < node->end(); i++) {
	node->stat().min_bandwidth_kernel_.Init
	  (std::min(sqrt(node->stat().min_bandwidth_kernel_.bandwidth_sq()),
		    sqrt(kernels_[i].bandwidth_sq())));
	node->stat().max_bandwidth_kernel_.Init
	  (std::max(sqrt(node->stat().max_bandwidth_kernel_.bandwidth_sq()),
		    sqrt(kernels_[i].bandwidth_sq())));
	node->stat().weight_sum_ += rset_weights_[i];
      }
    }
  }
}

template<typename TKernel>
void DualtreeVKde<TKernel>::PostProcess(Tree *qnode) {
    
  VKdeStat<TKernel> &qstat = qnode->stat();
  
  // For a leaf query node,
  if(qnode->is_leaf()) {

    // Clear the summary statistics of the current query node so that
    // we can refine it to better bounds.
    qstat.ResetBoundStatistics();

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Add all postponed quantities.
      DualtreeKdeCommon::AddPostponed(qnode, q, this);

      // If leave-one-out, then subtract the weight of the point from
      // the accumulated sum.
      if(leave_one_out_) {
	densities_e_[q] -= rset_weights_[q];
	
	densities_l_[q] *= (mult_const_ / 
			    (rset_weight_sum_ - rset_weights_[q]));
	densities_e_[q] *= (mult_const_ / 
			    (rset_weight_sum_ - rset_weights_[q]));
	densities_u_[q] *= (mult_const_ / 
			    (rset_weight_sum_ - rset_weights_[q]));
      }
      else {
	// Normalize the densities.
	densities_l_[q] *= (mult_const_ / rset_weight_sum_);
	densities_e_[q] *= (mult_const_ / rset_weight_sum_);
	densities_u_[q] *= (mult_const_ / rset_weight_sum_);
      }

      // Refine bound statistics using the finalized query point sum.
      DualtreeKdeCommon::RefineBoundStatistics(q, qnode, this);
    }

    // Clear postponed approximations since they have been
    // incorporated.
    qstat.ClearPostponed();
  }
  else {
    
    // Push down approximations.
    qnode->left()->stat().AddPostponed(qstat);
    qnode->right()->stat().AddPostponed(qstat);

    // Clear postponed approximations.
    qstat.ClearPostponed();

    // Recurse to the left and to the right.
    PostProcess(qnode->left());
    PostProcess(qnode->right());

    // Refine statistics after recursing.
    qstat.RefineBoundStatistics(qnode->left()->stat(), qnode->right()->stat());
  }
}
