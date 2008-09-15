#ifndef DUALTREE_KDE_CV_COMMON_H
#define DUALTREE_KDE_CV_COMMON_H

#include "inverse_normal_cdf.h"

class DualtreeKdeCVCommon {

 public:

  template<typename TTree, typename TAlgorithm>
  bool MonteCarloPrunable_
  (TTree *qnode, TTree *rnode, double probability, DRange &dsqd_range,
   DRange &kernel_value_range, double &dl, double &de, double &du, 
   double &used_error, double &n_pruned, TAlgorithm *kde_object) {
    
    // If the reference node contains too few points, then return.
    if(qnode->count() * rnode->count() < 25) {
      return false;
    }
    
    // Refine the lower bound using the new lower bound info.
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
      int num_samples = 25;
      int total_samples = 0;
      
      do {
	for(index_t s = 0; s < num_samples; s++) {
	  
	  index_t random_query_point_index =
	    math::RandInt(qnode->begin(), qnode->end());
	  index_t random_reference_point_index = 
	    math::RandInt(rnode->begin(), rnode->end());
	  
	  // Get the pointer to the current query point.
	  const double *query_point = 
	    (kde_object->qset_).GetColumnPtr(random_query_point_index);
	  
	  // Get the pointer to the current reference point.
	  const double *reference_point = 
	    (kde_object->rset_).GetColumnPtr(random_reference_point_index);
	  
	  // Compute the pairwise distance and kernel value.
	  double squared_distance = la::DistanceSqEuclidean
	    ((kde_object->rset_).n_rows(), query_point, reference_point);
	  
	  double weighted_kernel_value = 
	    kde_object->EvalUnnormOnSq_(random_reference_point_index,
					squared_distance);
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
	double new_used_error = qnode->stat().used_error_ +
	  qnode->stat().postponed_used_error_;
	double new_n_pruned = qnode->stat().n_pruned_ + 
	  qnode->stat().postponed_n_pruned_;
	
	// The currently proven lower bound.
	double new_mass_l = qnode->stat().mass_l_ + 
	  qnode->stat().postponed_l_ + dl;
	double right_hand_side = 
	  (kde_object->relative_error_ * new_mass_l - new_used_error) /
	  (kde_object->rroot_->stat().get_weight_sum() - new_n_pruned);
	
	// NOTE: It is very important that the following pruning rule is
	// a strict inequality!
	if(sqrt(sample_variance) * standard_score < right_hand_side) {
	  kernel_sums = kernel_sums / ((double) total_samples) * 
	    rnode->stat().get_weight_sum();
	  max_used_error = rnode->stat().get_weight_sum() * 
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

  template<typename TTree, typename TAlgorithm>
  static bool MonteCarloPrunableByOrderStatistics_
  (TTree *qnode, TTree *rnode, double probability, DRange &dsqd_range,
   DRange &kernel_value_range, double &dl, double &de, double &du, 
   double &used_error, double &n_pruned, TAlgorithm *kde_object) {
    
    // Currently running minimum/maximum kernel values.
    double min_kernel_value = -DBL_MAX;
    
    // Locate the minimum required number of samples to achieve the
    // prescribed probability level.
    int num_samples = 0;
    
    for(index_t i = kde_object->coverage_probabilities_.length() - 1; i >= 0; 
	i--) {
      if(kde_object->coverage_probabilities_[i] >= probability) {
	num_samples = (kde_object->sample_multiple_) * (i + 1);
	break;
      }
    }
    
    if(num_samples == 0 || num_samples > qnode->count() * rnode->count()) {
      return false;
    }
    
    for(index_t s = 0; s < num_samples; s++) {
      
      index_t random_query_point_index =
	math::RandInt(qnode->begin(), qnode->end());
      index_t random_reference_point_index = 
	math::RandInt(rnode->begin(), rnode->end());
      
      // Get the pointer to the current query point.
      const double *query_point = 
	(kde_object->qset_).GetColumnPtr(random_query_point_index);
      
      // Get the pointer to the current reference point.
      const double *reference_point = 
	(kde_object->rset_).GetColumnPtr(random_reference_point_index);
      
      // Compute the pairwise distance and kernel value.
      double squared_distance = la::DistanceSqEuclidean
	((kde_object->rset_).n_rows(), query_point, reference_point);
      
      double kernel_value = kde_object->EvalUnnormOnSq_
	(random_reference_point_index, squared_distance);
      min_kernel_value = std::max(min_kernel_value, kernel_value);
      
    } // end of taking samples for this routine...
    
    // Compute the current threshold for guaranteeing the relative
    // error bound.
    double new_used_error = qnode->stat().used_error_ +
      qnode->stat().postponed_used_error_;
    double new_n_pruned = qnode->stat().n_pruned_ + 
      qnode->stat().postponed_n_pruned_;
    
    // The probabilistic lower bound change due to sampling.
    dl = rnode->stat().get_weight_sum() * min_kernel_value;
    
    // The currently proven lower bound.
    double new_mass_l = qnode->stat().mass_l_ + 
      qnode->stat().postponed_l_ + dl;
    double left_hand_side = 0.5 * (kernel_value_range.hi - min_kernel_value);
    double right_hand_side = 
      (kde_object->relative_error_ * new_mass_l - new_used_error) / 
      (kde_object->rroot_->stat().get_weight_sum() - new_n_pruned);
    
    if(left_hand_side <= right_hand_side) {
      de = 0.5 * (min_kernel_value + kernel_value_range.hi) * 
	rnode->stat().get_weight_sum();
      used_error = rnode->stat().get_weight_sum() *
	0.5 * (kernel_value_range.hi - min_kernel_value);
      return true;
    }
    else {
      return false;
    }
  }
  
  template<typename TTree, typename TAlgorithm>
  static bool Prunable(TTree *qnode, TTree *rnode, double probability, 
		       DRange &dsqd_range, DRange &first_kernel_value_range,
		       DRange &second_kernel_value_range,
		       double &first_dl, double &first_de, double &first_du, 
		       double &first_used_error, 
		       double &second_dl, double &second_de, double &second_du,
		       double &second_used_error, double &n_pruned,
		       TAlgorithm *kde_object) {
    
    // the new lower bound after incorporating new info
    first_dl = first_kernel_value_range.lo * qnode->count() * 
      rnode->stat().get_weight_sum();
    first_de = 0.5 * qnode->count() * rnode->stat().get_weight_sum() * 
      (first_kernel_value_range.lo + first_kernel_value_range.hi);
    first_du = (first_kernel_value_range.hi - 1) * qnode->count() *
      rnode->stat().get_weight_sum();
    second_dl = second_kernel_value_range.lo * qnode->count() *
      rnode->stat().get_weight_sum();
    second_de = 0.5 * qnode->count() * rnode->stat().get_weight_sum() * 
      (second_kernel_value_range.lo + second_kernel_value_range.hi);
    second_du = (second_kernel_value_range.hi - 1) * qnode->count() * 
      rnode->stat().get_weight_sum();
   
    // Refine the lower bound using the new lower bound info.
    double first_new_mass_l = (kde_object->first_sum_l_) + first_dl;
    double second_new_mass_l = (kde_object-> second_sum_l_) + second_dl;
    
    // Compute the allowed error.
    double proportion =  1.0 / (1.0 - n_pruned) * 
      (((double)qnode->count()) / ((double) kde_object->rroot_->count())) *
      (rnode->stat().get_weight_sum() / 
       kde_object->rroot_->stat().get_weight_sum());
    double first_allowed_err = 
      (kde_object->relative_error_ * first_new_mass_l - 
       kde_object->first_used_error_) *
      proportion;
    double second_allowed_err =
      (kde_object->relative_error_ * second_new_mass_l - 
       kde_object->second_used_error_) *
      proportion;

    // This is error per each query/reference pair for a fixed query
    double first_kernel_diff = 
      0.5 * (first_kernel_value_range.hi - first_kernel_value_range.lo);
    double second_kernel_diff =
      0.5 * (second_kernel_value_range.hi - second_kernel_value_range.lo);
    
    // this is total error for each query point
    first_used_error = first_kernel_diff * qnode->count() * 
      rnode->stat().get_weight_sum();
    second_used_error = second_kernel_diff * qnode->count() *
      rnode->stat().get_weight_sum();
    
    // number of reference points for possible pruning.
    n_pruned = (((double) qnode->count()) / 
		((double) kde_object->rroot_->count())) *
      (rnode->stat().get_weight_sum() /
       kde_object->rroot_->stat().get_weight_sum());
    
    // If the error bound is satisfied by the hard error bound, it is
    // safe to prune.
    return (!isnan(first_allowed_err)) && (!isnan(second_allowed_err)) &&
      (first_used_error <= first_allowed_err) &&
      (second_used_error <= second_allowed_err);
  }
};

#endif
