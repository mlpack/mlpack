#ifndef INSIDE_DUALTREE_KDE_H
#error "This is not a public header file!"
#endif

#include "inverse_normal_cdf.h"

template<typename TKernelAux>
double DualtreeKde<TKernelAux>::OuterConfidenceInterval
(double population_size, double sample_size,
 double sample_order_statistics_min_index,
 double population_order_statistics_min_index) {
  
  double total_probability = 0;
  double lower_percentile = population_order_statistics_min_index /
    population_size;

  for(double r_star = sample_order_statistics_min_index;
      r_star <= std::min(population_order_statistics_min_index, sample_size);
      r_star += 1.0) {
    
    // If any of the arguments to the binomial coefficient is
    // invalid, then the contribution is zero.
    if(r_star > population_order_statistics_min_index ||
       sample_size - r_star < 0 || 
       population_size - population_order_statistics_min_index < 0 ||
       sample_size - r_star >
       population_size - population_order_statistics_min_index) {
      continue;
    }
    
    /*
    total_probability +=
      BinomialCoefficientHelper_
      (population_order_statistics_min_index, r_star,
       population_size - population_order_statistics_min_index,
       sample_size - r_star, population_size, sample_size);
    */
    total_probability +=
      math::BinomialCoefficient((int) sample_size, (int) r_star) *
      pow(lower_percentile, r_star) * 
      pow(1 - lower_percentile, sample_size - r_star);
  }
  return std::max(std::min(total_probability, 1.0), 0.0);
}

template<typename TKernelAux>
double DualtreeKde<TKernelAux>::BinomialCoefficientHelper_
(double n3, double k3, double n1, double k1, double n2, double k2) {
  
  double n_k3 = n3 - k3;
  double n_k1 = n1 - k1;
  double n_k2 = n2 - k2;
  double nchsk = 1;
  double i;

  if(k3 > n3 || k3 < 0 || k1 > n1 || k1 < 0 || k2 > n2 || k2 < 0) {
    return 0;
  }
  
  if(k3 < n_k3) {
    k3 = n_k3;
    n_k3 = n3 - k3;
  }
  if(k1 < n_k1) {
    k1 = n_k1;
    n_k1 = n1 - k1;
  }
  if(k2 < n_k2) {
    k2 = n_k2;
    n_k2 = n2 - k2;
  }
  
  double min_index = std::min(n_k1, n_k2);
  double max_index = std::max(n_k1, n_k2);
  for(i = 1; i <= min_index; i += 1.0) {
    k1 += 1.0;
    k2 += 1.0;
    nchsk *= k1;
    nchsk /= k2;
  }
  for(i = min_index + 1; i <= max_index; i += 1.0) {
    if(n_k1 < n_k2) {
      k2 += 1.0;
      nchsk *= i;
      nchsk /= k2;
    }
    else {
      k1 += 1.0;
      nchsk *= k1;
      nchsk /= i;
    }
  }
  for(i = 1; i <= n_k3; i += 1.0) {
    k3 += 1.0;
    nchsk *= k3;
    nchsk /= i;
  }
  
  return nchsk;
}

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::AddPostponed_(Tree *node, index_t destination) {
  densities_l_[destination] += node->stat().postponed_l_;
  densities_e_[destination] += node->stat().postponed_e_;
  densities_u_[destination] += node->stat().postponed_u_;
  used_error_[destination] += node->stat().postponed_used_error_;
  n_pruned_[destination] += node->stat().postponed_n_pruned_; 
}

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::RefineBoundStatistics_(index_t q, Tree *qnode) {
  qnode->stat().mass_l_ = std::min(qnode->stat().mass_l_, densities_l_[q]);
  qnode->stat().mass_u_ = std::max(qnode->stat().mass_u_, densities_u_[q]);
  qnode->stat().used_error_ = std::max(qnode->stat().used_error_,
				       used_error_[q]);
  qnode->stat().n_pruned_ = std::min(qnode->stat().n_pruned_, n_pruned_[q]);
}

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::RefineBoundStatistics_(Tree *qnode) {
  
  Vector v;
  v.Init(qnode->count());
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    v[q - qnode->begin()] = densities_l_[q];
  }
  qsort((void *) v.ptr(), qnode->count(), sizeof(double), 
	&DualtreeKde<GaussianKernelMultAux>::qsort_comparator_);

  qnode->stat().mass_l_ = v[(int) floor(lower_percentile_ * qnode->count())];
}

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::ShuffleAccordingToPermutation_
(Vector &v, const ArrayList<index_t> &permutation) {
  
  Vector v_tmp;
  v_tmp.Init(v.length());
  for(index_t i = 0; i < v_tmp.length(); i++) {
    v_tmp[i] = v[permutation[i]];
  }
  v.CopyValues(v_tmp);
}

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::DualtreeKdeBase_(Tree *qnode, Tree *rnode,
					       double probability) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().ResetBoundStatistics();

  // Compute unnormalized sum for each query point.
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {

    // Incorporate the postponed information.
    AddPostponed_(qnode, q);

    // Get the query point.
    const double *q_col = qset_.GetColumnPtr(q);
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // pairwise distance and kernel value
      double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
      double kernel_value = ka_.kernel_.EvalUnnormOnSq(dsqd);
      double weighted_kernel_value = rset_weights_[r] * kernel_value;
      
      densities_l_[q] += weighted_kernel_value;
      densities_e_[q] += weighted_kernel_value;
      densities_u_[q] += weighted_kernel_value;
    } // end of iterating over each reference point.
    
    // Each query point has taken care of all reference points.
    n_pruned_[q] += rnode->stat().farfield_expansion_.get_weight_sum();
    
    // Subtract the number of reference points to undo the assumption
    // made in the function PreProcess.
    densities_u_[q] -= rnode->stat().farfield_expansion_.get_weight_sum();
    
    // Refine min and max summary statistics.
    RefineBoundStatistics_(q, qnode);

  } // end of looping over each query point.

  // Clear postponed information.
  qnode->stat().ClearPostponed();

  // Refine statistics again.
  RefineBoundStatistics_(qnode);
}

template<typename TKernelAux>
bool DualtreeKde<TKernelAux>::PrunableEnhanced_
(Tree *qnode, Tree *rnode, double probability, DRange &dsqd_range, 
 DRange &kernel_value_range, double &dl, double &du, double &used_error, 
 double &n_pruned, int &order_farfield_to_local, int &order_farfield, 
 int &order_local) {
  
  int dim = rset_.n_rows();
  
  // actual amount of error incurred per each query/ref pair
  double actual_err_farfield_to_local = 0;
  double actual_err_farfield = 0;
  double actual_err_local = 0;
  
  // estimated computational cost
  int cost_farfield_to_local = INT_MAX;
  int cost_farfield = INT_MAX;
  int cost_local = INT_MAX;
  int cost_exhaustive = (qnode->count()) * (rnode->count()) * dim;
  int min_cost = 0;
  
  // query node and reference node statistics
  KdeStat<TKernelAux> &qstat = qnode->stat();
  KdeStat<TKernelAux> &rstat = rnode->stat();
  
  // expansion objects
  typename TKernelAux::TFarFieldExpansion &farfield_expansion = 
    rstat.farfield_expansion_;
  typename TKernelAux::TLocalExpansion &local_expansion = 
    qstat.local_expansion_;
  
  // Refine the lower bound using the new lower bound info
  double new_mass_l = qstat.mass_l_ + qstat.postponed_l_ + dl;
  double new_used_error = qstat.used_error_ + qstat.postponed_used_error_;
  double new_n_pruned = qstat.n_pruned_ + qstat.postponed_n_pruned_;
  double allowed_err =
    (tau_ * new_mass_l - new_used_error) /
    ((double) rroot_->stat().farfield_expansion_.get_weight_sum() - 
     new_n_pruned);

  // If the allowed error is not defined (NaN), then we cannot
  // approximate.
  if(isnan(allowed_err)) {
    return false;
  }

  // Get the order of approximations.
  order_farfield_to_local = 
    farfield_expansion.OrderForConvertingToLocal
    (rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi, 
     allowed_err, &actual_err_farfield_to_local);
  order_farfield = 
    farfield_expansion.OrderForEvaluating(rnode->bound(), qnode->bound(),
					  dsqd_range.lo, dsqd_range.hi,
					  allowed_err, &actual_err_farfield);
  order_local = 
    local_expansion.OrderForEvaluating(rnode->bound(), qnode->bound(), 
				       dsqd_range.lo, dsqd_range.hi,
				       allowed_err, &actual_err_local);
  
  // Update computational cost and compute the minimum.
  if(order_farfield_to_local >= 0) {
    cost_farfield_to_local = (int) 
      ka_.sea_.FarFieldToLocalTranslationCost(order_farfield_to_local);
  }
  if(order_farfield >= 0) {
    cost_farfield = (int) 
      ka_.sea_.FarFieldEvaluationCost(order_farfield) * (qnode->count());
  }
  if(order_local >= 0) {
    cost_local = (int) 
      ka_.sea_.DirectLocalAccumulationCost(order_local) * (rnode->count());
  }
  
  min_cost = min(cost_farfield_to_local, 
		 min(cost_farfield, min(cost_local, cost_exhaustive)));
  
  if(cost_farfield_to_local == min_cost) {
    used_error = rnode->stat().farfield_expansion_.get_weight_sum() * 
      actual_err_farfield_to_local;
    n_pruned = rnode->stat().farfield_expansion_.get_weight_sum();
    order_farfield = order_local = -1;
    num_farfield_to_local_prunes_++;
    return true;
  }
  
  if(cost_farfield == min_cost) {
    used_error = rnode->stat().farfield_expansion_.get_weight_sum() * 
      actual_err_farfield;
    n_pruned = rnode->stat().farfield_expansion_.get_weight_sum();
    order_farfield_to_local = order_local = -1;
    num_farfield_prunes_++;
    return true;
  }
  
  if(cost_local == min_cost) {
    used_error = rnode->stat().farfield_expansion_.get_weight_sum() * 
      actual_err_local;
    n_pruned = rnode->stat().farfield_expansion_.get_weight_sum();
    order_farfield_to_local = order_farfield = -1;
    num_local_prunes_++;
    return true;
  }
  
  order_farfield_to_local = order_farfield = order_local = -1;
  dl = du = used_error = n_pruned = 0;
  return false;
}

template<typename TKernelAux>
bool DualtreeKde<TKernelAux>::MonteCarloPrunableByOrderStatistics_
(Tree *qnode, Tree *rnode, double probability, DRange &dsqd_range,
 DRange &kernel_value_range, double &dl, double &de, double &du, 
 double &used_error, double &n_pruned) {

  KdeStat<TKernelAux> &stat = qnode->stat();

  // Currently running minimum/maximum kernel values.
  double min_kernel_value = -DBL_MAX;

  // Locate the minimum required number of samples to achieve the
  // prescribed probability level.
  int num_samples = 0;

  for(index_t i = coverage_probabilities_.length() - 1; i >= 0; i--) {
    if(coverage_probabilities_[i] >= probability) {
      num_samples = sample_multiple_ * (i + 1);
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
      qset_.GetColumnPtr(random_query_point_index);
    
    // Get the pointer to the current reference point.
    const double *reference_point = 
      rset_.GetColumnPtr(random_reference_point_index);
    
    // Compute the pairwise distance and kernel value.
    double squared_distance = la::DistanceSqEuclidean(rset_.n_rows(), 
						      query_point,
						      reference_point);
    
    double kernel_value = ka_.kernel_.EvalUnnormOnSq(squared_distance);
    min_kernel_value = std::max(min_kernel_value, kernel_value);
    
  } // end of taking samples for this roune...
  
  // Compute the current threshold for guaranteeing the relative
  // error bound.
  double new_used_error = stat.used_error_ +
    stat.postponed_used_error_;
  double new_n_pruned = stat.n_pruned_ + stat.postponed_n_pruned_;
  
  // The probabilistic lower bound change due to sampling.
  dl = rnode->stat().farfield_expansion_.get_weight_sum() * min_kernel_value;
  
  // The currently proven lower bound.
  double new_mass_l = stat.mass_l_ + stat.postponed_l_ + dl;
  double left_hand_side = 0.5 * (kernel_value_range.hi - min_kernel_value)
    * (1 - threshold_);
  double right_hand_side = (tau_ * new_mass_l - new_used_error) / 
    (rroot_->stat().farfield_expansion_.get_weight_sum() - new_n_pruned);
  
  if(left_hand_side <= right_hand_side) {
    de = 0.5 * (min_kernel_value + kernel_value_range.hi) * 
      rnode->stat().farfield_expansion_.get_weight_sum();
    used_error = rnode->stat().farfield_expansion_.get_weight_sum() *
      0.5 * (kernel_value_range.hi - min_kernel_value);
    return true;
  }
  else {
    return false;
  }
}

template<typename TKernelAux>
bool DualtreeKde<TKernelAux>::MonteCarloPrunable_
(Tree *qnode, Tree *rnode, double probability, DRange &dsqd_range,
 DRange &kernel_value_range, double &dl, double &de, double &du, 
 double &used_error, double &n_pruned) {

  // If the reference node contains too few points, then return.
  if(qnode->count() * rnode->count() < num_initial_samples_per_query_) {
    return false;
  }

  // Refine the lower bound using the new lower bound info.
  KdeStat<TKernelAux> &stat = qnode->stat();
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

	double weighted_kernel_value = 
	  ka_.kernel_.EvalUnnormOnSq(squared_distance);
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
      double right_hand_side = 
	(tau_ * new_mass_l - new_used_error) /
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

template<typename TKernelAux>
bool DualtreeKde<TKernelAux>::Prunable_
(Tree *qnode, Tree *rnode, double probability, DRange &dsqd_range,
 DRange &kernel_value_range, double &dl, double &de, double &du, 
 double &used_error, double &n_pruned) {
  
  // The query node stat
  KdeStat<TKernelAux> &stat = qnode->stat();
  
  // Try pruning after bound refinement: first compute distance/kernel
  // value bounds.
  dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
  dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
  kernel_value_range = ka_.kernel_.RangeUnnormOnSq(dsqd_range);
  
  // the new lower bound after incorporating new info
  dl = kernel_value_range.lo * 
    rnode->stat().farfield_expansion_.get_weight_sum();
  de = 0.5 * rnode->stat().farfield_expansion_.get_weight_sum() * 
    (kernel_value_range.lo + kernel_value_range.hi);
  du = (kernel_value_range.hi - 1) * 
    rnode->stat().farfield_expansion_.get_weight_sum();
  
  // refine the lower bound using the new lower bound info
  double new_mass_l = stat.mass_l_ + stat.postponed_l_ + dl;
  double new_used_error = stat.used_error_ + stat.postponed_used_error_;
  double new_n_pruned = stat.n_pruned_ + stat.postponed_n_pruned_;
  
  double allowed_err;

  // Compute the allowed error.
  allowed_err = (tau_ * new_mass_l - new_used_error) *
    rnode->stat().farfield_expansion_.get_weight_sum() / 
    ((double) rroot_->stat().farfield_expansion_.get_weight_sum() - 
     new_n_pruned);

  // This is error per each query/reference pair for a fixed query
  double kernel_diff = 0.5 * (kernel_value_range.hi - kernel_value_range.lo);
  
  // this is total error for each query point
  used_error = kernel_diff * 
    rnode->stat().farfield_expansion_.get_weight_sum();
  
  // number of reference points for possible pruning.
  n_pruned = rnode->stat().farfield_expansion_.get_weight_sum();

  // If the error bound is satisfied by the hard error bound, it is
  // safe to prune.
  return (!isnan(allowed_err)) && (used_error <= allowed_err);
}

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::BestNodePartners
(Tree *nd, Tree *nd1, Tree *nd2, double probability,
 Tree **partner1, double *probability1, Tree **partner2, 
 double *probability2) {
  
  double d1 = nd->bound().MinDistanceSq(nd1->bound());
  double d2 = nd->bound().MinDistanceSq(nd2->bound());

  // Prioritized traversal based on the squared distance bounds.
  if(d1 <= d2) {
    *partner1 = nd1;
    *probability1 = sqrt(probability);
    *partner2 = nd2;
    *probability2 = sqrt(probability);
  }
  else {
    *partner1 = nd2;
    *probability1 = sqrt(probability);
    *partner2 = nd1;
    *probability2 = sqrt(probability);
  }
}

template<typename TKernelAux>
bool DualtreeKde<TKernelAux>::DualtreeKdeCanonical_
(Tree *qnode, Tree *rnode, double probability) {

  // temporary variable for storing lower bound change.
  double dl = 0, de = 0, du = 0;
  int order_farfield_to_local = -1, order_farfield = -1, order_local = -1;
  
  // temporary variables for holding used error for pruning.
  double used_error = 0, n_pruned = 0;
  
  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;
  
  // Try finite difference pruning first.
  if(Prunable_(qnode, rnode, probability, dsqd_range, kernel_value_range, dl, 
	       de, du, used_error, n_pruned)) {
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
	  MonteCarloPrunableByOrderStatistics_
	  (qnode, rnode, probability, dsqd_range, 
	   kernel_value_range, dl, de, du, used_error, n_pruned)) {
    qnode->stat().postponed_l_ += dl;
    qnode->stat().postponed_e_ += de;
    qnode->stat().postponed_u_ += du;
    qnode->stat().postponed_used_error_ += used_error;
    qnode->stat().postponed_n_pruned_ += n_pruned;
    num_monte_carlo_prunes_++;
    return false;
  }

  else if(qset_.n_rows() <= 5 &&
	  PrunableEnhanced_(qnode, rnode, probability, dsqd_range, 
			    kernel_value_range, dl, du, used_error, n_pruned, 
			    order_farfield_to_local, order_farfield, 
			    order_local)) {
    
    // far field to local translation
    if(order_farfield_to_local >= 0) {
      rnode->stat().farfield_expansion_.TranslateToLocal
	(qnode->stat().local_expansion_, order_farfield_to_local);
    }
    // far field pruning
    else if(order_farfield >= 0) {
      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	densities_e_[q] += 
	  rnode->stat().farfield_expansion_.EvaluateField(qset_, q, 
							  order_farfield);
      }
    }
    // local accumulation pruning
    else if(order_local >= 0) {
      qnode->stat().local_expansion_.AccumulateCoeffs(rset_, rset_weights_,
						      rnode->begin(), 
						      rnode->end(),
						      order_local);
    }
    qnode->stat().postponed_l_ += dl;
    qnode->stat().postponed_u_ += du;
    qnode->stat().postponed_used_error_ += used_error;
    qnode->stat().postponed_n_pruned_ += n_pruned;
    return true;
  }
  
  // For a leaf query node,
  if(qnode->is_leaf()) {
    
    // For leaf pairs, do exhaustive computations.
    if(rnode->is_leaf()) {
      DualtreeKdeBase_(qnode, rnode, probability);
      return true;
    }
    
    // For a non-leaf reference, expand reference node,
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
      BestNodePartners(qnode, rnode->left(), rnode->right(), probability,
		       &rnode_first, &probability_first,
		       &rnode_second, &probability_second);
      
      bool first_result = 
	DualtreeKdeCanonical_(qnode, rnode_first, probability_first);

      // If the first recursion is computed exactly, then increment
      // the probability tolerance for the second recursion.
      if(first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool second_result =
	DualtreeKdeCanonical_(qnode, rnode_second, probability_second);
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

      BestNodePartners(rnode, qnode->left(), qnode->right(), probability,
		       &qnode_first, &probability_first,
		       &qnode_second, &probability_second);
      bool first_result =
	DualtreeKdeCanonical_(qnode_first, rnode, probability);
      bool second_result =
	DualtreeKdeCanonical_(qnode_second, rnode, probability);
      result = first_result && second_result;
    }
    
    // For a non-leaf reference node, expand both query and reference
    // nodes.
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
  
      // Fix the query node to be the left child, and recurse.
      BestNodePartners(qnode->left(), rnode->left(), rnode->right(), 
		       probability, &rnode_first, &probability_first,
		       &rnode_second, &probability_second);
      bool left_first_result =
	DualtreeKdeCanonical_(qnode->left(), rnode_first, probability_first);

      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(left_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool left_second_result = 
	DualtreeKdeCanonical_(qnode->left(), rnode_second, probability_second);
      
      // Fix the query node to be the right child, and recurse.
      BestNodePartners(qnode->right(), rnode->left(), rnode->right(), 
		       probability, &rnode_first, &probability_first,
		       &rnode_second, &probability_second);
      bool right_first_result =
	DualtreeKdeCanonical_(qnode->right(), rnode_first, probability_first);


      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(right_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool right_second_result =
	DualtreeKdeCanonical_(qnode->right(), rnode_second,
			      probability_second);

      result = left_first_result && left_second_result &&
	right_first_result && right_second_result;
    }
    
    // Reaccumulate the summary statistics.
    qnode->stat().RefineBoundStatistics(qnode->left()->stat(),
					qnode->right()->stat());
    return result;
  } // end of the case: non-leaf query node.

} // end of DualtreeKdeCanonical_

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::PreProcess(Tree *node) {

  // Initialize the center of expansions and bandwidth for series
  // expansion.
  Vector bounding_box_center;
  node->stat().Init(ka_);
  node->bound().CalculateMidpoint(&bounding_box_center);
  (node->stat().farfield_expansion_.get_center())->CopyValues
    (bounding_box_center);
  (node->stat().local_expansion_.get_center())->CopyValues
    (bounding_box_center);
  
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
    
    PreProcess(node->left());
    PreProcess(node->right());
    
    // Translate the multipole moments.
    node->stat().farfield_expansion_.TranslateFromFarField
      (node->left()->stat().farfield_expansion_);
    node->stat().farfield_expansion_.TranslateFromFarField
      (node->right()->stat().farfield_expansion_);
  }
  else {
    
    // Exhaustively compute multipole moments.
    node->stat().farfield_expansion_.RefineCoeffs(rset_, rset_weights_,
						  node->begin(), node->end(),
						  ka_.sea_.get_max_order());
  }
}

template<typename TKernelAux>
void DualtreeKde<TKernelAux>::PostProcess(Tree *qnode) {
    
  KdeStat<TKernelAux> &qstat = qnode->stat();
  
  // For a leaf query node,
  if(qnode->is_leaf()) {

    // Clear the summary statistics of the current query node so that
    // we can refine it to better bounds.
    qstat.ResetBoundStatistics();

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Add all postponed quantities.
      AddPostponed_(qnode, q);

      // Finally evaluate the local expansion and add in the
      // contributions.
      densities_e_[q] += qstat.local_expansion_.EvaluateField(qset_, q);
      
      // Normalize the densities.
      densities_l_[q] *= mult_const_;
      densities_e_[q] *= mult_const_;
      densities_u_[q] *= mult_const_;

      // Refine bound statistics using the finalized query point sum.
      RefineBoundStatistics_(q, qnode);
    }

    // Clear postponed approximations since they have been
    // incorporated.
    qstat.ClearPostponed();
  }
  else {
    
    // Push down approximations.
    qnode->left()->stat().AddPostponed(qstat);
    qnode->right()->stat().AddPostponed(qstat);

    // Translate local expansions.
    qstat.local_expansion_.TranslateToLocal
      (qnode->left()->stat().local_expansion_);
    qstat.local_expansion_.TranslateToLocal
      (qnode->right()->stat().local_expansion_);
    
    // Clear postponed approximations.
    qstat.ClearPostponed();

    // Recurse to the left and to the right.
    PostProcess(qnode->left());
    PostProcess(qnode->right());

    // Refine statistics after recursing.
    qstat.RefineBoundStatistics(qnode->left()->stat(), qnode->right()->stat());
  }
}
