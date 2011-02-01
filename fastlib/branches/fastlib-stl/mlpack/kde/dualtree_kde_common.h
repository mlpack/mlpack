#ifndef DUALTREE_KDE_COMMON_H
#define DUALTREE_KDE_COMMON_H

#include "inverse_normal_cdf.h"

class DualtreeKdeCommon {

 public:

  /** @brief The comparison function used for the quick-sort.
   */
  static int qsort_comparator(const void *a, const void *b) {
    double *typecasted_a = (double *) a;
    double *typecasted_b = (double *) b;
    
    if(*typecasted_a < *typecasted_b) {
      return -1;
    }
    else if(*typecasted_a > *typecasted_b) {
      return 1;
    }
    else {
      return 0;
    }
  }
 
  /** @brief Adds the postponed information for the given query node
   *         to a given query point contribution.
   *
   *  @param node The query node.
   *  @param destination The query index destination.
   *  @param kde_object The KDE computation object that contains the query 
   *                    point information.
   */
  template<typename TTree, typename TAlgorithm>
  static void AddPostponed(TTree *node, index_t destination, 
			   TAlgorithm *kde_object) {

    kde_object->densities_l_[destination] += node->stat().postponed_l_;
    kde_object->densities_e_[destination] += node->stat().postponed_e_;
    kde_object->densities_u_[destination] += node->stat().postponed_u_;
    kde_object->used_error_[destination] += node->stat().postponed_used_error_;
    kde_object->n_pruned_[destination] += node->stat().postponed_n_pruned_; 
  }

  template<typename TTree>
  static void BestNodePartners
  (TTree *nd, TTree *nd1, TTree *nd2, double probability, 
   TTree **partner1, double *probability1, 
   TTree **partner2, double *probability2) {
  
  double d1 = nd->bound().MinDistanceSq(nd1->bound());
  double d2 = nd->bound().MinDistanceSq(nd2->bound());

  // Prioritized traversal based on the squared distance bounds.
  if(d1 <= d2) {
    *partner1 = nd1;
    *probability1 = (probability);
    *partner2 = nd2;
    *probability2 = (probability);
  }
  else {
    *partner1 = nd2;
    *probability1 = (probability);
    *partner2 = nd1;
    *probability2 = (probability);
  }
}

  /** @brief Refines the summary statistics of the given query node
   *         using a newly updated information for the query point
   *         $q$.
   *
   *  @param q The query point index.
   *  @param qnode The query node.
   *  @param kde_object The KDE computation object that contains the
   *                    query point information.
   */
  template<typename TTree, typename TAlgorithm>
  static void RefineBoundStatistics(index_t q, TTree *qnode, 
				    TAlgorithm *kde_object) {
    
    qnode->stat().mass_l_ = std::min(qnode->stat().mass_l_, 
				     kde_object->densities_l_[q]);
    qnode->stat().mass_u_ = std::max(qnode->stat().mass_u_, 
				     kde_object->densities_u_[q]);
    qnode->stat().used_error_ = std::max(qnode->stat().used_error_,
					 kde_object->used_error_[q]);
    qnode->stat().n_pruned_ = std::min(qnode->stat().n_pruned_, 
				       kde_object->n_pruned_[q]);
  }

  /** @brief Shuffles a vector according to a given permutation.
   *
   *  @param v The vector to be shuffled.
   *  @param permutation The permutation.
   */
  static void ShuffleAccordingToPermutation
  (arma::vec& v, const arma::Col<index_t> &permutation) {
    
    arma::vec v_tmp(v.n_elem);
    for(index_t i = 0; i < v_tmp.n_elem; i++) {
      v_tmp[i] = v[permutation[i]];
    }
    v = v_tmp;
  }

  static double OuterConfidenceInterval
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
  
  static double BinomialCoefficientHelper(double n3, double k3, double n1, 
					  double k1, double n2, double k2) {
    
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

  template<typename TTree, typename TAlgorithm>
  static bool MonteCarloPrunable_
  (TTree *qnode, TTree *rnode, double probability, DRange &dsqd_range,
   DRange &kernel_value_range, double &dl, double &de, double &du, 
   double &used_error, double &n_pruned, TAlgorithm *kde_object) {
    
    // If the reference node contains too few points, then return.
    if(qnode->count() * rnode->count() < 50) {
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
	    (kde_object->qset_).colptr(random_query_point_index);
	  
	  // Get the pointer to the current reference point.
	  const double *reference_point = 
	    (kde_object->rset_).colptr(random_reference_point_index);
	  
	  // Compute the pairwise distance and kernel value.
	  double squared_distance = la::DistanceSqEuclidean
	    ((kde_object->rset_).n_rows, query_point, reference_point);
	  
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
  static bool Prunable(TTree *qnode, TTree *rnode, double probability, 
		       DRange &dsqd_range, DRange &kernel_value_range, 
		       double &dl, double &de, double &du, 
		       double &used_error, double &n_pruned, 
		       TAlgorithm *kde_object) {
    
    // the new lower bound after incorporating new info
    dl = kernel_value_range.lo * rnode->stat().get_weight_sum();
    de = 0.5 * rnode->stat().get_weight_sum() * 
      (kernel_value_range.lo + kernel_value_range.hi);
    du = (kernel_value_range.hi - 1) * rnode->stat().get_weight_sum();
    
    // refine the lower bound using the new lower bound info
    double new_mass_l = qnode->stat().mass_l_ + 
      qnode->stat().postponed_l_ + dl;
    double new_used_error = qnode->stat().used_error_ + 
      qnode->stat().postponed_used_error_;
    double new_n_pruned = qnode->stat().n_pruned_ + 
      qnode->stat().postponed_n_pruned_;
    
    double allowed_err;
    
    // Compute the allowed error.
    allowed_err = (kde_object->relative_error_ * new_mass_l - new_used_error) *
      rnode->stat().get_weight_sum() / 
      ((double) kde_object->rroot_->stat().get_weight_sum() - new_n_pruned);
    
    // This is error per each query/reference pair for a fixed query
    double kernel_diff = 0.5 * (kernel_value_range.hi - kernel_value_range.lo);
    
    // this is total error for each query point
    used_error = kernel_diff * rnode->stat().get_weight_sum();
    
    // number of reference points for possible pruning.
    n_pruned = rnode->stat().get_weight_sum();
    
    // If the error bound is satisfied by the hard error bound, it is
    // safe to prune.
    return (!isnan(allowed_err)) && (used_error <= allowed_err);
  }

};

#endif
