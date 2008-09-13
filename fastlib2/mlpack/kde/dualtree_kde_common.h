#ifndef DUALTREE_KDE_COMMON_H
#define DUALTREE_KDE_COMMON_H

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
  (Vector &v, const ArrayList<index_t> &permutation) {
    
    Vector v_tmp;
    v_tmp.Init(v.length());
    for(index_t i = 0; i < v_tmp.length(); i++) {
      v_tmp[i] = v[permutation[i]];
    }
    v.CopyValues(v_tmp);
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
};

#endif
