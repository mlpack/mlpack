#ifndef DUALTREE_KDE_COMMON_H
#define DUALTREE_KDE_COMMON_H

class DualtreeKdeCommon {

 public:

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
