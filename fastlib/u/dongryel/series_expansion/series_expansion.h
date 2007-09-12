/**
 * @file series_expansion.h
 *
 * The header file for the series expansion.
 */

#ifndef SERIES_EXPANSION
#define SERIES_EXPANSION

#include <values.h>

#include "fastlib/fastlib.h"

#include "series_expansion_aux.h"

/**
 * Series expansion class.
 */
template<typename TKernel>
class SeriesExpansion {
  FORBID_COPY(SeriesExpansion);
  
 public:
  
  typedef TKernel Kernel;

  enum ExpansionType { FARFIELD, LOCAL };

 private:

  /** The type of the kernel */
  Kernel kernel_;

  /** The type of coefficients: far-field or local */
  ExpansionType expansion_type_;
  
  /** The center of the expansion */
  Vector center_;

  /** The coefficients */
  Vector coeffs_;

  /** order */
  int order_;
  
  /** precomputed quantities */
  SeriesExpansionAux *sea_;

 public:

  SeriesExpansion() {}
  
  ~SeriesExpansion() {}

  // getters and setters
  
  /** Get the coefficients */
  double bandwidth_sq() const { return kernel_.bandwidth_sq(); }

  /** Get the center of expansion */
  const Vector& get_center() const { return center_; }

  /** Get the coefficients */
  const Vector& get_coeffs() const { return coeffs_; }

  /** Get the expansion type */
  ExpansionType get_expansion_type() const { return expansion_type_; }

  /** Get the approximation order */
  int get_order() const { return order_; }

  // interesting functions...

  /**
   * Computes the far-field coefficients for the given data
   */
  void ComputeFarFieldCoeffs(const Matrix& data, const Vector& weights,
			     const ArrayList<int>& rows, int order);

  /**
   * Computes the local coefficients for the given data
   */
  void ComputeLocalCoeffs(const Matrix& data, const Vector& weights,
			  const ArrayList<int>& rows, int order);

  /**
   * Evaluates the far-field coefficients at the given point
   */
  double EvaluateFarField(Matrix* data=NULL, int row_num=-1,
			  Vector* point=NULL);

  /**
   * Evaluates the local-field coefficients at the given point
   */
  double EvaluateLocalField(Matrix* data=NULL, int row_num=-1,
			    Vector* point=NULL);

  /**
   * Initializes the current SeriesExpansion object with the given
   * center.
   */
  void Init(const TKernel& kernel, ExpansionType expansion_type,
	    const Vector& center, SeriesExpansionAux *sea);

  /**
   * Prints out the series expansion represented by this object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

  /**
   * Far-field to Far-field translation operator: translates the given
   * far-field expansion to the new center
   */
  void TransFarToFar(const SeriesExpansion &se);

  /**
   * Far-field to local translation operator: translates the given far
   * expansion to the local expansion at the new center.
   */
  void TransFarToLocal(const SeriesExpansion &se);

  /**
   * Local to local translation operator: translates the given local
   * expansion to the local expansion at the new center.
   */
  void TransLocalToLocal(const SeriesExpansion &se);

};

template<typename TKernel>
void SeriesExpansion<TKernel>::ComputeFarFieldCoeffs
(const Matrix& data, const Vector& weights,
 const ArrayList<int>& rows, int order) {

  int dim = data.n_rows();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector tmp;
  int num_rows = rows.size();
  int r, i, j, k, t, tail;
  Vector heads;
  Vector x_r;
  Vector C_k;
  double bw_times_sqrt_two = sqrt(2 * kernel_.bandwidth_sq());

  // initialize temporary variables
  tmp.Init(total_num_coeffs);
  heads.Init(dim + 1);
  x_r.Init(dim);

  // If we have more than what we need, return.
  if(order_ >= order) {
    return;
  }
  else {
    order_ = order;
  }
    
  // Repeat for each reference point in this reference node.
  for(r = 0; r < num_rows; r++) {
    
    // get the row number.
    int row_num = rows[r];
    
    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, row_num) - center_[i]) / bw_times_sqrt_two;
    }

    // initialize heads
    heads.SetZero();
    heads[dim] = MAXINT;
    
    tmp[0] = 1.0;
    
    for(k = 1, t = 1, tail = 1; k <= order; k++, tail = t) {
      for(i = 0; i < dim; i++) {
	int head = (int) heads[i];
	heads[i] = t;
	
	for(j = head; j < tail; j++, t++) {
	  tmp[t] = tmp[j] * x_r[i];
	}
      }
    }
    
    // Tally up the result in A_k.
    for(i = 0; i < total_num_coeffs; i++) {
      coeffs_[i] += weights[row_num] * tmp[i];
    }
    
  } // End of looping through each reference point

  // get multiindex factors
  C_k.Alias(sea_->get_inv_multiindex_factorials());

  for(r = 1; r < total_num_coeffs; r++) {
    coeffs_[r] = coeffs_[r] * C_k[r];
  }
}

template<typename TKernel>
void SeriesExpansion<TKernel>::ComputeLocalCoeffs
(const Matrix& data, const Vector& weights, const ArrayList<int>& rows, 
 int order) {


  if(order > order_) {
    order_ = order;
  }

  int dim = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  
  // get inverse factorials (precomputed)
  Vector neg_inv_multiindex_factorials;
  neg_inv_multiindex_factorials.Alias
    (sea_->get_neg_inv_multiindex_factorials());

  // declare hermite mapping
  Matrix hermite_map;
  hermite_map.Init(dim, order + 1);
  
  // some temporary variables
  Vector arrtmp;
  arrtmp.Init(total_num_coeffs);
  Vector x_r_minus_x_Q;
  x_r_minus_x_Q.Init(dim);

  // sqrt two times bandwidth
  double sqrt_two_bandwidth = sqrt(2 * kernel_.bandwidth_sq());

  // for each data point,
  for(index_t r = 0; r < rows.size(); r++) {

    // get the row number
    int row_num = rows[r];

    // calculate x_r - x_Q
    for(index_t d = 0; d < dim; d++) {
      x_r_minus_x_Q[d] = (center_[d] - data.get(d, row_num)) / 
	sqrt_two_bandwidth;
    }
    
    // precompute necessary Hermite polynomials based on coordinate difference
    for(index_t d = 0; d < dim; d++) {

      double coord_div_band = x_r_minus_x_Q[d];
      double d2 = 2 * coord_div_band;
      double facj = exp(-coord_div_band * coord_div_band);
      
      hermite_map.set(d, 0, facj);
      
      if(order > 0) {

	hermite_map.set(d, 1, d2 * facj);
	
	if(order > 1) {
	  for(index_t k = 1; k < order; k++) {
	    int k2 = k * 2;
	    hermite_map.set(d, k + 1, d2 * hermite_map.get(d, k) -
			    k2 * hermite_map.get(d, k - 1));
	  }
	}
      }
    } // end of looping over each dimension
    
    // compute h_{beta}((x_r - x_Q) / sqrt(2h^2))
    for(index_t j = 0; j < total_num_coeffs; j++) {
      ArrayList<int> mapping = sea_->get_multiindex(j);
      arrtmp[j] = 1.0;

      for(index_t d = 0; d < dim; d++) {
        arrtmp[j] *= hermite_map.get(d, mapping[d]);
      }
    }

    for(index_t j = 0; j < total_num_coeffs; j++) {
      coeffs_[j] += neg_inv_multiindex_factorials[j] * weights[row_num] * 
	arrtmp[j];
    }
  } // End of looping through each reference point.
}

template<typename TKernel>
double SeriesExpansion<TKernel>::EvaluateFarField
(Matrix* data, int row_num, Vector* x_q) {
  
  // dimension
  int dim = sea_->get_dimension();

  // total number of coefficients
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  // square root times bandwidth
  double sqrt_two_bandwidth = sqrt(2 * kernel_.bandwidth_sq());
  
  // the evaluated sum
  double multipole_sum = 0;
  
  // computed derivative map
  Matrix derivative_map;
  derivative_map.Init(dim, order_ + 1);

  // temporary variable
  Vector arrtmp;
  arrtmp.Init(total_num_coeffs);

  // (x_q - x_R) scaled by bandwidth
  Vector x_q_minus_x_R;
  x_q_minus_x_R.Init(dim);

  // compute (x_q - x_R) / (sqrt(2h^2))
  for(index_t d = 0; d < dim; d++) {
    if(x_q == NULL) {
      x_q_minus_x_R[d] = (data->get(d, row_num) - center_[d]) / 
	sqrt_two_bandwidth;
    }
    else {
      x_q_minus_x_R[d] = ((*x_q)[d] - center_[d]) / sqrt_two_bandwidth;
    }
  }

  // compute deriative maps based on coordinate difference.
  for(index_t d = 0; d < dim; d++) {
    double coord_div_band = x_q_minus_x_R[d];
    double d2 = 2 * coord_div_band;
    double facj = exp(-coord_div_band * coord_div_band);

    derivative_map.set(d, 0, facj);

    if(order_ > 0) {
      derivative_map.set(d, 1, d2 * facj);
    
      if(order_ > 1) {
	for(index_t k = 1; k < order_; k++) {
	  int k2 = k * 2;
	  derivative_map.set(d, k + 1, d2 * derivative_map.get(d, k) -
			     k2 * derivative_map.get(d, k - 1));
	}
      }
    }
  }

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2))
  for(index_t j = 0; j < total_num_coeffs; j++) {
    ArrayList<int> mapping = sea_->get_multiindex(j);
    arrtmp[j] = 1.0;
    
    for(index_t d = 0; d < dim; d++) {
      arrtmp[j] *= derivative_map.get(d, mapping[d]);
    }
  }
  
  // tally up the multipole sum
  for(index_t j = 0; j < total_num_coeffs; j++) {
    multipole_sum += coeffs_[j] * arrtmp[j];
  }

  return multipole_sum;
}

template<typename TKernel>
double SeriesExpansion<TKernel>::EvaluateLocalField
(Matrix* data, int row_num, Vector* x_q) {

  index_t k, t, tail;
  
  // total number of coefficient
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  // number of dimensions
  int dim = sea_->get_dimension();

  // evaluated sum to be returned
  double sum = 0;
  
  // sqrt two bandwidth
  double sqrt_two_bandwidth = sqrt(2 * kernel_.bandwidth_sq());

  // temporary variable
  Vector x_Q_to_x_q;
  x_Q_to_x_q.Init(dim);
  Vector tmp;
  tmp.Init(total_num_coeffs);
  ArrayList<int> heads;
  heads.Init(dim + 1);
  
  // compute (x_q - x_Q) / (sqrt(2h^2))
  for(index_t i = 0; i < dim; i++) {
    
    if(data == NULL) {
      x_Q_to_x_q[i] = ((*x_q)[i] - center_[i]) / sqrt_two_bandwidth;
    }
    else {
      x_Q_to_x_q[i] = (data->get(i, row_num) - center_[i]) / 
	sqrt_two_bandwidth;
    }
  }
  
  for(index_t i = 0; i < dim; i++)
    heads[i] = 0;
  heads[dim] = MAXINT;

  tmp[0] = 1.0;

  for(k = 1, t = 1, tail = 1; k <= order_; k++, tail = t) {

    for(index_t i = 0; i < dim; i++) {
      int head = heads[i];
      heads[i] = t;

      for(index_t j = head; j < tail; j++, t++) {
        tmp[t] = tmp[j] * x_Q_to_x_q[i];
      }
    }
  }

  for(index_t i = 0; i < total_num_coeffs; i++) {
    sum += coeffs_[i] * tmp[i];
  }

  return sum;
}

template<typename TKernel>
void SeriesExpansion<TKernel>::Init(const TKernel &kernel,
				    ExpansionType expansion_type, 
				    const Vector& center, 
				    SeriesExpansionAux* sea) {

  // copy kernel type, center, and bandwidth squared
  kernel_.Init(sqrt(kernel.bandwidth_sq()));
  expansion_type_ = expansion_type;
  center_.Copy(center);
  order_ = 0;
  sea_ = sea;

  // initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernel>
void SeriesExpansion<TKernel>::PrintDebug(const char *name, 
					  FILE *stream) const {
  
  int dim = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Expansion type: %s\n", (expansion_type_ == FARFIELD) ?
	  "FARFIELD":"LOCAL");
  fprintf(stream, "Center: ");
  
  for (index_t i = 0; i < center_.length(); i++) {
    fprintf(stream, "%g ", center_[i]);
  }
  fprintf(stream, "\n");
  
  fprintf(stream, "f(");
  for(index_t d = 0; d < dim; d++) {
    fprintf(stream, "x_q%d", d);
    if(d < dim - 1)
      fprintf(stream, ",");
  }
  fprintf(stream, ") = \\sum\\limits_{x_r \\in R} K(||x_q - x_r||) = ");
  
  for (index_t i = 0; i < total_num_coeffs; i++) {
    ArrayList<int> mapping = sea_->get_multiindex(i);
    if(expansion_type_ == LOCAL) {
      fprintf(stream, "%g", coeffs_[i]);
      
      for(index_t d = 0; d < dim; d++) {
	fprintf(stream, "(x_q%d - (%g))^%d ", d, center_[d], mapping[d]);
      }
    }
    else {
      fprintf(stream, "%g ", coeffs_[i]);
      
      fprintf(stream, "(-1)^(");
      for(index_t d = 0; d < dim; d++) {
	fprintf(stream, "%d", mapping[d]);
	if(d < dim - 1)
	  fprintf(stream, " + ");
      }
      fprintf(stream, ") D^((");
      for(index_t d = 0; d < dim; d++) {
	fprintf(stream, "%d", mapping[d]);
	
	if(d < dim - 1)
	  fprintf(stream, ",");
      }
      fprintf(stream, "))");
      for(index_t d = 0; d < dim; d++) {
	fprintf(stream, "(x_q%d - (%g))^%d ", d, center_[d], mapping[d]);
      }
    }
    if(i < total_num_coeffs - 1) {
      fprintf(stream, " + ");
    }
  }
  fprintf(stream, "\n");
}

template<typename TKernel>
void SeriesExpansion<TKernel>::TransFarToFar(const SeriesExpansion &se) {

  double sqrt_two_bandwidth = sqrt(2 * se.bandwidth_sq());
  int dim = sea_->get_dimension();
  int order = se.get_order();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector prev_coeffs;
  Vector prev_center;
  const ArrayList < int > *multiindex_mapping = sea_->get_multiindex_mapping();
  const ArrayList < int > *lower_mapping_index = 
    sea_->get_lower_mapping_index();

  ArrayList <int> tmp_storage;
  Vector center_diff;
  Vector inv_multiindex_factorials;

  center_diff.Init(dim);

  // retrieve coefficients to be translated and helper mappings
  prev_coeffs.Alias(se.get_coeffs());
  prev_center.Alias(se.get_center());
  tmp_storage.Init(sea_->get_dimension());
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // no coefficients can be translated
  if(order == 0)
    return;
  else
    order_ = order;
  
  // the first order (the sum of the weights) stays constant regardless
  // of the location of the center.
  coeffs_.SetZero();
  coeffs_[0] = prev_coeffs[0];
  
  // compute center difference
  for(index_t j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  for(index_t j = 1; j < total_num_coeffs; j++) {
    
    ArrayList <int> gamma_mapping = multiindex_mapping[j];
    ArrayList <int> lower_mappings_for_gamma = lower_mapping_index[j];

    for(index_t k = 0; k < lower_mappings_for_gamma.size(); k++) {

      ArrayList <int> inner_mapping = 
	multiindex_mapping[lower_mappings_for_gamma[k]];

      int flag = 0;
      double diff1;
      
      // compute gamma minus alpha
      for(index_t l = 0; l < dim; l++) {
	tmp_storage[l] = gamma_mapping[l] - inner_mapping[l];

	if(tmp_storage[l] < 0) {
	  flag = 1;
	  break;
	}
      }
      
      if(flag) {
	continue;
      }
      
      diff1 = 1.0;
      
      for(index_t l = 0; l < dim; l++) {

	diff1 *= pow(center_diff[l] / sqrt_two_bandwidth, tmp_storage[l]);
      }

      coeffs_[j] += prev_coeffs[lower_mappings_for_gamma[k]] * diff1 * 
	inv_multiindex_factorials
	[sea_->ComputeMultiindexPosition(tmp_storage)];

    } // end of k-loop
  } // end of j-loop
}

template<typename TKernel>
void SeriesExpansion<TKernel>::TransFarToLocal(const SeriesExpansion &se) {

  Vector arrtmp;
  Matrix hermite_map;
  Vector far_center;
  Vector far_coeffs;
  int dimension = sea_->get_dimension();
  int far_order = se.get_order();
  int total_num_coeffs = sea_->get_total_num_coeffs(far_order);
  int limit;
  double bw_times_sqrt_two = sqrt(2 * kernel_.bandwidth_sq());

  // get center and coefficients for far field expansion
  far_center.Alias(se.get_center());
  far_coeffs.Alias(se.get_coeffs());

  // if the order of the far field expansion is greater than the
  // local one we are adding onto, then increase the order.
  if(far_order > order_) {
    order_ = far_order;
  }

  // compute Gaussian derivative
  limit = 2 * order_ + 1;
  hermite_map.Init(dimension, limit);
  arrtmp.Init(total_num_coeffs);
  
  for(index_t j = 0; j < dimension; j++) {
    double coord_div_band = (center_[j] - far_center[j]) / bw_times_sqrt_two;
    double d2 = 2 * coord_div_band;
    double facj = exp(-coord_div_band * coord_div_band);

    hermite_map.set(j, 0, facj);
    
    if(order_ > 0) {
      hermite_map.set(j, 1, d2 * facj);

      for(index_t k = 1; k < limit; k++) {
	int k2 = k * 2;
	hermite_map.set(j, k + 1, d2 * hermite_map.get(j, k) - k2 * 
			hermite_map.get(j, k - 1));
      } // end of k loop
    }
  } // end of j-loop

  for(index_t j = 0; j < total_num_coeffs; j++) {

    ArrayList<int> beta_mapping = sea_->get_multiindex(j);
    arrtmp[j] = 0;

    for(index_t k = 0; k < total_num_coeffs; k++) {

      ArrayList<int> alpha_mapping = sea_->get_multiindex(k);
      double hermite_factor = 1.0;

      for(index_t d = 0; d < dimension; d++) {
	hermite_factor *= 
	  hermite_map.get(d, beta_mapping[d] + alpha_mapping[d]);
      }
      
      arrtmp[j] += far_coeffs[k] * hermite_factor;

    } // end of k-loop
  } // end of j-loop

  Vector C_k_neg = sea_->get_neg_inv_multiindex_factorials();
  for(index_t j = 0; j < total_num_coeffs; j++) {
    coeffs_[j] += arrtmp[j] * C_k_neg[j];
  }
}

template<typename TKernel>
void SeriesExpansion<TKernel>::TransLocalToLocal(const SeriesExpansion &se) {
  
  // get the center and the order and the total number of coefficients of 
  // the expansion we are translating from. Also get coefficients we
  // are translating
  Vector prev_center;
  prev_center.Alias(se.get_center());
  int prev_order = se.get_order();
  int total_num_coeffs = sea_->get_total_num_coeffs(prev_order);
  const ArrayList < int > *upper_mapping_index = 
    sea_->get_upper_mapping_index();
  Vector prev_coeffs;
  prev_coeffs.Alias(se.get_coeffs());

  // dimension
  int dim = sea_->get_dimension();

  // temporary variable
  ArrayList<int> tmp_storage;
  tmp_storage.Init(dim);

  // sqrt two times bandwidth
  double sqrt_two_bandwidth = sqrt(2 * kernel_.bandwidth_sq());

  // center difference between the old center and the new one
  Vector center_diff;
  center_diff.Init(dim);
  for(index_t d = 0; d < dim; d++) {
    center_diff[d] = (center_[d] - prev_center[d]) / sqrt_two_bandwidth;
  }

  // set to the new order if the order of the expansion we are translating
  // from is higher
  if(prev_order > order_) {
    order_ = prev_order;
  }

  // inverse multiindex factorials
  Vector C_k;
  C_k.Alias(sea_->get_inv_multiindex_factorials());
  
  // do the actual translation
  for(index_t j = 0; j < total_num_coeffs; j++) {

    ArrayList<int> alpha_mapping = sea_->get_multiindex(j);
    ArrayList <int> upper_mappings_for_alpha = upper_mapping_index[j];

    for(index_t k = 0; k < upper_mappings_for_alpha.size(); k++) {
      
      if(upper_mappings_for_alpha[k] >= total_num_coeffs) {
	break;
      }

      ArrayList<int> beta_mapping = 
	sea_->get_multiindex(upper_mappings_for_alpha[k]);
      int flag = 0;
      double diff1 = 1.0;

      for(index_t l = 0; l < dim; l++) {
	tmp_storage[l] = beta_mapping[l] - alpha_mapping[l];

	if(tmp_storage[l] < 0) {
	  flag = 1;
	  break;
	}
      } // end of looping over dimension
      
      if(flag)
	continue;

      for(index_t l = 0; l < dim; l++) {
	diff1 *= pow(center_diff[l], tmp_storage[l]);
      }
      coeffs_[j] += prev_coeffs[upper_mappings_for_alpha[k]] * diff1 *
	sea_->get_n_multichoose_k_by_pos(upper_mappings_for_alpha[k], j);

    } // end of k loop
  } // end of j loop
}

#endif
