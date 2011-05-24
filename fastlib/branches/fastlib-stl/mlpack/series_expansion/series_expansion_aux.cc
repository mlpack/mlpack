#include "series_expansion_aux.h"

const arma::vec& SeriesExpansionAux::get_inv_multiindex_factorials() const {
  return inv_multiindex_factorials_;
}

index_t SeriesExpansionAux::get_max_total_num_coeffs() const {
  return list_total_num_coeffs_[max_order_];
}

const std::vector < std::vector<index_t> >& SeriesExpansionAux::get_lower_mapping_index() 
  const {
  return lower_mapping_index_;
}

index_t SeriesExpansionAux::get_max_order() const {
  return max_order_;
}

const std::vector < index_t > &SeriesExpansionAux::get_multiindex(index_t pos) 
  const {
  return multiindex_mapping_[pos];
}

const std::vector < std::vector<index_t> >& SeriesExpansionAux::get_multiindex_mapping() 
  const {
  return multiindex_mapping_;
}

const arma::vec& SeriesExpansionAux::get_neg_inv_multiindex_factorials() const {
  return neg_inv_multiindex_factorials_;
}

double SeriesExpansionAux::get_n_choose_k(index_t n, index_t k) const {
  return n_choose_k_(n, (index_t) math::ClampNonNegative(k));
}

double SeriesExpansionAux::get_n_multichoose_k_by_pos(index_t n, index_t k) const {
  return multiindex_combination_(n, k);
}

index_t SeriesExpansionAux::get_total_num_coeffs(index_t order) const {

  return list_total_num_coeffs_[order];
}

const std::vector < std::vector< index_t > >& SeriesExpansionAux::get_upper_mapping_index() 
  const {

  return upper_mapping_index_;
}

index_t SeriesExpansionAux::ComputeMultiindexPosition
(const std::vector<index_t> &multiindex) const {

  index_t dim = multiindex.size();
  index_t mapping_sum = 0;
  index_t index = 0;

  for(index_t j = 0; j < dim; j++) {

    // If any of the index is negative, then it does not exist!
    if(multiindex[j] < 0) {
      index = -1;
      break;
    }
    
    mapping_sum += multiindex[j];
  }
  if(index >= 0) {
    for(index_t j = 0; j < dim; j++) {
      index += (index_t) get_n_choose_k(mapping_sum + dim - j - 1, dim - j);
      mapping_sum -= multiindex[j];
    }
  }
  
  return index;
}

double SeriesExpansionAux::FarFieldEvaluationCost(index_t order) const {
  return pow(dim_, order + 1);
}

double SeriesExpansionAux::FarFieldToLocalTranslationCost(index_t order) const {
  return pow(dim_, 2 * order + 1);
}

double SeriesExpansionAux::DirectLocalAccumulationCost(index_t order) const {
  return pow(dim_, order + 1);
}

void SeriesExpansionAux::Init(index_t max_order, index_t dim) {

  index_t p, k, t, tail, i, j;
  std::vector<index_t> heads;
  std::vector<index_t> cinds;

  // initialize max order and dimension
  dim_ = dim;
  max_order_ = max_order;
  
  // compute the list of total number of coefficients for p-th order expansion
  index_t limit = 2 * max_order + 1;
  list_total_num_coeffs_.reserve(limit);
  for(p = 0; p < limit; p++) {
    list_total_num_coeffs_[p] = (index_t) math::BinomialCoefficient(p + dim, dim);
  }

  // compute factorials
  ComputeFactorials();

  // allocate space for inverse factorial and 
  // negative inverse factorials and multiindex mapping and n_choose_k 
  // and multiindex_combination precomputed factors
  inv_multiindex_factorials_.set_size(list_total_num_coeffs_[limit - 1]);  
  neg_inv_multiindex_factorials_.set_size(list_total_num_coeffs_[limit - 1]);
  multiindex_mapping_.reserve(list_total_num_coeffs_[limit - 1]);
  (multiindex_mapping_[0]).reserve(dim_);
  for(j = 0; j < dim; j++) {
    (multiindex_mapping_[0])[j] = 0;
  }
  n_choose_k_.zeros((limit - 1) + dim + 1, (limit - 1) + dim + 1);

  // initialization of temporary variables for computation...
  heads.reserve(dim + 1);
  cinds.reserve(list_total_num_coeffs_[limit - 1]);

  for(i = 0; i < dim; i++) {
    heads[i] = 0;
  }
  heads[dim] = INT_MAX;
  cinds[0] = 0;
  
  // compute inverse factorial and negative inverse factorials and
  // multiindex mappings...
  inv_multiindex_factorials_[0] = 1.0;
  neg_inv_multiindex_factorials_[0] = 1.0;
  for(k = 1, t = 1, tail = 1; k <= 2 * max_order_; k++, tail = t) {
    for(i = 0; i < dim; i++) {
      index_t head = (index_t) heads[i];
      heads[i] = t;
      for(j = head; j < tail; j++, t++) {
	cinds[t] = (j < heads[i + 1]) ? cinds[j] + 1 : 1;
	inv_multiindex_factorials_[t] = 
	  inv_multiindex_factorials_[j] / cinds[t];
	neg_inv_multiindex_factorials_[t] =
	  -neg_inv_multiindex_factorials_[j] / cinds[t];
	
	(multiindex_mapping_[t]).assign(multiindex_mapping_[j].begin(), multiindex_mapping_[j].end());
	(multiindex_mapping_[t])[i] = (multiindex_mapping_[t])[i] + 1;
      }
    }
  }

  // compute n choose k's
  for(j = 0; j <= 2 * max_order + dim; j++) {
    for(k = 0; k <= 2 * max_order + dim; k++) {
      n_choose_k_(j, k) = math::BinomialCoefficient(j, k);
    }
  }

  // initialize multiindex_combination matrix beta choose alpha
  ComputeMultiindexCombination();

  // compute the lower_mapping_index_ and the upper_mapping_index_
  // (see series_expansion_aux.h for explanation)
  ComputeLowerMappingIndex();
  ComputeUpperMappingIndex();
}

void SeriesExpansionAux::PrintDebug(const char *name, FILE *stream) const {

  fprintf(stream, "----- SERIESEXPANSIONAUX %s ------\n", name);
  fprintf(stream, "Max order: %"LI", dimension: %"LI"\n", max_order_, dim_);

  fprintf(stream, "Multiindex mapping: ");
  for (index_t i = 0; i < multiindex_mapping_.size(); i++) {

    DEBUG_ASSERT_MSG(ComputeMultiindexPosition(multiindex_mapping_[i]) == i,
		     "REIMPLEMENT ComputeMultiindexPosition function!");
    fprintf(stream, "( ");
    for(index_t j = 0; j < dim_; j++) {
      fprintf(stream, "%"LI" ", multiindex_mapping_[i][j]);
    }
    fprintf(stream, "): %g %g ", inv_multiindex_factorials_[i],
	    neg_inv_multiindex_factorials_[i]);
  }
  fprintf(stream, "\n");
}
