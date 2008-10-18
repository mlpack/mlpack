#ifndef NWRCDE_STAT_H
#define NWRCDE_STAT_H

template<typename TKernelAux, typename TQueryPostponed, typename TQuerySummary>
class NWRCdeQueryStat {
 public:

  TQueryPostponed postponed;

  TQuerySummary summary;

  typename TKernelAux::TLocalExpansion nwr_numerator_local_expansion;
  
  typename TKernelAux::TLocalExpansion nwr_denominator_local_expansion;

  OT_DEF_BASIC(NWRCdeQueryStat) {
    OT_MY_OBJECT(postponed);
    OT_MY_OBJECT(summary);
    OT_MY_OBJECT(nwr_numerator_local_expansion);
    OT_MY_OBJECT(nwr_denominator_local_expansion);
  }

 public:

  void SetZero() {
    postponed.SetZero();
    summary.SetZero();
  }

  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const NWRCdeQueryStat& left_stat, 
	    const NWRCdeQueryStat& right_stat) {
  }

  void Init(const TKernelAux &kernel_aux_in) {
    nwr_numerator_local_expansion.Init(kernel_aux_in);
    nwr_denominator_local_expansion.Init(kernel_aux_in);
  }

  template<typename TBound>
  void Init(const TBound &bounding_primitive,
	    const TKernelAux &kernel_aux_in) {
 
    // Initialize the center of expansions and bandwidth for series
    // expansion.
    Vector bounding_box_center;
    Init(kernel_aux_in);
    bounding_primitive.CalculateMidpoint(&bounding_box_center);
    (nwr_numerator_local_expansion.get_center())->CopyValues
      (bounding_box_center);
    (nwr_denominator_local_expansion.get_center())->CopyValues
      (bounding_box_center);
   
    // Reset the postponed quantities to zero.
    SetZero();
  }   
};

template<typename TKernelAux>
class NWRCdeReferenceStat {
 public:

  /** @brief The far field expansion for the numerator created by the
   *         reference points in this node.
   */
  typename TKernelAux::TFarFieldExpansion nwr_numerator_farfield_expansion;

  /** @brief The far field expansion for the denominator created by
   *         the reference points in this node.
   */
  typename TKernelAux::TFarFieldExpansion nwr_denominator_farfield_expansion;

  double sum_of_target_values;

  OT_DEF_BASIC(NWRCdeReferenceStat) {
    OT_MY_OBJECT(nwr_numerator_farfield_expansion);
    OT_MY_OBJECT(nwr_denominator_farfield_expansion);
    OT_MY_OBJECT(sum_of_target_values);
  }

 public:

  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const NWRCdeReferenceStat& left_stat, 
	    const NWRCdeReferenceStat& right_stat) {
    
  }

  template<typename TBound>
  void PostInitCommon(const TBound &bounding_primitive,
		      const TKernelAux &kernel_aux_in) {
    
    // Initialize the center of expansions and bandwidth for series
    // expansion.
    Vector bounding_box_center;
    nwr_numerator_farfield_expansion.Init(kernel_aux_in);
    nwr_denominator_farfield_expansion.Init(kernel_aux_in);
    bounding_primitive.CalculateMidpoint(&bounding_box_center);
    (nwr_numerator_farfield_expansion.get_center())->CopyValues
      (bounding_box_center);
    (nwr_denominator_farfield_expansion.get_center())->CopyValues
      (bounding_box_center);
  }

  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for a leaf node.
   */
  template<typename TBound>
  void PostInit(const TBound &bounding_primitive,
		const TKernelAux &kernel_aux_in,
		const Matrix &reference_set, 
		const Vector &nwr_numerator_weights,
		const Vector &nwr_denominator_weights,
		index_t start, index_t count) {

    PostInitCommon(bounding_primitive, kernel_aux_in);

    // Exhaustively compute multipole moments.
    nwr_numerator_farfield_expansion.AccumulateCoeffs
      (reference_set, nwr_numerator_weights, start, start + count,
       kernel_aux_in.sea_.get_max_order());
    nwr_denominator_farfield_expansion.AccumulateCoeffs
      (reference_set, nwr_denominator_weights, start, start + count,
       kernel_aux_in.sea_.get_max_order());

    sum_of_target_values = 0;
    for(index_t i = start; i < start + count; i++) {
      sum_of_target_values += nwr_numerator_weights[i];
    }
  }
    
  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for an internal node.
   */
  template<typename TBound>
  void PostInit(const TBound &bounding_primitive,
		const TKernelAux &kernel_aux_in,
		const Matrix &reference_set,
		const Vector &nwr_numerator_weights,
		const Vector &nwr_denominator_weights, 
		index_t start, index_t count, 
		const NWRCdeReferenceStat& left_stat, 
		const NWRCdeReferenceStat& right_stat) {

    PostInitCommon(bounding_primitive, kernel_aux_in);
    
    // Translate the moments up from the two children's moments.
    nwr_numerator_farfield_expansion.TranslateFromFarField
      (left_stat.nwr_numerator_farfield_expansion);
    nwr_numerator_farfield_expansion.TranslateFromFarField
      (right_stat.nwr_numerator_farfield_expansion);
    nwr_denominator_farfield_expansion.TranslateFromFarField
      (left_stat.nwr_denominator_farfield_expansion);
    nwr_denominator_farfield_expansion.TranslateFromFarField
      (right_stat.nwr_denominator_farfield_expansion);
    
    sum_of_target_values = left_stat.sum_of_target_values +
      right_stat.sum_of_target_values;
  }

};

#endif
