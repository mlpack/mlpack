#ifndef INSIDE_KDE_PROBLEM_H
#error "This is not a public header file!"
#endif

class MultiTreeQueryStat {
 public:

  MultiTreeQueryPostponed postponed;
  
  MultiTreeQuerySummary summary;
  
  typename TKernelAux::TLocalExpansion local_expansion;
  
  double priority;
  
  bool in_strata;

  double num_precomputed_tuples;

  OT_DEF_BASIC(MultiTreeQueryStat) {
    OT_MY_OBJECT(postponed);
    OT_MY_OBJECT(summary);
    OT_MY_OBJECT(local_expansion);
    OT_MY_OBJECT(priority);
    OT_MY_OBJECT(in_strata);
    OT_MY_OBJECT(num_precomputed_tuples);
  }

 public:
  
  void FinalPush(MultiTreeQueryStat &child_stat) {
    child_stat.postponed.ApplyPostponed(postponed);
    local_expansion.TranslateToLocal(child_stat.local_expansion);
  }
  
  void SetZero() {
    postponed.SetZero();
    summary.SetZero();
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const MultiTreeQueryStat& left_stat,
	    const MultiTreeQueryStat& right_stat) {
  }
    
  void Init(const TKernelAux &kernel_aux_in) {
    local_expansion.Init(kernel_aux_in);
  }
    
  template<typename TBound>
  void Init(const TBound &bounding_primitive,
	    const TKernelAux &kernel_aux_in) {
      
    // Initialize the center of expansions and bandwidth for series
    // expansion.
    Vector bounding_box_center;
    Init(kernel_aux_in);
    bounding_primitive.CalculateMidpoint(&bounding_box_center);
    (local_expansion.get_center())->CopyValues(bounding_box_center);
      
    // Reset the postponed quantities to zero.
    SetZero();
  }
};

class MultiTreeReferenceStat {
 public:

  /** @brief The far field expansion for the numerator created by the
   *         reference points in this node.
   */
  typename TKernelAux::TFarFieldExpansion farfield_expansion;
    
  OT_DEF_BASIC(MultiTreeReferenceStat) {
    OT_MY_OBJECT(farfield_expansion);
  }
    
 public:
    
  double get_weight_sum() {
    return farfield_expansion.get_weight_sum();
  }
  
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const MultiTreeReferenceStat& left_stat,
	    const MultiTreeReferenceStat& right_stat) {
      
  }
    
  template<typename TBound>
  void PostInitCommon(const TBound &bounding_primitive,
		      const TKernelAux &kernel_aux_in) {
      
    // Initialize the center of expansions and bandwidth for series
    // expansion.
    Vector bounding_box_center;
    farfield_expansion.Init(kernel_aux_in);
    bounding_primitive.CalculateMidpoint(&bounding_box_center);
    (farfield_expansion.get_center())->CopyValues(bounding_box_center);
  }
    
  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for a leaf node.
   */
  template<typename TBound>
  void PostInit(const TBound &bounding_primitive,
		const TKernelAux &kernel_aux_in,
		const ArrayList<Matrix *> &reference_sets,
		const ArrayList<Matrix *> &targets,
		index_t start, index_t count) {
      
    PostInitCommon(bounding_primitive, kernel_aux_in);
      
    // Exhaustively compute multipole moments.
    const Matrix &reference_set = *(reference_sets[0]);
    const Matrix &targets_dereferenced = *(targets[0]);
    Vector nwr_numerator_weights_alias;
    targets_dereferenced.MakeColumnVector(0, &nwr_numerator_weights_alias);

    farfield_expansion.AccumulateCoeffs
      (reference_set, nwr_numerator_weights_alias, start, start + count,
       kernel_aux_in.sea_.get_max_order());
  }
    
  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for an internal node.
   */
  template<typename TBound>
  void PostInit(const TBound &bounding_primitive,
		const TKernelAux &kernel_aux_in,
		const ArrayList<Matrix *> &reference_sets,
		const ArrayList<Matrix *> &targets,
		index_t start, index_t count,
		const MultiTreeReferenceStat& left_stat,
		const MultiTreeReferenceStat& right_stat) {
      
    PostInitCommon(bounding_primitive, kernel_aux_in);
      
    // Translate the moments up from the two children's moments.
    farfield_expansion.TranslateFromFarField(left_stat.farfield_expansion);
    farfield_expansion.TranslateFromFarField(right_stat.farfield_expansion);

  }
    
};
