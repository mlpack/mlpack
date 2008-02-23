// Make sure this file is included only in krylov_lpr.h. This is not a
// public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

/** @brief The node statistics used for the reference tree.
 */
template<typename TKernel>
class KrylovLprRStat {

 public:

  ////////// Member Variables //////////

  /** @brief The vector summing up the reference polynomial term
   *         weighted by its target training value (i.e. B^T Y).
   */
  Vector sum_target_weighted_data_;

  /** @brief The norm of the summed up vector B^T Y used for the
   *         error criterion.
   */
  double sum_target_weighted_data_error_norm_;
    
  /** @brief The norm of the summed up vector B^T Y used for the
   *         pruning error allocation.
   */
  double sum_target_weighted_data_alloc_norm_;
    
  /** @brief The far field expansion created by the target
   *         weighted reference set. The i-th element denotes
   *         the far-field expansion of the i-th component of
   *         the sum_target_weighted_data_ vector.
   */
  ArrayList< EpanKernelMomentInfo >
    target_weighted_data_far_field_expansion_;
    
  /** @brief The vector summing up the reference point expansion.
   */
  Vector sum_reference_point_expansion_;

  /** @brief The norm of the sum_reference_point_expansion_
   */
  double sum_reference_point_expansion_norm_;

  /** @brief The far field expansion created by the reference point
   *         expansion set.
   */
  ArrayList< EpanKernelMomentInfo > 
    reference_point_expansion_far_field_expansion_;

  /** @brief The minimum bandwidth among the reference point.
   */
  TKernel min_bandwidth_kernel;
    
  /** @brief The maximum bandwidth among the reference point.
   */
  TKernel max_bandwidth_kernel;

  /** @brief The bounding box for the reference point expansion */
  DHrectBound<2> reference_point_expansion_bound_;

  ////////// Constructor/Destructor //////////

  /** @brief The constructor which does not do anything. */
  KrylovLprRStat() {}
    
  /** @brief The destructor which does not do anything. */
  ~KrylovLprRStat() {}
    
  ////////// Functions during the tree construction //////////

  /** @brief Resets the statistics to be a default value.
   */
  void Reset() {

    sum_target_weighted_data_.SetZero();
    sum_target_weighted_data_error_norm_ = 0;
    sum_target_weighted_data_alloc_norm_ = 0;

    sum_reference_point_expansion_.SetZero();
    sum_reference_point_expansion_norm_ = 0;

    // Initialize the bandwidth information to defaults.
    min_bandwidth_kernel.Init(DBL_MAX);
    max_bandwidth_kernel.Init(0);

    reference_point_expansion_bound_.Reset();

    for(index_t j = 0; j < target_weighted_data_far_field_expansion_.size(); 
	j++) {
      target_weighted_data_far_field_expansion_[j].Reset();
      reference_point_expansion_far_field_expansion_[j].Reset();
    }
  }

  /** @brief Allocate and initialize memory for the given dimension.
   *
   *  @param dimension The dimensionality.
   */
  void AllocateMemory(int dimension) {

    // For local polynomial regression order p, each vector contains
    // (D + p) choose D numbers.
    int lpr_order = fx_param_int_req(NULL, "lpr_order");
    int matrix_dimension = 
      (int) math::BinomialCoefficient(dimension + lpr_order, dimension);

    sum_target_weighted_data_.Init(matrix_dimension);
    target_weighted_data_far_field_expansion_.Init(matrix_dimension);
    sum_reference_point_expansion_.Init(matrix_dimension);

    for(index_t j = 0; j < matrix_dimension; j++) {
      target_weighted_data_far_field_expansion_[j].Init(dimension);
    }

    // Initialize memory for bound on reference point expansions.
    reference_point_expansion_bound_.Init(matrix_dimension);
    reference_point_expansion_far_field_expansion_.Init(matrix_dimension);
    for(index_t j = 0; j < matrix_dimension; j++) {
      reference_point_expansion_far_field_expansion_[j].Init(dimension);
    }
  }

  /** @brief Computing the statistics for a leaf node involves
   *         explicitly running over the points owned by the node.
   */
  void Init(const Matrix &dataset, index_t start, index_t count) {

    // Allocate all memory required for the statistics.
    AllocateMemory(dataset.n_rows());
  }

  void Init(const Matrix &dataset, index_t start, index_t count,
	    const KrylovLprRStat &left_stat,
	    const KrylovLprRStat &right_stat) {

    // Allocate all memory required for the statatistics.
    AllocateMemory(dataset.n_rows());
  }

};

/** @brief The node statistics used for the query tree.
 */
template<typename TKernel>
class KrylovLprQStat {

public:

  ////////// Member Variables //////////

  /** @brief The lower bound on the norm of the vector computation.
   */
  double ll_vector_norm_l_;
    
  /** @brief The upper bound on the used error for approximating the
   *         positive components of the vector computation.
   */
  double ll_vector_used_error_;

  /** @brief The lower bound on the portion of the reference set
   *         pruned for the query points owned by this node.
   */
  double ll_vector_n_pruned_;

  /** @brief The lower bound on the norm of the negative components
   *         of the vector computation.
   */
  double neg_ll_vector_norm_l_;

  /** @brief The upper bound on the used error for approximating the
   *         negative components of the vector computation.
   */
  double neg_ll_vector_used_error_;

  /** @brief The lower bound on the portion of the reference set
   *         pruned for the query points owned by this node for the
   *         negative components.
   */
  double neg_ll_vector_n_pruned_;

  /** @brief The lower bound vector offset passed from the above on
   *         each sum component of the vector owned by this node.
   */
  Vector postponed_ll_vector_l_;
    
  /** @brief This stores the portion pruned by finite difference for
   *         each sum component.
   */
  Vector postponed_ll_vector_e_;
  
  ArrayList<EpanKernelMomentInfo> postponed_moment_ll_vector_e_;

  /** @brief The amount of used error passed down from above for
   *         approximating the positive components of the vector sum.
   */
  double postponed_ll_vector_used_error_;
  
  /** @brief The portion of the reference set pruned for approximating
   *         the positive components of the vector sum passed down
   *         from above.
   */
  double postponed_ll_vector_n_pruned_;

  /** @brief This stores the portion pruned by finite difference for
   *         each negative sum component of the vector owned by this
   *         node.
   */
  Vector postponed_neg_ll_vector_e_;

  /** @brief The upper bound vector offset passed from above on each
   *         negative sum component of the right hand sides owned by
   *         this node.
   */
  Vector postponed_neg_ll_vector_u_;

  /** @brief The amount of used error passed down from above for
   *         approximating the negative components of the vector sum.
   */
  double postponed_neg_ll_vector_used_error_;
  
  /** @brief The portion of the reference set pruned for approximating
   *         the negative components of the vector sum passed down
   *         from above.
   */
  double postponed_neg_ll_vector_n_pruned_;

  /** @brief The bounding box for the Lanczos vectors. */
  DHrectBound<2> lanczos_vectors_bound_;

  /** @brief The list of reference nodes that were pruned using the
   *         Epanechnikov series expansion.
   */
  ArrayList<BinarySpaceTree< DHrectBound<2>, Matrix, KrylovLprRStat<TKernel> >  *> epanechnikov_pruned_reference_nodes_;

  ////////// Constructor/Destructor //////////

  /** @brief The constructor which does not do anything. */
  KrylovLprQStat() {}
    
  /** @brief The destructor which does not do anything. */
  ~KrylovLprQStat() {}
    
  ////////// Functions during the tree construction //////////

  /** @brief Resets all bounds to zero.
   */
  void Reset() {
    ll_vector_norm_l_ = 0;
    ll_vector_used_error_ = 0;
    ll_vector_n_pruned_ = 0;
    neg_ll_vector_norm_l_ = 0;
    neg_ll_vector_used_error_ = 0;
    neg_ll_vector_n_pruned_ = 0;
    postponed_ll_vector_l_.SetZero();
    postponed_ll_vector_e_.SetZero();
    postponed_ll_vector_used_error_ = 0;
    postponed_ll_vector_n_pruned_ = 0;
    postponed_neg_ll_vector_e_.SetZero();
    postponed_neg_ll_vector_u_.SetZero();
    postponed_neg_ll_vector_used_error_ = 0;
    postponed_neg_ll_vector_n_pruned_ = 0;
    lanczos_vectors_bound_.Reset();

    for(index_t i = 0; i < postponed_moment_ll_vector_e_.size(); i++) {
      postponed_moment_ll_vector_e_[i].Reset();
    }
    epanechnikov_pruned_reference_nodes_.Resize(0);
  }

  /** @brief Allocate and initialize memory for the given dimension.
   *
   *  @param dimension The dimensionality.
   */
  void AllocateMemory(int dimension) {

    // For local polynomial regression order p, each vector contains
    // (D + p) choose D numbers.
    int lpr_order = fx_param_int_req(NULL, "lpr_order");
    int matrix_dimension = 
      (int) math::BinomialCoefficient(dimension + lpr_order, dimension);

    postponed_ll_vector_l_.Init(matrix_dimension);
    postponed_ll_vector_e_.Init(matrix_dimension);
    postponed_moment_ll_vector_e_.Init(matrix_dimension);
    for(index_t i = 0; i < postponed_moment_ll_vector_e_.size(); i++) {
      postponed_moment_ll_vector_e_[i].Init(dimension);
    }
    postponed_neg_ll_vector_e_.Init(matrix_dimension);
    postponed_neg_ll_vector_u_.Init(matrix_dimension);

    lanczos_vectors_bound_.Init(matrix_dimension);
    
    epanechnikov_pruned_reference_nodes_.Init();
  }

  /** @brief Computing the statistics for a leaf node involves
   *         explicitly running over the points owned by the node.
   */
  void Init(const Matrix &dataset, index_t start, index_t count) {

    // Allocate all memory required for the statistics.
    AllocateMemory(dataset.n_rows());
  }

  void Init(const Matrix &dataset, index_t start, index_t count,
	    const KrylovLprQStat &left_stat,
	    const KrylovLprQStat &right_stat) {

    // Allocate all memory required for the statatistics.
    AllocateMemory(dataset.n_rows());
  }

};
