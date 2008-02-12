/** @file local_linear_krylov.h
 *
 *  This implementation can handle only non-negative training target
 *  values and points that lie the positive quadrant.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see local_linear_krylov_main.cc
 *
 *  @bug No known bugs.
 */

#ifndef LOCAL_LINEAR_KRYLOV_H
#define LOCAL_LINEAR_KRYLOV_H

#include "fastlib/fastlib.h"

/** @brief A computation class for dual-tree based local linear
 *         regression using a matrix-free Krylov subspace based method
 *         for simulataneous matrix inversion.
 *
 *  This class is only intended to compute once per instantiation.
 */
template<typename TKernel>
class LocalLinearKrylov {

  FORBID_ACCIDENTAL_COPIES(LocalLinearKrylov);

 private:

  ////////// Private Type Declarations //////////

  /** @brief The node statistics used for the tree.
   */
  class LocalLinearKrylovStat {

   public:

    ////////// Member Variables //////////

    /** @brief The general purpose lower bound on each sum component
     *         for the local linear computation.
     */
    Vector ll_vector_l_;
    
    /** @brief The general purpose upper bound on each sum component
     *         for the local linear computation.
     */
    Vector ll_vector_u_;

    /** @brief The general purpose lower bound on each negative sum
     *         component for the local linear computation.
     */
    Vector neg_ll_vector_l_;
    
    /** @brief The general purpose upper bound on each negative sum
     *         component for the local linear computation.
     */
    Vector neg_ll_vector_u_;

    /** @brief The lower bound vector offset passed from the above on
     *         each sum component of the vector owned by this node.
     */
    Vector postponed_ll_vector_l_;
    
    /** @brief This stores the portion pruned by finite difference for
     *         each sum component.
     */
    Vector postponed_ll_vector_e_;

    /** @brief The upper bound vector offset passed from above on each
     *         sum component owned by this node.
     */
    Vector postponed_ll_vector_u_;

    /** @brief The lower bound vector offset passed from the above on
     *         each negative sum component owned by this node.
     */
    Vector postponed_neg_ll_vector_l_;
    
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

    /** @brief The data weighted by the target values. */
    Vector sum_targets_weighted_by_data_;

    /** @brief The 1-norm of the vector containing the target values
     *         weighted by the data.
     */
    double l1_norm_sum_targets_weighted_by_data_;

    /** @brief For local linear regression, this is the sum of the
     *         coordinates of the points owned by this node (with the
     *         first coordinate denoting the number of points, and the
     *         remaining D coordinates being the sum).
     */
    Vector sum_coordinates_;

    /** @brief The 1-norm of the sum of the coordinates of the points
     *         owned by this node (with the first coordinate denoting
     *         the number of points, and the remaining D coordinates
     *         being the sum).
     */
    double l1_norm_sum_coordinates_;

    /** @brief The bounding box for the Lanczos vectors. */
    DHrectBound<2> lanczos_vectors_bound_;

    ////////// Constructor/Destructor //////////

    /** @brief The constructor which does not do anything. */
    LocalLinearKrylovStat() {}

    /** @brief The destructor which does not do anything. */
    ~LocalLinearKrylovStat() {}
    
    ////////// Functions during the tree construction //////////

    /** @brief Allocate and initialize memory for the given dimension.
     *
     *  @param dimension The dimensionality.
     */
    void AllocateMemory(int dimension) {

      // For local linear regression, each vector contains (D + 1)
      // numbers.
      ll_vector_l_.Init(dimension + 1);
      ll_vector_u_.Init(dimension + 1);
      neg_ll_vector_l_.Init(dimension + 1);
      neg_ll_vector_u_.Init(dimension + 1);

      postponed_ll_vector_l_.Init(dimension + 1);
      postponed_ll_vector_e_.Init(dimension + 1);
      postponed_ll_vector_u_.Init(dimension + 1);
      postponed_neg_ll_vector_l_.Init(dimension + 1);
      postponed_neg_ll_vector_e_.Init(dimension + 1);
      postponed_neg_ll_vector_u_.Init(dimension + 1);

      sum_targets_weighted_by_data_.Init(dimension + 1);
      sum_coordinates_.Init(dimension + 1);
      lanczos_vectors_bound_.Init(dimension + 1);

      l1_norm_sum_targets_weighted_by_data_ = 0;
      l1_norm_sum_coordinates_ = 0;
    }

    /** @brief Computing the statistics for a leaf node involves
     *         explicitly running over the points owned by the node.
     *
     *
     */
    void Init(const Matrix &dataset, index_t start, index_t count) {

      // Allocate all memory required for the statistics.
      AllocateMemory(dataset.n_rows());

      // Here, run over each point and compute the coordinate sums.
      sum_coordinates_.SetZero();
      for(index_t i = 0; i < count; i++) {
	const double *point = dataset.GetColumnPtr(i + start);

	for(index_t j = 1; j <= dataset.n_rows(); j++) {
	  sum_coordinates_[j] += point[j - 1];
	  l1_norm_sum_coordinates_ += point[j - 1];
	}
      }
      sum_coordinates_[0] = ((double) count);
      l1_norm_sum_coordinates_ += ((double) count);
    }

    void Init(const Matrix &dataset, index_t start, index_t count,
	      const LocalLinearKrylovStat &left_stat,
	      const LocalLinearKrylovStat &right_stat) {

      // Allocate all memory required for the statatistics.
      AllocateMemory(dataset.n_rows());

      // Combine the two coordinate sums.
      la::AddOverwrite(left_stat.sum_coordinates_,
		       right_stat.sum_coordinates_, &sum_coordinates_);
      l1_norm_sum_coordinates_ = left_stat.l1_norm_sum_coordinates_ + 
	right_stat.l1_norm_sum_coordinates_;
    }

  };

  /** @brief The internal tree type used for the computation. */
  typedef BinarySpaceTree< DHrectBound<2>, Matrix, LocalLinearKrylovStat > 
    Tree;

  ////////// Private Member Variables //////////

  /** @brief The required relative error. */
  double relative_error_;

  /** @brief The module holding the list of parameters. */
  struct datanode *module_;

  /** @brief The column-oriented query dataset. */
  Matrix qset_;
  
  /** @brief The query tree. */
  Tree *qroot_;
   
  /** @brief The permutation mapping indices of queries_ to original
   *         order.
   */
  ArrayList<index_t> old_from_new_queries_;

  /** @brief The column-oriented reference dataset. */
  Matrix rset_;
 
  /** @brief The permutation mapping indices of references_ to
   *         original order.
   */
  ArrayList<index_t> old_from_new_references_;

  /** @brief The reference tree. */
  Tree *rroot_;
  
  /** @brief The original training target value for the reference
   *         dataset.
   */
  Vector rset_targets_;

  /** @brief The original training target value for the reference
   *         dataset weighted by the reference coordinate.  (i.e. y_i
   *         [1; r^T]^T ).
   */
  Matrix rset_targets_weighted_by_coordinates_;

  /** @brief The dimensionality of each point.
   */
  int dimension_;

  /** @brief The length of each column vector in local linear regression.
   */
  int row_length_;

  /** @brief The lower bounds on the right hand side of the linear system we
   *         are solving for each query point. (i.e. B^T W(q) Y)
   */
  Matrix vector_l_;

  /** @brief The approximated right hand side of the linear system we
   *         are solving for each query point. (i.e. B^T W(q) Y)
   */
  Matrix vector_e_;

  /** @brief The upper bounds on the right hand side of the linear system we
   *         are solving for each query point. (i.e. B^T W(q) Y)
   */
  Matrix vector_u_;

  /** @brief The lower bounds on the right hand side of the linear system we
   *         are solving for each query point. (i.e. B^T W(q) Y)
   */
  Matrix neg_vector_l_;

  /** @brief The approximated right hand side of the linear system we
   *         are solving for each query point. (i.e. B^T W(q) Y)
   */
  Matrix neg_vector_e_;

  /** @brief The upper bounds on the right hand side of the linear system we
   *         are solving for each query point. (i.e. B^T W(q) Y)
   */
  Matrix neg_vector_u_;

  /** @brief The estimate of the solution vector of (B^T W(q) B)^+
   *         (B^T W(q) Y) for each query point.
   */
  Matrix solution_vectors_e_;
    
  /** @brief The final regression estimate for each query point.
   */
  Vector regression_estimates_;

  /** @brief The kernel function.
   */
  TKernel kernel_;

  /** @brief The number of finite difference prunes made.
   */
  int num_finite_difference_prunes_;

  /** @brief Temporary variable for holding newly refined lower bound.
   */
  Vector new_vector_l_;

  Vector new_vector_u_;

  Vector new_neg_vector_l_;

  /** @brief Temporary variables for holding newly refined upper bound
   *         on the negative components.
   */
  Vector new_neg_vector_u_;

  /** @brief Temporary variable for holding lower bound change made
   *         during a prune.
   */
  Vector vector_l_change_;

  /** @brief Temporary variable for holding the pruned quantity.
   */
  Vector vector_e_change_;
  
  /** @brief Temporary variable for holding upper bound change made
   *         during a prune.
   */
  Vector vector_u_change_;

  /** @brief Temporary variable for holding lower bound change made
   *         during a prune.
   */
  Vector neg_vector_l_change_;

  /** @brief Temporary variable for holding the pruned quantity.
   */
  Vector neg_vector_e_change_;
  
  /** @brief Temporary variable for holding upper bound change made
   *         during a prune.
   */
  Vector neg_vector_u_change_;

  ////////// Private Member Functions //////////

  void LocalLinearKrylov<TKernel>::MaximumRelativeErrorInL1Norm_
    (const Matrix &exact_vector_e, const Matrix &approximated,
     const ArrayList<bool> *query_should_exit_the_loop);

  /** @brief This function tests the first phase computation (i.e.,
   *         the computation of B^T W(q) Y vectors for each query
   *         point).
   */
  void TestRightHandSideComputation_(const Matrix &approximated);

  /** @brief This function test the second phase computation (i.e.
   *         the computation of the product of B^T W(q) B and z(q).
   */
  void TestKrylovComputation_
    (const Matrix &approximated, const Matrix &current_lanczos_vectors,
     const ArrayList<bool> &query_should_exit_the_loop);

  void NormalizeMatrixColumnVectors_(Matrix &m, Vector &lengths) {
    
    for(index_t i = 0; i < m.n_cols(); i++) {
      double *column_vector = m.GetColumnPtr(i);
      lengths[i] = la::LengthEuclidean(row_length_, column_vector);
      
      if(lengths[i] > 0) {
	la::Scale(row_length_, 1.0 / lengths[i], column_vector);
      }
    }
  }

  /** @brief Computes the minimum L1 norm.
   */
  double MinL1Norm_(const Vector &negative_lower_limit, 
		    const Vector &negative_upper_limit,
		    const Vector &positive_lower_limit,
		    const Vector &positive_upper_limit) {

    double norm = 0;

    for(index_t i = 0; i < negative_lower_limit.length(); i++) {
      double upper_limit = negative_upper_limit[i] +
	positive_upper_limit[i];
      double lower_limit = negative_lower_limit[i] +
	positive_lower_limit[i];

      //DEBUG_ASSERT(upper_limit >= lower_limit);

      if(lower_limit > 0) {
	norm += lower_limit;
      }
      else if(upper_limit < 0) {
	norm += (-upper_limit);
      }
    }
    DEBUG_ASSERT(norm >= 0);
    return norm;
  }
  
  /** @brief Compute the L1 norm of the given vector.
   *
   *  @param v The vector for which we want to compute the L1 norm.
   *  @return The L1 norm of the vector.
   */
  double L1Norm_(Vector &v) {

    double norm = 0;
    
    for(index_t i = 0; i < v.length(); i++) {
      norm += fabs(v[i]);
    }
    return norm;
  }

  /** @brief Determine which of the node to expand first.
   */
  void BestNodePartners_(Tree *nd, Tree *nd1, Tree *nd2, Tree **partner1,
			 Tree **partner2) {
    
    double d1 = nd->bound().MinDistanceSq(nd1->bound());
    double d2 = nd->bound().MinDistanceSq(nd2->bound());
    
    if(d1 <= d2) {
      *partner1 = nd1;
      *partner2 = nd2;
    }
    else {
      *partner1 = nd2;
      *partner2 = nd1;
    }
  }

  /** @brief Compute the dot-product bounds possible for a pair of
   *         point lying in each of the two given regions.
   */
  void DotProductBetweenTwoBounds_(Tree *qnode, Tree *rnode, 
				   DRange &negative_dot_product_range,
				   DRange &positive_dot_product_range);

  /** @brief Initialize the bound statistics relevant to the right
   *         hand side computation.
   */
  void InitializeQueryTreeRightHandSides_(Tree *qnode);

  /** @brief The postprocessing function to finalize the computation
   *         of the right-hand sides of the linear system for each
   *         query point.
   */
  void FinalizeQueryTreeRightHandSides_(Tree *qnode);

  /** @brief Preprocess the reference tree for bottom up statistics
   *         computation.
   */
  void ComputeWeightedTargetVectors_(Tree *rnode);

  /** @brief Determine whether the given query and the reference node
   *         pair can be pruned.
   *
   *  @return True, if it can be pruned. False, otherwise.
   */
  bool PrunableRightHandSides_(Tree *qnode, Tree *rnode, DRange &dsqd_range,
			       DRange &kernel_value_range, double &used_error);

  /** @brief The base-case exhaustive computation for dual-tree based
   *         computation of B^T W(q) Y.
   *
   *  @param qnode The query node.
   *  @param rnode The reference node.
   */
  void DualtreeRightHandSidesBase_(Tree *qnode, Tree *rnode);

  /** @brief The canonical case for dual-tree based computation of B^T
   *         W(q) Y.
   *
   *  @param qnode The query node.
   *  @param rnode The reference node.
   */
  void DualtreeRightHandSidesCanonical_(Tree *qnode, Tree *rnode);

  /** @brief Compute B^T W(q) Y vector for each query point, which
   *         essentially becomes the right-hand side for the linear
   *         system associated with each query point: (B^T W(q) B)
   *         z(q) = B^T W(q) Y. This function calls a dual-tree based
   *         fast vector summation to achieve this effect.
   */
  void ComputeRightHandSides_() {

    InitializeQueryTreeRightHandSides_(qroot_);
    DualtreeRightHandSidesCanonical_(qroot_, rroot_);
    FinalizeQueryTreeRightHandSides_(qroot_);
  }

  /** @brief Initialize the query tree for an iteration inside a
   *         Krylov solver. This forms the bounds for the solution
   *         vectors owned by the query points for a given query node.
   *
   *  @param qnode The current query node.
   *  @param current_lanczos_vectors Each column of this matrix is a current
   *                                 Lanczos vector for each query point.
   */
  void InitializeQueryTreeLanczosVectorBound_
    (Tree *qnode, Matrix &current_lanczos_vectors,
     const ArrayList<bool> &exclude_query_flag);

  /** @brief Initialize the query tree for an iteration inside a
   *         Krylov solver. This resets the required vector bound
   *         statistics to default.
   */
  void InitializeQueryTreeSumBound_
    (Tree *qnode, DRange &root_negative_dot_product_range,
     DRange &root_posistive_dot_product_range);

  /** @brief Finalize the Lanczos vector generator by traversing the
   *         query tree and summing up any unincorporated quantities.
   *
   *  @param qnode The query node.
   */
  void FinalizeQueryTreeLanczosMultiplier_(Tree *qnode);
  
  /** @brief Determine whether the given query and the reference node
   *         pair can be pruned.
   *
   *  @return True, if it can be pruned. False, otherwise.
   */
  bool PrunableSolver_(Tree *qnode, Tree *rnode, 
		       Matrix &current_lanczos_vectors, 
		       DRange &root_negative_dot_product_range,
		       DRange &root_positive_dot_product_range,
		       DRange &dsqd_range,
		       DRange &kernel_value_range, double &used_error);

  /** @brief The base-case exhaustive computation for dual-tree based
   *         computation of (B^T W(q) B) z(q).
   *
   *  @param qnode The query node.
   *  @param rnode The reference node.
   *  @param current_lanczos_vectors Each column of this matrix is a current
   *                                 Lanczos vector for each query point.
   */
  void DualtreeSolverBase_(Tree *qnode, Tree *rnode,
			   Matrix &current_lanczos_vectors,
			   DRange &root_negative_dot_product_range,
			   DRange &root_positive_dot_product_range,
			   const ArrayList<bool> &query_should_exit_the_loop);

  /** @brief The canonical case for dual-tree based computation of
   *         (B^T W(q) B) z(q)
   *
   *  @param qnode The query node.
   *  @param rnode The reference node.
   *  @param current_lanczos_vectors Each column of this matrix is a current
   *                                 Lanczos vector for each query point.
   */
  void DualtreeSolverCanonical_
    (Tree *qnode, Tree *rnode, Matrix &current_lanczos_vectors,
     DRange &root_negative_dot_product_range,
     DRange &root_positive_dot_product_range,
     const ArrayList<bool> &query_should_exit_the_loop);

  void SolveLeastSquaresByKrylov_();

  /** @brief Finalize the regression estimate for each query point by
   *         taking the dot-product between [1; q^T] and the final
   *         solution vector for (B^T W(q) B)^+ (B^T W(q) Y).
   */
  void FinalizeRegressionEstimates_() {

    // Loop over each query point and take the dot-product.
    for(index_t i = 0; i < qset_.n_cols(); i++) {

      // Make aliases of the current query point and its associated
      // solution vector.
      const double *query_pt = qset_.GetColumnPtr(i);
      const double *query_pt_solution = solution_vectors_e_.GetColumnPtr(i);

      // Set the first component of the dot-product.
      regression_estimates_[i] = query_pt_solution[0];

      // Loop over each dimension.
      for(index_t j = 1; j <= dimension_; j++) {
	regression_estimates_[i] += query_pt[j - 1] * query_pt_solution[j];
      }
      regression_estimates_[i] = std::max(regression_estimates_[i], 0.0);
    }
  }

 public:
  
  ////////// Constructor/Destructor //////////
  
  /** @brief The constructor that sets every pointer owned by this
   *         object to NULL.
   */
  LocalLinearKrylov() {
    qroot_ = rroot_ = NULL;
  }

  /** @brief The destructor that frees memory owned by the trees.
   */
  ~LocalLinearKrylov() {

    // If the query and the reference share the same tree, delete only one.
    if(rroot_ == qroot_) {
      delete rroot_;
      rroot_ = qroot_ = NULL;
    }
    else {
      delete rroot_;
      delete qroot_;
    }
  }

  ////////// Getter/Setters //////////

  /** @brief Get the regression estimates.
   *
   *  @param results The uninitialized vector which will be filled
   *                 with the computed regression estimates.
   */
  void get_regression_estimates(Vector *results) { 
    results->Init(regression_estimates_.length());
    
    for(index_t i = 0; i < regression_estimates_.length(); i++) {
      (*results)[i] = regression_estimates_[i];
    }
  }

  ////////// User-level Functions //////////

  void Compute() {
    
    // Zero out statistics.
    num_finite_difference_prunes_ = 0;
    
    // Set relative error.
    relative_error_ = fx_param_double(module_, "relative_error", 0.01);

    // The computation proceeds in three phases:
    //
    // Phase 1: Compute B^T W(q) Y vector for each query point.
    // Phase 2: Compute z(q) = (B^T W(q) B)^+ (B^T W(q) Y) for each query point
    //          using a matrix-free Krylov solver.
    // Phase 3: Compute [1; q^T] z(q) for each query point (the final
    //          post-processing step.)

    // The first phase computes B^T W(q) Y vector for each query
    // point. This essentially becomes the right-hand side for each
    // query point.
    fx_timer_start(module_, "local_linear_compute");
    printf("Starting Phase 1...\n");
    ComputeRightHandSides_();
    printf("Phase 1 completed...\n");

    // The second phase solves the least squares problem: (B^T W(q) B)
    // z(q) = B^T W(q) Y for each query point q.
    printf("Starting Phase 2...\n");
    SolveLeastSquaresByKrylov_();
    printf("Phase 2 completed...\n");

    // Proceed with the third phase of the computation to output the
    // final regression value.
    printf("Starting Phase 3...\n");
    FinalizeRegressionEstimates_();
    printf("Phase 3 completed...\n");
    fx_timer_stop(module_, "local_linear_compute");


    // Reshuffle the results to account for dataset reshuffling
    // resulted from tree constructions
    Vector tmp_q_results;
    tmp_q_results.Init(regression_estimates_.length());
    
    for(index_t i = 0; i < tmp_q_results.length(); i++) {
      tmp_q_results[old_from_new_queries_[i]] =	regression_estimates_[i];
    }
    for(index_t i = 0; i < tmp_q_results.length(); i++) {
      regression_estimates_[i] = tmp_q_results[i];
    }
  }

  void Init(Matrix &queries, Matrix &references, Matrix &reference_targets,
	    struct datanode *module_in) {
    
    // point to the incoming module
    module_ = module_in;
    
    // read in the number of points owned by a leaf
    int leaflen = fx_param_int(module_in, "leaflen", 20);
    
    // copy reference dataset and reference weights.
    rset_.Copy(references);
    rset_targets_.Copy(reference_targets.GetColumnPtr(0),
		       reference_targets.n_cols());
    
    // Record dimensionality and the appropriately cache the number of
    // components required for local linear (which is D + 1).
    dimension_ = rset_.n_rows();
    row_length_ = dimension_ + 1;

    // copy query dataset.
    qset_.Copy(queries);
    
    // Start measuring the tree construction time.
    fx_timer_start(NULL, "tree_d");

    // Construct the reference tree.
    rroot_ = tree::MakeKdTreeMidpoint<Tree>(rset_, leaflen,
					    &old_from_new_references_, NULL);

    // We need to shuffle the reference training target values
    // according to the shuffled order of the reference dataset.
    Vector tmp_rset_targets;
    tmp_rset_targets.Init(rset_targets_.length());
    for(index_t j = 0; j < rset_targets_.length(); j++) {
      tmp_rset_targets[j] = rset_targets_[old_from_new_references_[j]];
    }
    rset_targets_.CopyValues(tmp_rset_targets);

    // Construct the query tree.
    qroot_ = tree::MakeKdTreeMidpoint<Tree>(qset_, leaflen,
					    &old_from_new_queries_, NULL);

    fx_timer_stop(NULL, "tree_d");

    // initialize the kernel.
    kernel_.Init(fx_param_double_req(module_, "bandwidth"));

    // allocate memory for storing computation results.
    rset_targets_weighted_by_coordinates_.Init(row_length_, rset_.n_cols());
    vector_l_.Init(row_length_, qset_.n_cols());
    vector_e_.Init(row_length_, qset_.n_cols());
    vector_u_.Init(row_length_, qset_.n_cols());
    neg_vector_l_.Init(row_length_, qset_.n_cols());
    neg_vector_e_.Init(row_length_, qset_.n_cols());
    neg_vector_u_.Init(row_length_, qset_.n_cols());
    
    solution_vectors_e_.Init(row_length_, qset_.n_cols());

    regression_estimates_.Init(qset_.n_cols());
    new_vector_l_.Init(row_length_);
    new_vector_u_.Init(row_length_);
    new_neg_vector_l_.Init(row_length_);
    new_neg_vector_u_.Init(row_length_);
    vector_l_change_.Init(row_length_);
    vector_e_change_.Init(row_length_);
    vector_u_change_.Init(row_length_);
    neg_vector_l_change_.Init(row_length_);
    neg_vector_e_change_.Init(row_length_);
    neg_vector_u_change_.Init(row_length_);

    // initialize the reference side statistics.
    ComputeWeightedTargetVectors_(rroot_);
  }

  void PrintDebug() {
    
    FILE *stream = stdout;
    const char *fname = NULL;
    
    if((fname = fx_param_str(module_, 
			     "fast_local_linear_output", 
			     "fast_local_linear_output.txt")) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", regression_estimates_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#define INSIDE_LOCAL_LINEAR_KRYLOV_H
#include "local_linear_krylov_setup_impl.h"
#include "local_linear_krylov_solver_impl.h"
#include "local_linear_krylov_test.h"
#undef INSIDE_LOCAL_LINEAR_KRYLOV_H

#endif
