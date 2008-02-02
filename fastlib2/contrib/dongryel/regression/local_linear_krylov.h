/** @file local_linear_krylov.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see local_linear_krylov_main.cc
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

    /** @brief The data weighted by the target values. */
    Vector data_weighted_by_target_values_;
    
    /** @brief The bounding box for the solution vectors. */
    DHrectBound<2> bound_for_solutions_;

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

      // For local linear regression, each solution vector is a vector
      // containing (D + 1) numbers.
      data_weighted_by_target_values_.Init(dimension + 1);
      bound_for_solutions_.Init(dimension + 1);
    }

    /** @brief Computing the statistics for a leaf node involves
     *         explicitly running over the points owned by the node.
     *
     *
     */
    void Init(const Matrix &dataset, index_t start, index_t count) {

      // Allocate all memory required for the statistics.
      AllocateMemory(dataset.n_rows());
    }

    void Init(const Matrix &dataset, index_t start, index_t count,
	      const LocalLinearKrylovStat &left_stat,
	      const LocalLinearKrylovStat &right_stat) {

      // Allocate all memory required for the statatistics.
      AllocateMemory(dataset.n_rows());
    }

  };

  /** @brief The internal tree type used for the computation. */
  typedef BinarySpaceTree< DHrectBound<2>, Matrix, LocalLinearKrylov > Tree;

  ////////// Private Member Variables //////////

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

  /** @brief The dimensionality of each point.
   */
  int dimension_;
  
  /** @brief The solution vector of (B^T W(q) B)^+ (B^T W(q) Y) for
   *         each query point.
   */
  Matrix solution_vectors_;

  /** @brief The final regression estimate for each query point.
   */
  Vector regression_estimates_;

  ////////// Private Member Functions //////////
  
  void DualtreeRightHandSidesCanonical_(Tree *qnode, Tree *rnode);

  /** @brief Compute B^T W(q) Y vector for each query point, which
   *         essentially becomes the right-hand side for the linear
   *         system associated with each query point: (B^T W(q) B)
   *         z(q) = B^T W(q) Y. This function calls a dual-tree based
   *         fast vector summation to achieve this effect.
   */
  void ComputeRightHandSides_() {

    DualtreeRightHandSidesCanonical_(qroot_, rroot_);
  }

  /** @brief Finalize the regression estimate for each query point by
   *         taking the dot-product between [1; q^T] and the final
   *         solution vector for (B^T W(q) B)^+ (B^T W(q) Y).
   */
  void FinalizeRegressionEstimates_() {

    // Loop over each query point and take the dot-product.
    for(index_t i = 0; i < qset_.n_cols(); i++) {

      // Make aliases of the current query point and its associated
      // solution vector.
      Vector query_pt, query_pt_solution;
      qset_.MakeColumnVector(i, &query_pt);
      solution_vectors_.MakeColumnVector(i, &query_pt_solution);

      // Set the first component of the dot-product.
      regression_estimates_[i] = query_pt_solution[0];

      // Loop over each dimension.
      for(index_t j = 1; j <= dimension_; j++) {
	regression_estimates_[i] += query_pt[j - 1] * query_pt_solution[i];
      }
    }
  }

 public:
  
  void Compute() {
    
    // Zero out computation results.
    solution_vectors_.SetZero();
    regression_estimates_.SetZero();

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
    ComputeRightHandSides_();
    
    // Proceed with the third phase of the computation to output the
    // final regression value.
    FinalizeRegressionEstimates_();
  }

  void Init(Matrix &queries, Matrix &references, Matrix &reference_targets,
	    bool queries_equal_references, struct datanode *module_in) {
    
    // point to the incoming module
    module_ = module_in;
    
    // read in the number of points owned by a leaf
    int leaflen = fx_param_int(module_in, "leaflen", 20);
    
    // copy reference dataset and reference weights.
    rset_.Copy(references);
    rset_targets_.Copy(reference_targets);
    dimension_ = rset_.n_rows();
    
    // copy query dataset.
    if(queries_equal_references) {
      qset_.Alias(rset_);
    }
    else {
      qset_.Copy(queries);
    }
    
    // construct query and reference trees
    fx_timer_start(NULL, "tree_d");
    rroot_ = tree::MakeKdTreeMidpoint<Tree>(rset_, leaflen,
					    &old_from_new_references_, NULL);
    
    if(queries_equal_references) {
      qroot_ = rroot_;
      old_from_new_queries_.Copy(old_from_new_references_);
    }
    else {
      qroot_ = tree::MakeKdTreeMidpoint<Tree>(qset_, leaflen,
					      &old_from_new_queries_, NULL);
    }
    fx_timer_stop(NULL, "tree_d");

    // allocate memory for storing computation results.
    solution_vectors_.Init(dimension_ + 1, qset_.n_cols());
    regression_estimates_.Init(qset_.n_cols());
  }
  
};

#include "local_linear_krylov_setup_impl.h"

#endif
