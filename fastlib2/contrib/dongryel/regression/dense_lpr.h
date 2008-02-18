/** @file dense_lpr.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see kde_main.cc
 *
 *  @bug No known bugs. However, This code works only for nonnegative
 *  reference training values and nonnegative reference dataset and
 *  the Epanechnikov kernel. The Gaussian kernel extension for
 *  supporting the series expansion is forth-coming.
 */

#ifndef DENSE_LPR_H
#define DENSE_LPR_H

#include "matrix_util.h"
#include "multi_index_util.h"
#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/mult_farfield_expansion.h"
#include "mlpack/series_expansion/mult_local_expansion.h"
#include "mlpack/series_expansion/kernel_aux.h"

/** @brief A computation class for dual-tree based local polynomial
 *         regression.
 *
 *  This class builds trees for input query and reference sets on
 *  Init. The LPR computation is then performed by calling Compute.
 *
 *  This class is only intended to compute once per instantiation.
 *
 *  Example use:
 *
 *  @code
 *    DualtreeKde fast_kde;
 *    struct datanode* kde_module;
 *    Vector results;
 *
 *    kde_module = fx_submodule(NULL, "kde", "kde_module");
 *    fast_kde.Init(queries, references, queries_equal_references,
 *                  kde_module);
 *
 *    // important to make sure that you don't call Init on results!
 *    fast_kde.Compute(&results);
 *  @endcode
 */
template<typename TKernelAux, typename TPruneRule>
class DenseLpr {
  
  FORBID_ACCIDENTAL_COPIES(DenseLpr);

  private:

    ////////// Private Class Definitions //////////

    class LprRStat {

      public:
      
        /** @brief The matrix summing up the unweighted data B^T B
	 *         under the current reference node.
	 */
        Matrix sum_data_outer_products_;

        /** @brief The norm of the summed up matrix B^T B used for the
	 *         error criterion.
	 */
        double sum_data_outer_products_error_norm_;

        /** @brief The norm of the summed up matrix B^T B used for the
	 *         pruning error allocation.
	 */
        double sum_data_outer_products_alloc_norm_;
      
        /** @brief The far field expansion created by the outer
	 *         products. The (i, j)-th element denotes the
	 *         far-field expansion of the (i, j)-th component of
	 *         the sum_data_outer_products_ matrix.
	 */
        ArrayList< ArrayList<typename TKernelAux::TFarFieldExpansion> >
	  data_outer_products_far_field_expansion_;
      
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
        ArrayList<typename TKernelAux::TFarFieldExpansion>
	  target_weighted_data_far_field_expansion_;

        /** @brief Initialize the far field expansion objects with the
	 *	   kernel auxiliary object. You need to call this
	 *	   function before doing anything to the expansion
	 *	   object!
	 */
        void Init(const TKernelAux &ka, int matrix_dimension) {

	  for(index_t j = 0; j < matrix_dimension; j++) {
	    target_weighted_data_far_field_expansion_[j].Init(ka);

	    for(index_t i = 0; i < matrix_dimension; i++) {
	      data_outer_products_far_field_expansion_[j][i].Init(ka);
	    }
	  }
	}

        /** @brief Basic memory allocation stuffs.
	 *
	 *  @param dimension The dimensionality of the dataset.
	 */
        void Init(int dimension) {
	  
	  int lpr_order = fx_param_int_req(NULL, "lpr_order");
	  int matrix_dimension = 
	    (int) math::BinomialCoefficient(dimension + lpr_order, dimension);

	  sum_data_outer_products_.Init(matrix_dimension, matrix_dimension);
	  data_outer_products_far_field_expansion_.Init(matrix_dimension);
	  for(index_t i = 0; i < matrix_dimension; i++) {
	    data_outer_products_far_field_expansion_[i].Init(matrix_dimension);
	  }

	  sum_target_weighted_data_.Init(matrix_dimension);
	  target_weighted_data_far_field_expansion_.Init(matrix_dimension);

	  sum_data_outer_products_error_norm_ = 0;
	  sum_data_outer_products_alloc_norm_ = 0;
	  sum_target_weighted_data_error_norm_ = 0;
	  sum_target_weighted_data_alloc_norm_ = 0;
        }

        /** @brief Computes the \sum\limits_{r \in R} [1 ; r^T]^T [1;
	 *         r^T] exhaustively for local linear regression.
	 *
	 *  @param dataset The reference dataset.
	 *  @param start The starting index of the reference dataset.
	 *  @param count The number of points in this reference node.
         */
        void Init(const Matrix& dataset, index_t &start, index_t &count) {

	  Init(dataset.n_rows());

	  int lpr_order = fx_param_int_req(NULL, "lpr_order");
	  // Temporary variables for multiindex looping
	  Vector reference_point_expansion;
	  reference_point_expansion.Init(sum_target_weighted_data_.length());

	  // Zero out the sum matrix before tallying up.
	  sum_data_outer_products_.SetZero();

	  // Loop over each reference point.
	  for(index_t r = 0; r < count; r++) {
	    
	    // Get the reference point.
	    const double *reference_point = dataset.GetColumnPtr(start + r);

	    MultiIndexUtil::ComputePointMultivariatePolynomial
	      (dataset.n_rows(), lpr_order, reference_point, 
	       reference_point_expansion.ptr());
	    
	    // Based on the polynomial expansion computed, sum up its
	    // outer product.
	    for(index_t i = 0; i < sum_data_outer_products_.n_cols(); i++) {

	      for(index_t j = 0; j < sum_data_outer_products_.n_rows(); j++) {
		sum_data_outer_products_.set
		  (j, i, sum_data_outer_products_.get(j, i) + 
		   reference_point_expansion[j] * 
		   reference_point_expansion[i]);
	      }
	    }
	  } // End of iterating over each reference point.

	  // Compute the norm of the B^T B used for error criterion
	  // and the pruning error allocation.
	  sum_data_outer_products_error_norm_ = 
	    MatrixUtil::EntrywiseLpNorm(sum_data_outer_products_, 1);
	  sum_data_outer_products_alloc_norm_ =
	    MatrixUtil::EntrywiseLpNorm(sum_data_outer_products_, 1);
	}
    
        /** @brief Computes \sum\limits_{r \in R} [1 ; r^T]^T [1; r^T] by
	 *         summing up the sub-sums computed by the children.
	 *
	 *  @param dataset The reference dataset
	 *  @param start The starting index of the reference dataset.
	 *  @param count The number of points in this reference node.
	 *  @param left_stat The statistics owned by the left child.
	 *  @param right_stat The statistics owned by the right child.
	 */
        void Init(const Matrix& dataset, index_t &start, index_t &count,
		  const LprRStat& left_stat, const LprRStat& right_stat) {
	  Init(dataset.n_rows());
	  
	  // Combine the two sub-sums.
	  la::AddOverwrite(left_stat.sum_data_outer_products_,
			   right_stat.sum_data_outer_products_,
			   &sum_data_outer_products_);

	  sum_data_outer_products_error_norm_ =
	    MatrixUtil::EntrywiseLpNorm(sum_data_outer_products_, 1);
	  sum_data_outer_products_alloc_norm_ =
	    MatrixUtil::EntrywiseLpNorm(sum_data_outer_products_, 1);
	}

        /** @brief The constructor which does not do anything. */
        LprRStat() {}

        /** @brief The destructor which does not do anything. */
        ~LprRStat() {}
    };

    class LprQStat {
      public:

        /** @brief The lower bound on the norm of the B^T W(q) Y
	 *         vector.
	 */
        double numerator_norm_l_;

        /** @brief The upper bound on the used error for the numerator
	 *         vector B^T W(q) Y for the query points owned by
	 *         this node.
	 */
        double numerator_used_error_;

        /** @brief The lower bound on the number of reference points
	 *         taken care of for the numerator vector B^T W(q) Y
	 *         for the query points owned by this node.
	 */
        double numerator_n_pruned_;
   
        /** @brief The componentwise lower bound offset for the B^T W(q) Y
	 *         vector passed from above.
	 */
        Vector postponed_numerator_l_;
    
        /** @brief Stores the portion pruned by finite difference for
	 *         the numerator vector B^T W(q) Y.
	 */
        Vector postponed_numerator_e_;

        /** @brief The total amount of error used in approximation for
	 *         all query points that must be propagated downwards.
	 */
        double postponed_numerator_used_error_;

        /** @brief The number of reference points that were taken care
	 *         of for all query points under this node; this
	 *         information must be propagated downwards.
	 */
        double postponed_numerator_n_pruned_;

        /** @brief The lower bound on the norm of the denominator
	 *         matrix B^T W(q) B for the query points owned by
	 *         this node.
	 */
        double denominator_norm_l_;

        /** @brief The lower bound on the kernel sum for the query
	 *         points belonging to this query node.
	 */
        double kernel_sum_l_;

        /** @brief The upper bound on the used error for the denominator
	 *         matrix B^T W(q) B for the query points owned by this
	 *         node.
	 */
        double denominator_used_error_;
      
        /** @brief The lower bound on the number of reference points
	 *         taken care of for the denominator matrix B^T W(q) B
	 *         for the query points owned by this node.
	 */
        double denominator_n_pruned_;

        /** @brief The lower bound offset for the norm of the
         *         numerator vector B^T W(q) B passed from above.
	 */
        Matrix postponed_denominator_l_;
    
        /** @brief Stores the portion pruned by finite difference for
         *         the numerator matrix B^T W(q) B.
         */
        Matrix postponed_denominator_e_;

        /** @brief The total amount of error used in approximation for
	 *         all query points that must be propagated downwards.
	 */
        double postponed_denominator_used_error_;

        /** @brief The number of reference points that were taken care
	 *         of for all query points under this node; this
	 *         information must be propagated downwards.
	 */
        double postponed_denominator_n_pruned_;

        /** @brief The lower bound on the norm of the denominator
	 *         matrix B^T W(q) B for the query points owned by
	 *         this node.
	 */
        double weight_diagram_numerator_norm_l_;

        /** @brief The upper bound on the used error for the denominator
	 *         matrix B^T W(q) B for the query points owned by this
	 *         node.
	 */
        double weight_diagram_numerator_used_error_;

        /** @brief The lower bound offset for the norm of the
         *         numerator vector B^T W(q) B passed from above.
	 */
        Matrix postponed_weight_diagram_numerator_l_;
    
        /** @brief Stores the portion pruned by finite difference for
         *         the numerator matrix B^T W(q) B.
         */
        Matrix postponed_weight_diagram_numerator_e_;

        /** @brief The total amount of error used in approximation for
	 *         all query points that must be propagated downwards.
	 */
        double postponed_weight_diagram_numerator_used_error_;
  
        /** @brief Resets the statistics to zero.
         */
        void SetZero() {
	  numerator_norm_l_ = 0;
	  numerator_used_error_ = 0;
	  numerator_n_pruned_ = 0;
	  postponed_numerator_l_.SetZero();
	  postponed_numerator_e_.SetZero();
	  postponed_numerator_used_error_ = 0;
	  postponed_numerator_n_pruned_ = 0;

	  denominator_norm_l_ = 0;
	  kernel_sum_l_ = 0;
	  denominator_used_error_ = 0;
	  denominator_n_pruned_ = 0;
	  postponed_denominator_l_.SetZero();
	  postponed_denominator_e_.SetZero();
	  postponed_denominator_used_error_ = 0;
	  postponed_denominator_n_pruned_ = 0;

 	  weight_diagram_numerator_norm_l_ = 0;
	  weight_diagram_numerator_used_error_ = 0;
	  postponed_weight_diagram_numerator_l_.SetZero();
	  postponed_weight_diagram_numerator_e_.SetZero();
	  postponed_weight_diagram_numerator_used_error_ = 0;
        }

        /** @brief Initialize the statistics by doing basic memory
	 *         allocations.
	 */
        void Init(int dimension) {

	  int lpr_order = fx_param_int_req(NULL, "lpr_order");
	  int matrix_dimension = 
	    (int) math::BinomialCoefficient(dimension + lpr_order, dimension);
	  
	  // Initialize quantities associated with the numerator matrix.
	  numerator_norm_l_ = 0;
	  numerator_used_error_ = 0;
	  numerator_n_pruned_ = 0;
	  postponed_numerator_l_.Init(matrix_dimension);
	  postponed_numerator_e_.Init(matrix_dimension);
	  postponed_numerator_used_error_ = 0;
	  postponed_numerator_n_pruned_ = 0;
	  
	  // Initialize quantities associated with the denominator matrix.
	  denominator_norm_l_ = 0;
	  kernel_sum_l_ = 0;
	  denominator_used_error_ = 0;
	  denominator_n_pruned_ = 0;
	  postponed_denominator_l_.Init(matrix_dimension, matrix_dimension);
	  postponed_denominator_e_.Init(matrix_dimension, matrix_dimension);
	  postponed_denominator_used_error_ = 0;
	  postponed_denominator_n_pruned_ = 0;

	  weight_diagram_numerator_norm_l_ = 0;
	  weight_diagram_numerator_used_error_ = 0;
	  postponed_weight_diagram_numerator_l_.Init(matrix_dimension,
						     matrix_dimension);
	  postponed_weight_diagram_numerator_e_.Init(matrix_dimension,
						     matrix_dimension);
	  postponed_weight_diagram_numerator_used_error_ = 0;
	}
      
        void Init(const Matrix& dataset, index_t &start, index_t &count) {
	  Init(dataset.n_rows());
	}
    
        void Init(const Matrix& dataset, index_t &start, index_t &count,
		  const LprQStat& left_stat, const LprQStat& right_stat) {
	  Init(dataset.n_rows());
	}
    
        /** @brief The constructor which does not do anything. */
        LprQStat() { }
    
        /** @brief The destructor which does not do anything. */
        ~LprQStat() { }
    };

    /** @brief The type of the query tree the local polynomial algorithm
     *         uses.
     */
    typedef BinarySpaceTree<DHrectBound<2>, Matrix, LprQStat > QueryTree;

    /** @brief The type of the reference tree the local polynomial
     *         algorithm uses.
     */
    typedef BinarySpaceTree<DHrectBound<2>, Matrix, LprRStat > ReferenceTree;

    ////////// Private Member Variables //////////

    /** @brief The number of finite difference prunes. */
    int num_finite_difference_prunes_;

    /** @brief The number of far-field prunes. */
    int num_far_field_prunes_;

    /** @brief The local polynomial order. */
    int lpr_order_;

    /** @brief The required relative error. */
    double relative_error_;

    /** @brief The internal relative error factor. */
    double internal_relative_error_;

    /** @brief The module holding the list of parameters. */
    struct datanode *module_;

    /** @brief The column-oriented reference dataset. */
    Matrix rset_;

    /** @brief The pointer to the reference tree. */
    ReferenceTree *rroot_;

    /** @brief The permutation mapping indices of references_ to
     *         original order.
     */
    ArrayList<index_t> old_from_new_references_;
  
    /** @brief The original training target value for the reference
     *         dataset.
     */
    Vector rset_targets_;

    /** @brief The original training target value for the reference
     *         dataset weighted by the reference coordinate.
     *         (i.e. y_i [1; r^T]^T ).
     */
    Matrix target_weighted_rset_;
  
    /** @brief The computed fit values at each reference point.
     */
    Vector rset_regression_estimates_;
  
    /** @brief The confidence band on the fit at each reference point.
     */
    ArrayList<DRange> rset_confidence_bands_;

    /** @brief The influence value at each reference point.
     */
    Vector rset_influence_values_;

    /** @brief The magnitude of the weight diagram vector at each
     *         reference point.
     */
    Vector rset_magnitude_weight_diagrams_;

    /** @brief The first degree of freedom, i.e. the sum of the
     *         influence value at each reference point.
     */
    double rset_first_degree_of_freedom_;
  
    /** @brief The second degree of freedom, i.e. the sum of the
     *         magnitudes of the weight diagram at each reference point.
     */
    double rset_second_degree_of_freedom_;

    /** @brief The variance of the reference set.
     */
    double rset_variance_;

    /** @brief The dimensionality of each point.
     */
    int dimension_;

    /** @brief The length of each column vector in local linear regression.
     */
    int row_length_;

    /** @brief The kernel function to use.
     */
    TKernelAux kernel_aux_;
 
    /** @brief The z-score for the confidence band.
     */
    double z_score_;

    ////////// Private Member Functions //////////

    /** @brief Computes the distance range and the kernel value ranges
     *         for a given query and a reference node pair.
     */
    void SqdistAndKernelRanges_(QueryTree *qnode, ReferenceTree *rnode,
				DRange &dsqd_range, DRange &kernel_value_range,
				Vector *furthest_point_in_qnode);

    /** @brief Resets bounds relevant to the given query point.
     */
    void ResetQuery_(int q, Matrix &numerator_l, Matrix &numerator_e,
		     Vector &numerator_used_error, Vector &numerator_n_pruned,
		     ArrayList<Matrix> &denominator_l,
		     ArrayList<Matrix> &denominator_e,
		     Vector &denominator_used_error,
		     Vector &denominator_n_pruned,
		     ArrayList<Matrix> &weight_matrix_numerator_l,
		     ArrayList<Matrix> &weight_matrix_numerator_e,
		     Vector &weight_matrix_used_error);

    /** @brief Initialize the query tree bounds.
     */
    void InitializeQueryTree_
    (QueryTree *qnode, Matrix &numerator_l, Matrix &numerator_e, 
     Vector &numerator_used_error, Vector &numerator_n_pruned,
     ArrayList<Matrix> &denominator_l, ArrayList<Matrix> &denominator_e,
     Vector &denominator_used_error, Vector &denominator_n_pruned,
     ArrayList<Matrix> &weight_diagram_numerator_l,
     ArrayList<Matrix> &weight_diagram_numerator_e,
     Vector &weight_diagram_used_error);

    /** @brief Computes the target weighted reference vectors and sums
     *         them up.
     *
     *  @param rnode The current reference node. Initially called with
     *               the root of the reference tree.
     */
    void ComputeTargetWeightedReferenceVectors_(ReferenceTree *rnode);

    void BestNodePartners_(QueryTree *nd, ReferenceTree *nd1, 
			   ReferenceTree *nd2, ReferenceTree **partner1, 
			   ReferenceTree **partner2);

    void BestNodePartners_(ReferenceTree *nd, QueryTree *nd1, 
			   QueryTree *nd2, QueryTree **partner1, 
			   QueryTree **partner2);

    /** @brief The exhaustive base LPR case.
     *
     *  @param qnode The query node.
     *  @param rnode The reference node.
     */
    void DualtreeLprBase_
    (QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
     Matrix &numerator_l, Matrix &numerator_e, Vector &numerator_used_error, 
     Vector &numerator_n_pruned, ArrayList<Matrix> &denominator_l, 
     ArrayList<Matrix> &denominator_e, Vector &denominator_used_error, 
     Vector &denominator_n_pruned,
     ArrayList<Matrix> &weight_diagram_numerator_l,
     ArrayList<Matrix> &weight_diagram_numerator_e,
     Vector &weight_diagram_used_error);
  
    /** @brief The canonical recursion for the LPR computation.
     *
     *  @param qnode The query node.
     *  @param rnode The reference node.
     */
    void DualtreeLprCanonical_
    (QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
     Matrix &numerator_l, Matrix &numerator_e, Vector &numerator_used_error, 
     Vector &numerator_n_pruned, ArrayList<Matrix> &denominator_l, 
     ArrayList<Matrix> &denominator_e, Vector &denominator_used_error, 
     Vector &denominator_n_pruned,
     ArrayList<Matrix> &weight_diagram_numerator_l,
     ArrayList<Matrix> &weight_diagram_numerator_e,
     Vector &weight_diagram_used_error);
  
    /** @brief Finalize the regression estimates.
     */
    void FinalizeQueryTree_
    (QueryTree *qnode, const Matrix &qset, Vector *query_regression_estimates,
     Vector *query_magnitude_weight_diagrams, Vector *query_influence_values,
     Matrix &numerator_l, Matrix &numerator_e, Vector &numerator_used_error, 
     Vector &numerator_n_pruned, ArrayList<Matrix> &denominator_l, 
     ArrayList<Matrix> &denominator_e, Vector &denominator_used_error, 
     Vector &denominator_n_pruned,
     ArrayList<Matrix> &weight_diagram_numerator_l,
     ArrayList<Matrix> &weight_diagram_numerator_e,
     Vector &weight_diagram_used_error);

    /** @brief Computes the variance by the normalized redisual sum of
     *         squares for the reference dataset.
     */
    void ComputeVariance_() {
      
      // Compute the degrees of freedom, i.e. the sum of the influence
      // values at each reference point and the sum of the squared
      // magnitudes of the weight diagram vectors at each reference
      // point.
      rset_first_degree_of_freedom_ = rset_second_degree_of_freedom_ = 0;
      for(index_t i = 0; i < rset_.n_cols(); i++) {
	rset_first_degree_of_freedom_ += rset_influence_values_[i];
	rset_second_degree_of_freedom_ += rset_magnitude_weight_diagrams_[i] * 
	  rset_magnitude_weight_diagrams_[i];
      }
      
      // Reset the sum accumulated to zero.
      rset_variance_ = 0;
      
      // Loop over each reference point and add up the residual.
      for(index_t i = 0; i < rset_.n_cols(); i++) {
	double prediction_error = rset_targets_[i] - 
	  rset_regression_estimates_[i];
	rset_variance_ += prediction_error * prediction_error;
      }
      
      // This could happen if enough matrices are singular...
      if(rset_.n_cols() - 2.0 * rset_first_degree_of_freedom_ +
	 rset_second_degree_of_freedom_ <= 0) {
	rset_variance_ = DBL_MAX;
      }
      
      rset_variance_ *= 1.0 / 
	(rset_.n_cols() - 2.0 * rset_first_degree_of_freedom_ +
	 rset_second_degree_of_freedom_);
    }

    void ComputeConfidenceBands_(const Matrix &queries,
				 Vector *query_regression_estimates,
				 ArrayList<DRange> *query_confidence_bands,
				 Vector *query_magnitude_weight_diagrams) {
      
      // Initialize the storage for the confidene bands.
      query_confidence_bands->Init(queries.n_cols());
      
      for(index_t q = 0; q < queries.n_cols(); q++) {
	DRange &q_confidence_band = (*query_confidence_bands)[q];
	double spread = z_score_ * (*query_magnitude_weight_diagrams)[q] * 
	  sqrt(rset_variance_);
	
	q_confidence_band.lo = (*query_regression_estimates)[q] - spread;
	q_confidence_band.hi = (*query_regression_estimates)[q] + spread;
      }
    }

    void BasicCompute_(const Matrix &queries,
		       Vector *query_regression_estimates,
		       ArrayList<DRange> *query_confidence_bands,
		       Vector *query_magnitude_weight_diagrams,
		       Vector *query_influence_values) {
      
      // Set the relative error tolerance.
      relative_error_ = fx_param_double(module_, "relative_error", 0.01);
      internal_relative_error_ = relative_error_ / (relative_error_ + 2.0);

      // Copy the query set.
      Matrix qset;
      qset.Copy(queries);

      // read in the number of points owned by a leaf
      int leaflen = fx_param_int(module_, "leaflen", 20);

      // Construct the query tree.
      ArrayList<index_t> old_from_new_queries;
      QueryTree *qroot = tree::MakeKdTreeMidpoint<QueryTree>
	(qset, leaflen, &old_from_new_queries, NULL);
      
      // Initialize storage space for intermediate computations.
      Matrix numerator_l, numerator_e;
      Vector numerator_used_error, numerator_n_pruned;
      ArrayList<Matrix> denominator_l, denominator_e;
      Vector denominator_used_error, denominator_n_pruned;
      numerator_l.Init(row_length_, queries.n_cols());
      numerator_e.Init(row_length_, queries.n_cols());
      numerator_used_error.Init(queries.n_cols());
      numerator_n_pruned.Init(queries.n_cols());
      denominator_l.Init(queries.n_cols());
      denominator_e.Init(queries.n_cols());
      for(index_t i = 0; i < queries.n_cols(); i++) {
	denominator_l[i].Init(row_length_, row_length_);
	denominator_e[i].Init(row_length_, row_length_);
      }
      denominator_used_error.Init(queries.n_cols());
      denominator_n_pruned.Init(queries.n_cols());      
      ArrayList<Matrix> weight_diagram_numerator_l, weight_diagram_numerator_e;
      Vector weight_diagram_used_error;
      weight_diagram_numerator_l.Init(queries.n_cols());
      weight_diagram_numerator_e.Init(queries.n_cols());
      for(index_t i = 0; i < queries.n_cols(); i++) {
	weight_diagram_numerator_l[i].Init(row_length_, row_length_);
	weight_diagram_numerator_e[i].Init(row_length_, row_length_);
      }
      weight_diagram_used_error.Init(queries.n_cols());

      // Initialize storage for the final results.
      query_regression_estimates->Init(queries.n_cols());
      query_magnitude_weight_diagrams->Init(queries.n_cols());
      query_influence_values->Init(queries.n_cols());
      
      // Three steps: initialize the query tree, then call dualtree,
      // then final postprocess.
      InitializeQueryTree_(qroot, numerator_l, numerator_e,
			   numerator_used_error, numerator_n_pruned,
			   denominator_l, denominator_e, 
			   denominator_used_error, denominator_n_pruned,
			   weight_diagram_numerator_l,
			   weight_diagram_numerator_e,
			   weight_diagram_used_error);
      DualtreeLprCanonical_
	(qroot, rroot_, qset, numerator_l, numerator_e, numerator_used_error, 
	 numerator_n_pruned, denominator_l, denominator_e,
	 denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e,
	 weight_diagram_used_error);
      FinalizeQueryTree_
	(qroot, qset, query_regression_estimates, 
	 query_magnitude_weight_diagrams, query_influence_values,
	 numerator_l, numerator_e, numerator_used_error, numerator_n_pruned, 
	 denominator_l, denominator_e, denominator_used_error, 
	 denominator_n_pruned, weight_diagram_numerator_l, 
	 weight_diagram_numerator_e, weight_diagram_used_error);

      // After the computation, we do not need the query tree, so we
      // free it.
      delete qroot;

      // Reshuffle the results to account for dataset reshuffling
      // resulted from tree constructions
      Vector tmp_q_results;
      tmp_q_results.Init(query_regression_estimates->length());
      
      for(index_t i = 0; i < tmp_q_results.length(); i++) {
	tmp_q_results[old_from_new_queries[i]] = 
	  (*query_regression_estimates)[i];
      }
      for(index_t i = 0; i < tmp_q_results.length(); i++) {
	(*query_regression_estimates)[i] = tmp_q_results[i];
      }
    }

    void ComputeMain_(const Matrix &queries,
		      Vector *query_regression_estimates,
		      ArrayList<DRange> *query_confidence_bands,
		      Vector *query_magnitude_weight_diagrams,
		      Vector *query_influence_values) {
 
      // Clear prune statistics.
      num_finite_difference_prunes_ = num_far_field_prunes_ = 0;

      // This is the basic N-body based computation.
      BasicCompute_(queries, query_regression_estimates,
		    query_confidence_bands, query_magnitude_weight_diagrams,
		    query_influence_values);

      printf("Number of finite difference prunes: %d\n",
	     num_finite_difference_prunes_);
      printf("Number of far-field prunes: %d\n", num_far_field_prunes_);

      // If the reference dataset is being used for training, then
      // compute variance and degrees of freedom.
      if(query_influence_values != NULL) {
	ComputeVariance_();
      }
      
      // Compute the confidence band around each query point.
      ComputeConfidenceBands_(queries, query_regression_estimates,
			      query_confidence_bands,
			      query_magnitude_weight_diagrams);
    }

  public:
  
    ////////// Constructor/Destructor //////////
  
    /** @brief The constructor which sets pointers to NULL. */
    DenseLpr() {
      rroot_ = NULL;
    }

    /** @brief The destructor which does not do anything. */
    ~DenseLpr() {
      if(rroot_ != NULL) {
	delete rroot_;
      }
    }

    ////////// Getter/Setters //////////

    /** @brief Gets the regresion estimates of the model.
     */
    void get_regression_estimates(Vector *rset_regression_estimates_copy) {
      rset_regression_estimates_copy->Copy(rset_regression_estimates_);
    }
  

    /////////// User-level Functions //////////

    /** @brief Computes the query regression estimates with the
     *         confidence bands.
     */
    void Compute(const Matrix &queries, Vector *query_regression_estimates,
		 ArrayList<DRange> *query_confidence_bands,
		 Vector *query_magnitude_weight_diagrams) {
      
      fx_timer_start(module_, "dense_lpr_prediction_time");
      ComputeMain_(queries, query_regression_estimates, query_confidence_bands,
		   query_magnitude_weight_diagrams, NULL);
      fx_timer_stop(module_, "dense_lpr_prediction_time");
    }

    /** @brief Initialize with the given reference set and the
     *         reference target set.
     */
    void Init(Matrix &references, Matrix &reference_targets,
	      struct datanode *module_in) {
      
      // set the incoming parameter module.
      module_ = module_in;
      
      // read in the number of points owned by a leaf
      int leaflen = fx_param_int(module_in, "leaflen", 20);
      
      // set the local polynomial approximation order.
      lpr_order_ = fx_param_int_req(NULL, "lpr_order");

      // copy reference dataset and reference weights.
      rset_.Copy(references);
      rset_targets_.Copy(reference_targets.GetColumnPtr(0),
			 reference_targets.n_cols());
      
      // Record dimensionality and the appropriately cache the number of
      // components required.
      dimension_ = rset_.n_rows();
      row_length_ = (int) math::BinomialCoefficient(dimension_ + lpr_order_,
						    dimension_);

      // Set the z-score necessary for computing the confidence band.
      z_score_ = fx_param_double(module_, "z_score", 1.96);
      
      // Start measuring the tree construction time.
      fx_timer_start(NULL, "dense_lpr_reference_tree_construct");
      
      // Construct the reference tree.
      rroot_ = tree::MakeKdTreeMidpoint<ReferenceTree>
	(rset_, leaflen, &old_from_new_references_, NULL);
      
      // We need to shuffle the reference training target values
      // according to the shuffled order of the reference dataset.
      Vector tmp_rset_targets;
      tmp_rset_targets.Init(rset_targets_.length());
      for(index_t j = 0; j < rset_targets_.length(); j++) {
	tmp_rset_targets[j] = rset_targets_[old_from_new_references_[j]];
      }
      rset_targets_.CopyValues(tmp_rset_targets);
      fx_timer_stop(NULL, "dense_lpr_reference_tree_construct");
      
      // Initialize the kernel.
      double bandwidth = fx_param_double_req(NULL, "bandwidth");
      kernel_aux_.Init(bandwidth, 2, dimension_);

      // initialize the reference side statistics.
      target_weighted_rset_.Init(row_length_, rset_.n_cols());
      ComputeTargetWeightedReferenceVectors_(rroot_);

      // Train the model using the reference set (i.e. compute
      // confidence interval and degrees of freedom.)
      fx_timer_start(module_, "dense_lpr_training_time");
      ComputeMain_(references, &rset_regression_estimates_,
		   &rset_confidence_bands_, &rset_magnitude_weight_diagrams_,
		   &rset_influence_values_);
      fx_timer_stop(module_, "dense_lpr_training_time");
    }

    void PrintDebug() {
    
      FILE *stream = stdout;
      const char *fname = NULL;
      
      if((fname = fx_param_str(module_, "fast_lpr_output",
			       "fast_lpr_output.txt")) != NULL) {
	stream = fopen(fname, "w+");
      }
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	fprintf(stream, "%g %g %g\n", rset_confidence_bands_[r].lo,
		rset_regression_estimates_[r], rset_confidence_bands_[r].hi);
      }
      
      if(stream != stdout) {
	fclose(stream);
      }
    }
};

#define INSIDE_DENSE_LPR_H
#include "dense_lpr_impl.h"
#undef INSIDE_DENSE_LPR_H

#endif
