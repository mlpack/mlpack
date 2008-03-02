/** @file dense_lpr.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see kde_main.cc
 *
 *  @bug No known bugs. However, This code works only for nonnegative
 *  reference training values and nonnegative reference dataset and
 *  the Epanechnikov kernel.
 */

#ifndef DENSE_LPR_H
#define DENSE_LPR_H

#include <iostream>
#include <sstream>
#include <string>
#include "epan_kernel_moment_info.h"
#include "matrix_util.h"
#include "multi_index_util.h"
#include "fastlib/fastlib.h"
#include "mlpack/allknn/allknn.h"

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
template<typename TKernel, typename TPruneRule>
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
        ArrayList< ArrayList< EpanKernelMomentInfo > >
	  data_outer_products_far_field_expansion_;
      
        /** @brief The far field expansion created by the outer
	 *         products. The (i, j)-th element denotes the
	 *         far-field expansion of the (i, j)-th component of
	 *         the sum_data_outer_products_ matrix.
	 */
        ArrayList< ArrayList< EpanKernelMomentInfo > >
	  weight_diagram_far_field_expansion_;

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

        /** @brief The minimum bandwidth among the reference point.
	 */
        TKernel min_bandwidth_kernel;

        /** @brief The maximum bandwidth among the reference point.
	 */
        TKernel max_bandwidth_kernel;

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
	  weight_diagram_far_field_expansion_.Init(matrix_dimension);
	  sum_target_weighted_data_.Init(matrix_dimension);
	  target_weighted_data_far_field_expansion_.Init(matrix_dimension);

	  for(index_t j = 0; j < matrix_dimension; j++) {
	    
	    target_weighted_data_far_field_expansion_[j].Init(dimension);
	    data_outer_products_far_field_expansion_[j].Init(matrix_dimension);
	    weight_diagram_far_field_expansion_[j].Init(matrix_dimension);

	    for(index_t i = 0; i < matrix_dimension; i++) {
	      data_outer_products_far_field_expansion_[j][i].Init(dimension);
	      weight_diagram_far_field_expansion_[j][i].Init(dimension);
	    }
	  }

	  sum_data_outer_products_error_norm_ = 0;
	  sum_data_outer_products_alloc_norm_ = 0;
	  sum_target_weighted_data_error_norm_ = 0;
	  sum_target_weighted_data_alloc_norm_ = 0;
	  
	  // Initialize the bandwidth information to defaults.
	  min_bandwidth_kernel.Init(DBL_MAX);
	  max_bandwidth_kernel.Init(0);
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

        /** @brief Stores the portion pruned by the Epanechnikov inclusion
	 *         pruning for the numerator vector B^T W(q) Y.
	 */
        ArrayList<EpanKernelMomentInfo> postponed_moment_numerator_e_;

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

        /** @brief Stores the series expansion based pruning for the
	 *         Epanechnikov kernel for the denominator matrix B^T W(q) B.
	 */
        ArrayList< ArrayList < EpanKernelMomentInfo > >
	  postponed_moment_denominator_e_;

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

        /** @brief Stores the portion pruned by the Epanechnikov
	 *         series expansion for the numerator matrix B^T
	 *         W(q)^2 B.
	 */
        ArrayList< ArrayList < EpanKernelMomentInfo > >
	  postponed_moment_weight_diagram_numerator_e_;

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

	  for(index_t i = 0; i < postponed_numerator_l_.length(); i++) {
	    postponed_moment_numerator_e_[i].Reset();
	    for(index_t j = 0; j < postponed_numerator_l_.length(); j++) {
	      postponed_moment_denominator_e_[i][j].Reset();
	      postponed_moment_weight_diagram_numerator_e_[i][j].Reset();
	    }
	  }
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
	  postponed_moment_numerator_e_.Init(matrix_dimension);
	  for(index_t i = 0; i < matrix_dimension; i++) {
	    postponed_moment_numerator_e_[i].Init(dimension);
	  }	  
	  postponed_numerator_used_error_ = 0;
	  postponed_numerator_n_pruned_ = 0;
	  
	  // Initialize quantities associated with the denominator matrix.
	  denominator_norm_l_ = 0;
	  kernel_sum_l_ = 0;
	  denominator_used_error_ = 0;
	  denominator_n_pruned_ = 0;
	  postponed_denominator_l_.Init(matrix_dimension, matrix_dimension);
	  postponed_denominator_e_.Init(matrix_dimension, matrix_dimension);
	  postponed_moment_denominator_e_.Init(matrix_dimension);
	  for(index_t i = 0; i < matrix_dimension; i++) {
	    postponed_moment_denominator_e_[i].Init(matrix_dimension);
	    for(index_t j = 0; j < matrix_dimension; j++) {
	      postponed_moment_denominator_e_[i][j].Init(dimension);
	    }
	  }
	  postponed_denominator_used_error_ = 0;
	  postponed_denominator_n_pruned_ = 0;

	  weight_diagram_numerator_norm_l_ = 0;
	  weight_diagram_numerator_used_error_ = 0;
	  postponed_weight_diagram_numerator_l_.Init(matrix_dimension,
						     matrix_dimension);
	  postponed_weight_diagram_numerator_e_.Init(matrix_dimension,
						     matrix_dimension);
	  postponed_moment_weight_diagram_numerator_e_.Init(matrix_dimension);
	  for(index_t i = 0; i < matrix_dimension; i++) {
	    postponed_moment_weight_diagram_numerator_e_[i].Init
	      (matrix_dimension);
	    for(index_t j = 0; j < matrix_dimension; j++) {
	      postponed_moment_weight_diagram_numerator_e_[i][j].Init
		(dimension);
	    }
	  }
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
  
    ArrayList<index_t> new_from_old_references_;

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

    /** @brief The leave-one-out fit values at each reference point.
     */
    Vector leave_one_out_rset_regression_estimates_;

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

    /** @brief The root mean square deviation used for cross-validating
     *         the model.
     */
    double root_mean_square_deviation_;

    /** @brief The dimensionality of each point.
     */
    int dimension_;

    /** @brief The length of each column vector in local linear regression.
     */
    int row_length_;

    /** @brief The kernel function to use.
     */
    ArrayList<TKernel> kernels_;
 
    /** @brief The z-score for the confidence band.
     */
    double z_score_;

    ////////// Private Member Functions //////////

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
    void InitializeReferenceStatistics_(ReferenceTree *rnode);

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
     Vector *leave_one_out_query_regression_estimates,
     Vector *query_magnitude_weight_diagrams, Vector *query_influence_values,
     Matrix &numerator_l, Matrix &numerator_e, Vector &numerator_used_error, 
     Vector &numerator_n_pruned, ArrayList<Matrix> &denominator_l, 
     ArrayList<Matrix> &denominator_e, Vector &denominator_used_error, 
     Vector &denominator_n_pruned,
     ArrayList<Matrix> &weight_diagram_numerator_l,
     ArrayList<Matrix> &weight_diagram_numerator_e,
     Vector &weight_diagram_used_error);

    /** @brief Computes the root mean square deviation of the current
     *         model. This function should be called after the model has
     *         been completely built.
     */
    void ComputeRootMeanSquareDeviation_() {
      
      root_mean_square_deviation_ = 0;
      for(index_t i = 0; i < rset_.n_cols(); i++) {

	double diff_regression = rset_targets_[new_from_old_references_[i]] - 
	  leave_one_out_rset_regression_estimates_[i];
	root_mean_square_deviation_ += diff_regression * diff_regression;
      }
      root_mean_square_deviation_ *= 1.0 / ((double) rset_.n_cols());
      root_mean_square_deviation_ = sqrt(root_mean_square_deviation_);
    }

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
	double prediction_error = rset_targets_[new_from_old_references_[i]] - 
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

      fx_format_result(module_, "reference_set_first_degree_of_freedom",
		       "%g", rset_first_degree_of_freedom_);
      fx_format_result(module_, "reference_set_second_degree_of_freedom",
		       "%g", rset_second_degree_of_freedom_);
      fx_format_result(module_, "reference_set_variance", "%g",
		       rset_variance_);
    }

    void ComputeConfidenceBands_(const Matrix &queries,
				 Vector *query_regression_estimates,
				 ArrayList<DRange> *query_confidence_bands,
				 Vector *query_magnitude_weight_diagrams,
				 bool queries_equal_references) {
      
      // Initialize the storage for the confidene bands.
      query_confidence_bands->Init(queries.n_cols());
      
      for(index_t q = 0; q < queries.n_cols(); q++) {
	DRange &q_confidence_band = (*query_confidence_bands)[q];
	double spread;

	if(queries_equal_references) {
	  spread = z_score_ * (*query_magnitude_weight_diagrams)[q] * 
	    sqrt(rset_variance_);
	}
	else {
	  spread = z_score_ * (1 + (*query_magnitude_weight_diagrams)[q]) * 
	    sqrt(rset_variance_);	  
	}
	
	q_confidence_band.lo = (*query_regression_estimates)[q] - spread;
	q_confidence_band.hi = (*query_regression_estimates)[q] + spread;
      }
    }
  
    void BasicComputeSingleTree_(const Matrix &queries,
				 Vector *query_regression_estimates,
				 Vector *leave_one_out_regression_estimates,
				 ArrayList<DRange> *query_confidence_bands,
				 Vector *query_magnitude_weight_diagrams,
				 Vector *query_influence_values);

    void BasicComputeDualTree_
    (const Matrix &queries, Vector *query_regression_estimates,
     Vector *leave_one_out_query_regression_estimates,
     ArrayList<DRange> *query_confidence_bands,
     Vector *query_magnitude_weight_diagrams, Vector *query_influence_values);

    void ComputeMain_(const Matrix &queries,
		      Vector *query_regression_estimates,
		      Vector *leave_one_out_query_regression_estimates,
		      ArrayList<DRange> *query_confidence_bands,
		      Vector *query_magnitude_weight_diagrams,
		      Vector *query_influence_values) {
 
      // Clear prune statistics.
      num_finite_difference_prunes_ = num_far_field_prunes_ = 0;

      // This is the basic N-body based computation.
      if(!strncmp(fx_param_str_req(module_, "method"),"st", 2)) {
	BasicComputeSingleTree_(queries, query_regression_estimates,
				leave_one_out_query_regression_estimates,
				query_confidence_bands,
				query_magnitude_weight_diagrams,
				query_influence_values);
      }
      else {
	BasicComputeDualTree_(queries, query_regression_estimates,
			      leave_one_out_query_regression_estimates,
			      query_confidence_bands,
			      query_magnitude_weight_diagrams,
			      query_influence_values);
      }

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
			      query_magnitude_weight_diagrams,
			      (query_influence_values != NULL));

      // If the reference dataset is being used for training, then
      // compute the root mean square deviation.
      if(query_influence_values != NULL) {
	ComputeRootMeanSquareDeviation_();
      }
    }

    /** @brief Initialize the bandwidth by either fixed bandwidth
     *         parameter or a nearest neighbor based one (i.e. perform
     *         nearest neighbor and set the bandwidth equal to the k-th
     *         nearest neighbor distance).
     */
    void InitializeBandwidths_() {
      
      kernels_.Init(rset_.n_cols());
      
      if(fx_param_exists(NULL, "bandwidth")) {
	printf("Using the fixed bandwidth method...\n");
	
	double bandwidth = fx_param_double_req(NULL, "bandwidth");
	for(index_t i = 0; i < kernels_.size(); i++) {	
	  kernels_[i].Init(bandwidth);
	}
      }
      else {
	printf("Using the nearest neighbor method...\n");
	AllkNN all_knn;
	double knn_factor = fx_param_double(NULL, "knn_factor", 0.2);
	int knns = (int) (knn_factor * rset_.n_cols());

	printf("Each reference point will look for %d nearest neighbors...\n",
	       knns);
	all_knn.Init(rset_, 20, knns);
	ArrayList<index_t> resulting_neighbors;
	ArrayList<double> distances;
	
	all_knn.ComputeNeighbors(&resulting_neighbors, &distances);
	
	for(index_t i = 0; i < distances.size(); i += knns) {
	  kernels_[i / knns].Init(sqrt(distances[i + knns - 1]));
	}
      }
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
  
    /** @brief Gets the confidence bands of the model.
     */
    void get_confidence_bands
      (ArrayList<DRange> *rset_confidence_bands_copy) {
      
      rset_confidence_bands_copy->Copy(rset_confidence_bands_);
    }

    /////////// User-level Functions //////////

    double root_mean_square_deviation() {
      return root_mean_square_deviation_;
    }
  
    /** @brief Computes the query regression estimates with the
     *         confidence bands.
     */
    void Compute(const Matrix &queries, Vector *query_regression_estimates,
		 ArrayList<DRange> *query_confidence_bands,
		 Vector *query_magnitude_weight_diagrams) {
      
      fx_timer_start(module_, "dense_lpr_prediction_time");
      ComputeMain_(queries, query_regression_estimates, NULL,
		   query_confidence_bands,
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
      int leaflen = fx_param_int(module_in, "leaflen", 40);
      
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
	(rset_, leaflen, &old_from_new_references_, 
	 &new_from_old_references_);

      // We need to shuffle the reference training target values
      // according to the shuffled order of the reference dataset.
      Vector tmp_rset_targets;
      tmp_rset_targets.Init(rset_targets_.length());
      for(index_t j = 0; j < rset_targets_.length(); j++) {
	tmp_rset_targets[j] = rset_targets_[old_from_new_references_[j]];
      }
      rset_targets_.CopyValues(tmp_rset_targets);
      fx_timer_stop(NULL, "dense_lpr_reference_tree_construct");

      // Initialize the kernel. It is important to initialize the
      // kernel after reshuffling of the reference dataset is done!
      InitializeBandwidths_();

      // initialize the reference side statistics.
      target_weighted_rset_.Init(row_length_, rset_.n_cols());
      InitializeReferenceStatistics_(rroot_);

      // Train the model using the reference set (i.e. compute
      // confidence interval and degrees of freedom.)
      fx_timer_start(module_, "dense_lpr_training_time");
      ComputeMain_(references, &rset_regression_estimates_,
		   &leave_one_out_rset_regression_estimates_,
		   &rset_confidence_bands_, &rset_magnitude_weight_diagrams_,
		   &rset_influence_values_);
      fx_timer_stop(module_, "dense_lpr_training_time");
    }

    void PrintDebug() {
    
      FILE *stream = NULL;
      std::ostringstream string_converter;

      // Initialize the output file name.
      std::string fname(fx_param_str_req(module_, "method"));

      // Convert the local polynomial order to string.
      string_converter << fx_param_int_req(NULL, "lpr_order");
      fname += "_lpr_order_" + string_converter.str();
      if(fx_param_exists(NULL, "bandwidth")) {
	string_converter.str(""); 
	string_converter << fx_param_double_req(NULL, "bandwidth");
	fname += "_bandwidth_" + string_converter.str();
      }
      if(fx_param_exists(NULL, "knn_factor")) {
	string_converter.str("");
	string_converter << fx_param_double_req(NULL, "knn_factor");
	fname += "_knn_factor_" + string_converter.str();	
      }
      fname += ".txt";

      // Open the file stream for writing
      stream = fopen(fname.c_str(), "w+");
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	fprintf(stream, "%g %g %g %g %g %g\n", rset_confidence_bands_[r].lo,
		rset_regression_estimates_[r], rset_confidence_bands_[r].hi,
		leave_one_out_rset_regression_estimates_[r],
		rset_magnitude_weight_diagrams_[r],
		rset_influence_values_[r]);
      }
      
      // Close the file stream.
      fclose(stream);
    }
};

#define INSIDE_DENSE_LPR_H
#include "dense_lpr_impl.h"
#undef INSIDE_DENSE_LPR_H

#endif
