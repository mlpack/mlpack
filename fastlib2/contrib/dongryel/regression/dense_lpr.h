/** @file dense_lpr.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see kde_main.cc
 *
 *  @bug No known bugs. However, This code only works for nonnegative
 *  reference training values and nonnegative reference dataset.
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
template<typename TKernel, int lpr_order, typename TPruneRule>
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

        /** @brief Basic memory allocation stuffs.
	 *
	 *  @param dimension The dimensionality of the dataset.
	 */
        void Init(int dimension) {
	  
	  int matrix_dimension = 
	    (int) math::BinomialCoefficient(dimension + lpr_order, dimension);

	  sum_data_outer_products_.Init(matrix_dimension, matrix_dimension);
	  sum_target_weighted_data_.Init(matrix_dimension);

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
	  denominator_used_error_ = 0;
	  denominator_n_pruned_ = 0;
	  postponed_denominator_l_.SetZero();
	  postponed_denominator_e_.SetZero();
	  postponed_denominator_used_error_ = 0;
	  postponed_denominator_n_pruned_ = 0;	  
        }

        /** @brief Initialize the statistics by doing basic memory
	 *         allocations.
	 */
        void Init(int dimension) {

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
	  denominator_used_error_ = 0;
	  denominator_n_pruned_ = 0;
	  postponed_denominator_l_.Init(matrix_dimension, matrix_dimension);
	  postponed_denominator_e_.Init(matrix_dimension, matrix_dimension);
	  postponed_denominator_used_error_ = 0;
	  postponed_denominator_n_pruned_ = 0;
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
  
    /** @brief The required relative error. */
    double relative_error_;

    /** @brief The internal relative error factor. */
    double internal_relative_error_;

    /** @brief The module holding the list of parameters. */
    struct datanode *module_;

    /** @brief The column-oriented query dataset. */
    Matrix qset_;

    /** @brief The pointer to the query tree. */
    QueryTree *qroot_;

    /** @brief The permutation mapping indices of queries_ to original
     *         order.
     */
    ArrayList<index_t> old_from_new_queries_;

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

    /** @brief The dimensionality of each point.
     */
    int dimension_;

    /** @brief The length of each column vector in local linear regression.
     */
    int row_length_;

    /** @brief The kernel function to use.
     */
    TKernel kernel_;

    /** @brief The final regression estimate for each query point.
     */
    Vector regression_estimates_;

    /** @brief The componentwise lower bound on the matrix B^T W(q) B.
     */
    ArrayList<Matrix> denominator_l_;
  
    /** @brief The componentwise estimate on the matrix B^T W(q) B.
     */
    ArrayList<Matrix> denominator_e_;

    /** @brief The used error for computing B^T W(q) B for each query.
     */
    Vector denominator_used_error_;
  
    /** @brief The portion of the reference points taken care of for
     *         each query.
     */
    Vector denominator_n_pruned_;

    /** @brief The componentwise lower bound on the vector B^T W(q) Y.
     */
    Matrix numerator_l_;

    /** @brief The componentwise estimate on the vector B^T W(q) Y.
     */
    Matrix numerator_e_;
    
    /** @brief The used error for computing B^T W(q) Y for each query.
     */
    Vector numerator_used_error_;
  
    /** @brief The portion of the reference points taken care of for
     *         each query.
     */
    Vector numerator_n_pruned_;
  
    ////////// Private Member Functions //////////

    /** @brief Computes the distance range and the kernel value ranges
     *         for a given query and a reference node pair.
     */
    void SqdistAndKernelRanges_(QueryTree *qnode, ReferenceTree *rnode,
				DRange &dsqd_range, 
				DRange &kernel_value_range);

    /** @brief Resets bounds relevant to the given query point.
     */
    void ResetQuery_(int q);

    /** @brief Initialize the query tree bounds.
     */
    void InitializeQueryTree_(QueryTree *qnode);

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
    void DualtreeLprBase_(QueryTree *qnode, ReferenceTree *rnode);

    void DualtreeLprCanonical_(QueryTree *qnode, ReferenceTree *rnode);

    /** @brief Finalize the regression estimates.
     */
    void FinalizeQueryTree_(QueryTree *qnode);

  public:
  
    /** @brief The constructor which sets pointers to NULL. */
    DenseLpr() {
      qroot_ = NULL;
      rroot_ = NULL;
    }

    /** @brief The destructor which does not do anything. */
    ~DenseLpr() {
    
      if(qroot_ != NULL) {
	delete qroot_;
      }
      if(rroot_ != NULL) {
	delete rroot_;
      }
    }

    /////////// User-level Functions //////////

    void Compute() {
      
      // Set the relative error tolerance.
      relative_error_ = fx_param_double(module_, "relative_error", 0.01);
      internal_relative_error_ = relative_error_ / (relative_error_ + 2.0);
      internal_relative_error_ *= internal_relative_error_;

      fx_timer_start(module_, "dense_lpr_compute");
      InitializeQueryTree_(qroot_);
      DualtreeLprCanonical_(qroot_, rroot_);
      FinalizeQueryTree_(qroot_);

      // Reshuffle the results to account for dataset reshuffling
      // resulted from tree constructions
      Vector tmp_q_results;
      tmp_q_results.Init(regression_estimates_.length());
      
      for(index_t i = 0; i < tmp_q_results.length(); i++) {
	tmp_q_results[old_from_new_queries_[i]] = regression_estimates_[i];
      }
      for(index_t i = 0; i < tmp_q_results.length(); i++) {
	regression_estimates_[i] = tmp_q_results[i];
      }

      fx_timer_stop(module_, "dense_lpr_compute");
    }

    void Init(Matrix &queries, Matrix &references, Matrix &reference_targets,
	      struct datanode *module_in) {
      
      module_ = module_in;
                
      // read in the number of points owned by a leaf
      int leaflen = fx_param_int(module_in, "leaflen", 20);
      
      // copy reference dataset and reference weights.
      rset_.Copy(references);
      rset_targets_.Copy(reference_targets.GetColumnPtr(0),
			 reference_targets.n_cols());
      
      // Record dimensionality and the appropriately cache the number of
      // components required.
      dimension_ = rset_.n_rows();
      row_length_ = (int) math::BinomialCoefficient(dimension_ + lpr_order,
						    dimension_);
      
      // copy query dataset.
      qset_.Copy(queries);
      
      // Start measuring the tree construction time.
      fx_timer_start(NULL, "tree_d");
      
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
      
      // Construct the query tree.
      qroot_ = tree::MakeKdTreeMidpoint<QueryTree>
	(qset_, leaflen, &old_from_new_queries_, NULL);
      
      fx_timer_stop(NULL, "tree_d");
      
      // initialize the kernel.
      kernel_.Init(fx_param_double_req(module_, "bandwidth"));
      
      // Allocate memory for storing computation results.
      target_weighted_rset_.Init(row_length_, rset_.n_cols());
      regression_estimates_.Init(qset_.n_cols());
      denominator_l_.Init(qset_.n_cols());
      denominator_e_.Init(qset_.n_cols());
      for(index_t q = 0; q < qset_.n_cols(); q++) {
	denominator_l_[q].Init(row_length_, row_length_);
	denominator_e_[q].Init(row_length_, row_length_);
      }
      denominator_used_error_.Init(qset_.n_cols());
      denominator_n_pruned_.Init(qset_.n_cols());
      numerator_l_.Init(row_length_, qset_.n_cols());
      numerator_e_.Init(row_length_, qset_.n_cols());
      numerator_used_error_.Init(qset_.n_cols());
      numerator_n_pruned_.Init(qset_.n_cols());
      
      // initialize the reference side statistics.
      ComputeTargetWeightedReferenceVectors_(rroot_);
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

#define INSIDE_DENSE_LPR_H
#include "dense_lpr_impl.h"
#undef INSIDE_DENSE_LPR_H

#endif
