/** @file dualtree_lpr.h
 *
 *  @author Dongryeol Lee (dongryel)
 *  @see kde_main.cc
 *
 *  @bug No known bugs. However, This code only works for nonnegative
 *  reference training values and nonnegative reference dataset.
 */

#ifndef DUALTREE_LPR_H
#define DUALTREE_LPR_H

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
template<typename TKernel, int lpr_order = 1>
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

        /** @brief The L1-norm of the summed up matrix B^T B.
	 */
        double l1_norm_sum_data_outer_products_;

        /** @brief The vector summing up the reference polynomial term
	 *         weighted by its target training value (i.e. B^T Y).
	 */
        Vector sum_target_weighted_data_;

        /** @brief THe L1-norm of the summed up vector B^T Y.
	 */
        double l1_norm_sum_target_weighted_data_;
      
        /** @brief Basic memory allocation stuffs.
	 *
	 *  @param dimension The dimensionality of the dataset.
	 */
        void Init(int dimension) {
	  
	  int matrix_dimension = 
	    (int) math::BinomialCoefficient(dimension + lpr_order, dimension);

	  sum_data_outer_products_.Init(matrix_dimension, matrix_dimension);
	  sum_target_weighted_data_.Init(matrix_dimension);

	  l1_norm_sum_data_outer_products_ = 0;
	  l1_norm_sum_target_weighted_data_ = 0;
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
	       reference_point_expansion);
	    
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

	  l1_norm_sum_data_outer_products_ = 
	    MatrixUtil::L1Norm(sum_data_outer_products_);
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
	  
	  // Combine the two sub-sums and compute its L1 norm.
	  la::AddOverwrite(left_stat.sum_data_outer_products_,
			   right_stat.sum_data_outer_products_,
			   &sum_data_outer_products_);
	  l1_norm_sum_data_outer_products_ = 
	    MatrixUtil::L1Norm(sum_data_outer_products_);
	}

        /** @brief The constructor which does not do anything. */
        LprRStat() {}

        /** @brief The destructor which does not do anything. */
        ~LprRStat() {}
    };

    class LprQStat {
      public:
     
        /** @brief The componentwise lower bound on the numerator matrix
	 *         B^T W(q) B for the query points owned by this node.
	 */
        Matrix numerator_l_;
    
        /** @brief The componentwise upper bound on the numerator matrix
	 *         B^T W(q) B for the query points owned by this node.
	 */
        Matrix numerator_u_;

        /** @brief The upper bound on the used error for the numerator
	 *         matrix B^T W(q) B for the query points owned by this
	 *         node.
	 */
        double numerator_used_error_;

        /** @brief The lower bound on the number of reference points taken
	 *         care of for the numerator matrix B^T W(q) B for the
	 *         query points owned by this node.
	 */
        double numerator_n_pruned_;
   
        /** @brief The lower bound offset for the numerator matrix B^T
         *         W(q) B passed from above.
	 */
        Matrix postponed_numerator_l_;
    
        /** @brief Stores the portion pruned by finite difference for the
	 *         numerator matrix B^T W(q) B.
	 */
        Matrix postponed_numerator_e_;

        /** @brief The upper bound offset for the numerator matrix B^T
	 *         W(q) B passed from above.
	 */
        Matrix postponed_numerator_u_;

        /** @brief The total amount of error used in approximation for
	 *         all query points that must be propagated downwards.
	 */
        double postponed_numerator_used_error_;

        /** @brief The number of reference points that were taken care
	 *         of for all query points under this node; this
	 *         information must be propagated downwards.
	 */
        double postponed_numerator_n_pruned_;

        /** @brief The componentwise lower bound on the denominator matrix
	 *         B^T W(q) Y for the query points owned by this node.
	 */
        Matrix denominator_l_;

        /** @brief The componentwise upper bound on the denominator matrix
	 *         B^T W(q) B for the query points owned by this node.
	 */
        Matrix denominator_u_;

        /** @brief The upper bound on the used error for the denominator
	 *         matrix B^T W(q) Y for the query points owned by this
	 *         node.
	 */
        double denominator_used_error_;
      
        /** @brief The lower bound on the number of reference points taken
	 *         care of for the denominator matrix B^T W(q) Y for the
	 *         query points owned by this node.
	 */
        double denominator_n_pruned_;

        /** @brief The lower bound offset for the denominator matrix B^T
	 *         W(q) Y passed from above.
	 */
        Matrix postponed_denominator_l_;
    
        /** @brief Stores the portion pruned by finite difference for the
         *         numerator matrix B^T W(q) Y.
         */
        Matrix postponed_denominator_e_;

        /** @brief The upper bound offset for the denominator matrix B^T
	 *         W(q) Y passed from above.
	 */
        Matrix postponed_denominator_u_;

        /** @brief The total amount of error used in approximation for
	 *         all query points that must be propagated downwards.
	 */
        double postponed_denominator_used_error_;

        /** @brief The number of reference points that were taken care of
	 *         for all query points under this node; this information
	 *         must be propagated downwards.
	 */
        double postponed_denominator_n_pruned_;
    
        /** @brief Initialize the statistics by doing basic memory
	 *         allocations.
	 */
        void Init(int dimension) {

	  int matrix_dimension = 
	    (int) math::BinomialCoefficient(dimension + lpr_order, dimension);
	  
	  // Initialize quantities associated with the numerator matrix.
	  numerator_l_.Init(matrix_dimension, matrix_dimension);
	  numerator_u_.Init(matrix_dimension, matrix_dimension);
	  numerator_used_error_ = 0;
	  numerator_n_pruned_ = 0;
	  postponed_numerator_l_.Init(matrix_dimension, matrix_dimension);
	  postponed_numerator_e_.Init(matrix_dimension, matrix_dimension);
	  postponed_numerator_u_.Init(matrix_dimension, matrix_dimension);
	  postponed_numerator_used_error_ = 0;
	  postponed_numerator_n_pruned_ = 0;
	  
	  // Initialize quantities associated with the denominator matrix.
	  denominator_l_.Init(matrix_dimension, matrix_dimension);
	  denominator_u_.Init(matrix_dimension, matrix_dimension);
	  denominator_used_error_ = 0;
	  denominator_n_pruned_ = 0;
	  postponed_denominator_l_.Init(matrix_dimension, matrix_dimension);
	  postponed_denominator_e_.Init(matrix_dimension, matrix_dimension);
	  postponed_denominator_u_.Init(matrix_dimension, matrix_dimension);
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

    ////////// Private Member Functions //////////
  
    void ComputeTargetWeightedReferenceVectors_(ReferenceTree *rnode);

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
      
      // allocate memory for storing computation results.
      target_weighted_rset_.Init(row_length_, rset_.n_cols());
      
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
