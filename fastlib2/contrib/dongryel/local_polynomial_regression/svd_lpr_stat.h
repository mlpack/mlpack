/** @author Dongryeol Lee (dongryel)
 *
 *  This header file declares the class prototype for the node type
 *  used in the tree-based computation in SVD-based local polynomial
 *  regression.
 *
 *  @bug None.
 */

#ifndef SVD_LPR_STAT_H
#define SVD_LPR_STAT_H

class SvdLprRStat {
  
 public:
  
  /** @brief The Frobenius norm of the unweighted data matrix under
   *         this reference node.
   */
  double frobenius_norm_unweighted_reference_points_;
  
  /** @brief The Frobenius norm of the unweighted reference target
   *         values under this reference node.
   */
  double frobenius_norm_unweighted_reference_targets_;

  /** @brief The statistics build up on the reference leaf case.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
    
    frobenius_norm_unweighted_reference_points_ = 0;

    for(index_t r = start; r < start + count; r++) {
      const double *reference_column_ptr = dataset.GetColumnPtr(r);
      double frobenius_norm = 
	la::Dot(dataset.n_rows(), reference_column_ptr, reference_column_ptr);
      frobenius_norm_unweighted_reference_points_ +=
	frobenius_norm;
    }
  }
  
  /** @brief The statistics build up on the reference internal node
   *         case.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const SvdLprRStat& left_stat, const SvdLprRStat& right_stat) {

    frobenius_norm_unweighted_reference_points_ +=
      left_stat.frobenius_norm_unweighted_reference_points_ +
      right_stat.frobenius_norm_unweighted_reference_points_;
  }

  
};

class SvdLprQStat {

public:

  /** @brief
   */

};

#endif
