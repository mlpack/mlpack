/** @author Dongryeol Lee (dongryel)
 *
 *  This header file declares function prototypes for computing local
 *  polynomial regression using a batch variant of singular value
 *  decomposition.
 *
 *  @bug None.
 */

#ifndef SVD_LPR_H
#define SVD_LPR_H

#include "fastlib/fastlib.h"

/** @brief A computation class for local polynomial regression.
 *
 */

#include "svd_lpr_stat.h"

template<typename TKernel>
class SvdLpr {

  // Do not copy this class object using a naive copy constructor!
  FORBID_ACCIDENTAL_COPIES(SvdLpr);

 private:

  /** @brief The dimensionality of the problem.
   */
  int dimension_;

  /** @brief The local polynomial order.
   */
  int lpr_order_;

  /** @brief The number of coefficients due to the chosen local
   *         polynomial order.
   */
  int num_lpr_coeffs_;

  /** @brief The module holding the list of parameters. */
  struct datanode *module_;

  /** @brief The root mean square deviation used for cross-validating
   *         the model.
   */
  double root_mean_square_deviation_;

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

  /** @brief The z-score for the confidence intervals.
   */
  double z_score_;

  /** @brief The confidence band on the fit at each reference point.
   */
  ArrayList<DRange> rset_confidence_bands_;

  /** @brief The reference dataset used to build the local polynomial
   *         regression model.
   */
  Matrix reference_set_;
  
  /** @brief The reference target training values.
   */
  Matrix reference_targets_;

  /** @brief The magnitude of the weight diagram vector at each
   *         reference point.
   */
  Vector rset_magnitude_weight_diagrams_;

  /** @brief The computed fit values at each reference point.
   */
  Vector rset_regression_estimates_;

 public:

  /////////// Constructor/Destructor //////////
  SvdLpr() {
  }

  ~SvdLpr() {
  }

  /////////// User-level Functions //////////
  
  /** @brief Initialize with the given reference set and the reference
   *         target set.
   */
  void Init(Matrix &reference_set, Matrix &reference_targets,
	    struct datanode *module_in);

  /** @brief Computes the query regression estimates with the
   *         confidence bands.
   */
  void Compute(const Matrix &queries, Vector *query_regression_estimates,
	       ArrayList<DRange> *query_confidence_bands,
	       Vector *query_magnitude_weight_diagrams);

};

#define INSIDE_SVD_LPR_H
#include "svd_lpr_user_level_impl.h"
#undef INSIDE_SVD_LPR_H

#endif
