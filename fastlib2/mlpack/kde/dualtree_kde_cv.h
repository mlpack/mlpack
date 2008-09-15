/** @file dualtree_kde_cv.h
 *
 *  This file contains an implementation of the fixed bandwidth
 *  cross-validation score computer for kernel density estimation for
 *  a linkable library component. It implements a rudimentary
 *  depth-first dual-tree algorithm with finite difference and
 *  series-expansion approximations, using the formalized GNP
 *  framework by Ryan and Garry. One should be able to use this module
 *  as a building block in a more general bandwidth optimizer.
 *
 *  For more details on mathematical derivations, please take a look at
 *  the published conference papers (in chronological order):
 *
 *  inproceedings{DBLP:conf/sdm/GrayM03,
 *   author    = {Alexander G. Gray and Andrew W. Moore},
 *   title     = {Nonparametric Density Estimation: Toward Computational 
 *                Tractability},
 *   booktitle = {SDM},
 *   year      = {2003},
 *   ee        = {http://www.siam.org/meetings/sdm03/proceedings/sdm03_19.pdf},
 *   crossref  = {DBLP:conf/sdm/2003},
 *   bibsource = {DBLP, http://dblp.uni-trier.de}
 *  }
 *
 *  misc{ gray03rapid,
 *   author = "A. Gray and A. Moore",
 *   title = "Rapid evaluation of multiple density models",
 *   booktitle = "In C. M. Bishop and B. J. Frey, editors, 
 *                Proceedings of the Ninth International Workshop on 
 *                Artificial Intelligence and Statistics",
 *   year = "2003",
 *   url = "citeseer.ist.psu.edu/gray03rapid.html"
 *  }
 *
 *  incollection{NIPS2005_570,
 *   title = {Dual-Tree Fast Gauss Transforms},
 *   author = {Dongryeol Lee and Alexander Gray and Andrew Moore},
 *   booktitle = {Advances in Neural Information Processing Systems 18},
 *   editor = {Y. Weiss and B. Sch\"{o}lkopf and J. Platt},
 *   publisher = {MIT Press},
 *   address = {Cambridge, MA},
 *   pages = {747--754},
 *   year = {2006}
 *  }
 *
 *  inproceedings{DBLP:conf/uai/LeeG06,
 *   author    = {Dongryeol Lee and Alexander G. Gray},
 *   title     = {Faster Gaussian Summation: Theory and Experiment},
 *   booktitle = {UAI},
 *   year      = {2006},
 *   crossref  = {DBLP:conf/uai/2006},
 *   bibsource = {DBLP, http://dblp.uni-trier.de}
 *  }
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug No known bugs.
 */

#ifndef DUALTREE_KDE_CV_H
#define DUALTREE_KDE_CV_H

#define INSIDE_DUALTREE_KDE_CV_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/mult_farfield_expansion.h"
#include "mlpack/series_expansion/mult_local_expansion.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "contrib/dongryel/proximity_project/gen_metric_tree.h"
#include "contrib/dongryel/proximity_project/subspace_stat.h"
#include "dualtree_kde_common.h"
#include "kde_cv_stat.h"

/** @brief A computation class for dual-tree based kernel density
 *         estimation cross-validation
 *
 *  This class builds trees for input query and reference sets on Init.
 *  The KDE computation is then performed by calling Compute.
 *
 *  This class is only intended to compute once per instantiation.
 *
 *  Example use:
 *
 *  @code
 *    DualtreeKdeCV fast_kde;
 *    struct datanode* kde_module;
 *    double score;
 *
 *    kde_module = fx_submodule(NULL, "kde", "kde_module");
 *    fast_kde.Init(queries, references, queries_equal_references,
 *                  kde_module);
 *
 *    score = fast_kde.Compute();
 *  @endcode
 */
template<typename TKernelAux>
class DualtreeKdeCV {

  friend class DualtreeKdeCommon;

 public:
  
  // our tree type using the KdeStat
  typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, KdeCVStat<TKernelAux> > Tree;
    
 private:

  ////////// Private Constants //////////

  /** @brief The number of initial samples to take per each query when
   *         doing Monte Carlo sampling.
   */
  static const int num_initial_samples_per_query_ = 25;

  static const int sample_multiple_ = 1;

  ////////// Private Member Variables //////////

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module_;

  /** @brief The series expansion auxililary object. For the Gaussian
   *         kernel, this is the kernel with the sqrt(2) h bandwidth.
   */
  TKernelAux first_ka_;

  /** @brief The series expansion auxilary object. For the Gaussian
   *         kernel, this is the Gaussian kernel with bandwidth $h$.
   */
  TKernelAux second_ka_;

  /** @brief The reference dataset.
   */
  Matrix rset_;
  
  /** @brief The reference tree.
   */
  Tree *rroot_;

  /** @brief The reference weights.
   */
  Vector rset_weights_;

  double first_sum_l_;
  
  /** @brief The first accumulated sum: for the Gaussian kernel, this
   *         would be roughly the pairwise kernel sums with the square
   *         root of the bandwidth requested.
   */
  double first_sum_e_;

  double first_sum_u_;

  double second_sum_l_;

  /** @brief The second accumulated sum: for the Gaussian kernel, this
   *         is just the pairwise kernel sums with the original
   *         bandwidth.
   */
  double second_sum_e_;

  double second_sum_u_;

  /** @brief The normalization constant for the first sum.
   */
  double first_mult_const_;
  
  /** @brief The normalization constant for the second sum.
   */
  double second_mult_const_;

  double first_used_error_;
  
  double second_used_error_;
  
  double n_pruned_;

  /** @brief The sum of all reference weights.
   */
  double rset_weight_sum_;

  /** @brief The accuracy parameter specifying the relative error
   *         bound.
   */
  double relative_error_;

  /** @brief The accuracy parameter: if the true sum is less than this
   *         value, then relative error is not guaranteed. Instead the
   *         sum is guaranteed an absolute error bound.
   */
  double threshold_;
  
  /** @brief The number of far-field to local conversions.
   */
  int num_farfield_to_local_prunes_;

  /** @brief The number of far-field evaluations.
   */
  int num_farfield_prunes_;
  
  /** @brief The number of local accumulations.
   */
  int num_local_prunes_;
  
  /** @brief The number of finite difference prunes.
   */
  int num_finite_difference_prunes_;

  /** @brief The number of prunes using Monte Carlo.
   */
  int num_monte_carlo_prunes_;
  
  /** @brief The permutation mapping indices of references_ to
   *         original order.
   */
  ArrayList<index_t> old_from_new_references_;

  ////////// Private Member Functions //////////

  /** @brief The exhaustive base KDE case.
   */
  void DualtreeKdeCVBase_(Tree *qnode, Tree *rnode, double probability);

  /** @brief Checking for prunability of the query and the reference
   *         pair using four types of pruning methods.
   */
  bool PrunableEnhanced_(Tree *qnode, Tree *rnode, double probability,
			 DRange &dsqd_range, DRange &kernel_value_range, 
			 double &dl, double &du,
			 double &used_error, double &n_pruned,
			 int &order_farfield_to_local,
			 int &order_farfield, int &order_local);
  
  double EvalUnnormOnSq_(index_t reference_point_index,
			 double squared_distance);

  /** @brief Canonical dualtree KDE case.
   *
   *  @param qnode The query node.
   *  @param rnode The reference node.
   *  @param probability The required probability; 1 for exact
   *         approximation.
   *
   *  @return true if the entire contribution of rnode has been
   *          approximated using an exact method, false otherwise.
   */
  bool DualtreeKdeCVCanonical_(Tree *qnode, Tree *rnode, double probability);

  /** @brief Pre-processing step - this wouldn't be necessary if the
   *         core fastlib supported a Init function for Stat objects
   *         that take more arguments.
   */
  void PreProcess(Tree *node);

 public:

  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  DualtreeKdeCV() {
    rroot_ = NULL;
  }

  /** @brief The default destructor which deletes the trees.
   */
  ~DualtreeKdeCV() {    
    delete rroot_;
  }

  ////////// Getters and Setters //////////


  ////////// User Level Functions //////////

  double Compute() {

    // Compute normalization constant.
    first_mult_const_ = 1.0 / 
      (pow(sqrt(2), rset_.n_rows()) * 
       first_ka_.kernel_.CalcNormConstant(rset_.n_rows()));
    second_mult_const_ = 1.0 /
      second_ka_.kernel_.CalcNormConstant(rset_.n_rows());

    // Set accuracy parameters.
    relative_error_ = fx_param_double(module_, "relative_error", 0.1);
    threshold_ = fx_param_double(module_, "threshold", 0) *
      first_ka_.kernel_.CalcNormConstant(rset_.n_rows());

    // Reset prune statistics.
    num_finite_difference_prunes_ = num_monte_carlo_prunes_ =
      num_farfield_to_local_prunes_ = num_farfield_prunes_ = 
      num_local_prunes_ = 0;

    printf("\nStarting fast KDE on bandwidth value of %g...\n",
	   sqrt(second_ka_.kernel_.bandwidth_sq()));
    fx_timer_start(NULL, "fast_kde_compute");

    // Reset the accumulated sum...
    first_sum_l_ = first_sum_e_ = 0;
    first_sum_u_ = rset_weight_sum_ * rroot_->count();
    second_sum_l_ = second_sum_e_ = 0;
    second_sum_u_ = rset_weight_sum_ * rroot_->count();
    first_used_error_ = second_used_error_ = 0;
    n_pruned_ = 0;

    // Preprocessing step for initializing series expansion objects
    PreProcess(rroot_);
        
    // Get the required probability guarantee for each query and call
    // the main routine.
    double probability = fx_param_double(module_, "probability", 1);
    DualtreeKdeCVCanonical_(rroot_, rroot_, probability);
    fx_timer_stop(NULL, "fast_kde_compute");
    printf("\nFast KDE completed...\n");
    printf("Finite difference prunes: %d\n", num_finite_difference_prunes_);
    printf("Monte Carlo prunes: %d\n", num_monte_carlo_prunes_);
    printf("F2L prunes: %d\n", num_farfield_to_local_prunes_);
    printf("F prunes: %d\n", num_farfield_prunes_);
    printf("L prunes: %d\n", num_local_prunes_);

    // Normalize accordingly.
    first_sum_e_ *= (first_mult_const_ / rset_weight_sum_);
    second_sum_e_ *= (second_mult_const_ / rset_weight_sum_);

    // Return the sum of the two sums.
    double lscv_score = 
      (first_sum_e_ - 2.0 * second_sum_e_ +
       2.0 * second_ka_.kernel_.EvalUnnormOnSq(0.0) / 
       second_ka_.kernel_.CalcNormConstant(rset_.n_rows())) /
      ((double) rset_.n_cols());
    return lscv_score;
  }

  void Init(const Matrix &references, const Matrix &rset_weights,
	    struct datanode *module_in) {

    // point to the incoming module
    module_ = module_in;

    // Read in the number of points owned by a leaf.
    int leaflen = fx_param_int(module_in, "leaflen", 20);
    
    // Copy reference dataset and reference weights and compute its
    // sum.
    rset_.Copy(references);
    rset_weights_.Init(rset_weights.n_cols());
    rset_weight_sum_ = 0;
    for(index_t i = 0; i < rset_weights.n_cols(); i++) {
      rset_weights_[i] = rset_weights.get(0, i);
      rset_weight_sum_ += rset_weights_[i];
    }

    // Construct query and reference trees. Shuffle the reference
    // weights according to the permutation of the reference set in
    // the reference tree.
    fx_timer_start(NULL, "tree_d");
    rroot_ = proximity::MakeGenMetricTree<Tree>(rset_, leaflen,
						&old_from_new_references_, 
						NULL);
    DualtreeKdeCommon::ShuffleAccordingToPermutation
      (rset_weights_, old_from_new_references_);
    fx_timer_stop(NULL, "tree_d");

    // Initialize the kernel.
    double bandwidth = fx_param_double_req(module_, "bandwidth");

    // Initialize the series expansion object. I should think about
    // whether this is true for kernels other than Gaussian.
    if(rset_.n_rows() <= 2) {
      first_ka_.Init(sqrt(2) * bandwidth, fx_param_int(module_, "order", 7), 
		     rset_.n_rows());
      second_ka_.Init(bandwidth, fx_param_int(module_, "order", 7), 
		      rset_.n_rows());
    }
    else if(rset_.n_rows() <= 3) {
      first_ka_.Init(sqrt(2) * bandwidth, fx_param_int(module_, "order", 5), 
		     rset_.n_rows());
      second_ka_.Init(bandwidth, fx_param_int(module_, "order", 5), 
		      rset_.n_rows());
    }
    else if(rset_.n_rows() <= 5) {
      first_ka_.Init(sqrt(2) * bandwidth, fx_param_int(module_, "order", 3), 
		     rset_.n_rows());
      second_ka_.Init(bandwidth, fx_param_int(module_, "order", 3), 
		      rset_.n_rows());
    }
    else if(rset_.n_rows() <= 6) {
      first_ka_.Init(sqrt(2) * bandwidth, fx_param_int(module_, "order", 1), 
		     rset_.n_rows());
      second_ka_.Init(bandwidth, fx_param_int(module_, "order", 1), 
		      rset_.n_rows());
    }
    else {
      first_ka_.Init(sqrt(2) * bandwidth, fx_param_int(module_, "order", 0), 
		     rset_.n_rows());
      second_ka_.Init(bandwidth, fx_param_int(module_, "order", 0), 
		      rset_.n_rows()); 
    }
  }
};

#include "dualtree_kde_cv_impl.h"
#undef INSIDE_DUALTREE_KDE_CV_H

#endif
