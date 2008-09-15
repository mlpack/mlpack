/** @file dualtree_kde.h
 *
 *  This file contains an implementation of kernel density estimation
 *  for a linkable library component. It implements a rudimentary
 *  depth-first dual-tree algorithm with finite difference and
 *  series-expansion approximations, using the formalized GNP
 *  framework by Ryan and Garry. Currently, it supports a
 *  fixed-bandwidth kernel density estimation with no multi-bandwidth
 *  optimizations. We assume that users will be able to cross-validate
 *  for the optimal bandwidth using a black-box optimizer which is not
 *  implemented in this code.
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
 *  @see kde_main.cc
 *  @bug No known bugs.
 */

#ifndef DUALTREE_KDE_H
#define DUALTREE_KDE_H

#define INSIDE_DUALTREE_KDE_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/mult_farfield_expansion.h"
#include "mlpack/series_expansion/mult_local_expansion.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "contrib/dongryel/proximity_project/gen_metric_tree.h"
#include "contrib/dongryel/proximity_project/subspace_stat.h"
#include "dualtree_kde_common.h"
#include "kde_stat.h"

////////// Documentation stuffs //////////
const fx_entry_doc kde_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   "  A file containing reference data.\n"},
  {"query", FX_PARAM, FX_STR, NULL,
   "  A file containing query data (defaults to data).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_entry_doc kde_entries[] = {
  {"bandwidth", FX_PARAM, FX_DOUBLE, NULL,
   "  The bandwidth parameter.\n"},
  {"coverage_percentile", FX_PARAM, FX_DOUBLE, NULL,
   "  The upper percentile of the estimates for the error guarantee.\n"},
  {"do_naive", FX_PARAM, FX_BOOL, NULL,
   "  Whether to perform naive computation as well.\n"},
  {"fast_kde_output", FX_PARAM, FX_STR, NULL,
   "  A file to receive the results of computation.\n"},
  {"kernel", FX_PARAM, FX_STR, NULL,
   "  The type of kernel to use.\n"},
  {"knn", FX_PARAM, FX_INT, NULL,
   "  The number of k-nearest neighbor to use for variable bandwidth.\n"},
  {"loo", FX_PARAM, FX_BOOL, NULL,
   "  Whether to output the density estimates using leave-one-out.\n"},
  {"mode", FX_PARAM, FX_STR, NULL,
   "  Fixed bandwidth or variable bandwidth mode.\n"},
  {"multiplicative_expansion", FX_PARAM, FX_BOOL, NULL,
   "  Whether to do O(p^D) kernel expansion instead of O(D^p).\n"},
  {"probability", FX_PARAM, FX_DOUBLE, NULL,
   "  The probability guarantee that the relative error accuracy holds.\n"},
  {"relative_error", FX_PARAM, FX_DOUBLE, NULL,
   "  The required relative error accuracy.\n"},
  {"threshold", FX_PARAM, FX_DOUBLE, NULL,
   "  If less than this value, then absolute error bound.\n"},
  {"scaling", FX_PARAM, FX_STR, NULL,
   "  The scaling option.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc kde_doc = {
  kde_entries, NULL,
  "Performs dual-tree kernel density estimate computation.\n"
};

const fx_submodule_doc kde_main_submodules[] = {
  {"kde", &kde_doc,
   "  Responsible for dual-tree kernel density estimate computation.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc kde_main_doc = {
  kde_main_entries, kde_main_submodules,
  "This is the driver for the kernel density estimator.\n"
};



/** @brief A computation class for dual-tree based kernel density
 *         estimation.
 *
 *  This class builds trees for input query and reference sets on Init.
 *  The KDE computation is then performed by calling Compute.
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
template<typename TKernelAux>
class DualtreeKde {

  friend class DualtreeKdeCommon;

 public:
  
  // our tree type using the KdeStat
  typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, KdeStat<TKernelAux> > Tree;
    
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

  /** @brief The boolean flag to control the leave-one-out computation.
   */
  bool leave_one_out_;

  /** @brief The normalization constant.
   */
  double mult_const_;

  /** @brief The series expansion auxililary object.
   */
  TKernelAux ka_;

  /** @brief The query dataset.
   */
  Matrix qset_;

  /** @brief The query tree.
   */
  Tree *qroot_;

  /** @brief The reference dataset.
   */
  Matrix rset_;
  
  /** @brief The reference tree.
   */
  Tree *rroot_;

  /** @brief The precomputed coverage probabilities.
   */
  Vector coverage_probabilities_;

  /** @brief The reference weights.
   */
  Vector rset_weights_;

  /** @brief The running lower bound on the densities.
   */
  Vector densities_l_;

  /** @brief The computed densities.
   */
  Vector densities_e_;

  /** @brief The running upper bound on the densities.
   */
  Vector densities_u_;

  /** @brief The amount of used error for each query.
   */
  Vector used_error_;

  /** @brief The number of reference points taken care of for each
   *         query.
   */
  Vector n_pruned_;

  /** @brief The temporary space to use for sorting.
   */
  Vector tmp_vector_for_sorting_;

  double lower_percentile_;

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

  /** @brief The permutation mapping indices of queries_ to original
   *         order.
   */
  ArrayList<index_t> old_from_new_queries_;
  
  /** @brief The permutation mapping indices of references_ to
   *         original order.
   */
  ArrayList<index_t> old_from_new_references_;

  ////////// Private Member Functions //////////

  void RefineBoundStatistics_(Tree *destination);

  /** @brief The exhaustive base KDE case.
   */
  void DualtreeKdeBase_(Tree *qnode, Tree *rnode, double probability);

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
  bool DualtreeKdeCanonical_(Tree *qnode, Tree *rnode, double probability);

  /** @brief Pre-processing step - this wouldn't be necessary if the
   *         core fastlib supported a Init function for Stat objects
   *         that take more arguments.
   */
  void PreProcess(Tree *node);

  /** @brief Post processing step.
   */
  void PostProcess(Tree *qnode);

 public:

  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  DualtreeKde() {
    qroot_ = rroot_ = NULL;
  }

  /** @brief The default destructor which deletes the trees.
   */
  ~DualtreeKde() { 
    
    if(qroot_ != rroot_ ) {
      delete qroot_; 
      delete rroot_; 
    } 
    else {
      delete rroot_;
    }

  }

  ////////// Getters and Setters //////////

  /** @brief Get the density estimate.
   */
  void get_density_estimates(Vector *results) { 
    results->Init(densities_e_.length());
    
    for(index_t i = 0; i < densities_e_.length(); i++) {
      (*results)[i] = densities_e_[i];
    }
  }

  ////////// User Level Functions //////////

  void Compute(Vector *results) {

    // compute normalization constant
    mult_const_ = 1.0 / ka_.kernel_.CalcNormConstant(qset_.n_rows());

    // Set accuracy parameters.
    relative_error_ = fx_param_double(module_, "relative_error", 0.1);
    threshold_ = fx_param_double(module_, "threshold", 0) *
      ka_.kernel_.CalcNormConstant(qset_.n_rows());
    
    // initialize the lower and upper bound densities
    densities_l_.SetZero();
    densities_e_.SetZero();
    densities_u_.SetAll(rset_weight_sum_);

    // Set zero for error accounting stuff.
    used_error_.SetZero();
    n_pruned_.SetZero();

    // Reset prune statistics.
    num_finite_difference_prunes_ = num_monte_carlo_prunes_ =
      num_farfield_to_local_prunes_ = num_farfield_prunes_ = 
      num_local_prunes_ = 0;

    printf("\nStarting fast KDE on bandwidth value of %g...\n",
	   sqrt(ka_.kernel_.bandwidth_sq()));
    fx_timer_start(NULL, "fast_kde_compute");

    // Preprocessing step for initializing series expansion objects
    PreProcess(rroot_);
    if(qroot_ != rroot_) {
      PreProcess(qroot_);
    }
    
    // Preprocessing step for initializing the coverage probabilities.
    fx_timer_start(fx_root, "coverage_probability_precompute");
    lower_percentile_ =
      (100.0 - fx_param_double(module_, "coverage_percentile", 100.0)) / 100.0;

    for(index_t j = 0; j < coverage_probabilities_.length(); j++) {
      coverage_probabilities_[j] = 
	DualtreeKdeCommon::OuterConfidenceInterval
	(ceil(qset_.n_cols()) * ceil(rset_.n_cols()), 
	 ceil(sample_multiple_ * (j + 1)), ceil(sample_multiple_ * (j + 1)),
	 ceil(qset_.n_cols()) * ceil(rset_.n_cols()) * lower_percentile_);
    }    
    fx_timer_stop(fx_root, "coverage_probability_precompute");
    coverage_probabilities_.PrintDebug();
    
    // Get the required probability guarantee for each query and call
    // the main routine.
    double probability = fx_param_double(module_, "probability", 1);
    DualtreeKdeCanonical_(qroot_, rroot_, probability);

    // Postprocessing step for finalizing the sums.
    PostProcess(qroot_);
    fx_timer_stop(NULL, "fast_kde_compute");
    printf("\nFast KDE completed...\n");
    printf("Finite difference prunes: %d\n", num_finite_difference_prunes_);
    printf("Monte Carlo prunes: %d\n", num_monte_carlo_prunes_);
    printf("F2L prunes: %d\n", num_farfield_to_local_prunes_);
    printf("F prunes: %d\n", num_farfield_prunes_);
    printf("L prunes: %d\n", num_local_prunes_);

    // Reshuffle the results to account for dataset reshuffling
    // resulted from tree constructions.
    Vector tmp_q_results;
    tmp_q_results.Init(densities_e_.length());
    
    for(index_t i = 0; i < tmp_q_results.length(); i++) {
      tmp_q_results[old_from_new_queries_[i]] =
	densities_e_[i];
    }
    for(index_t i = 0; i < tmp_q_results.length(); i++) {
      densities_e_[i] = tmp_q_results[i];
    }

    // Retrieve density estimates.
    get_density_estimates(results);
  }

  void Init(const Matrix &queries, const Matrix &references,
	    const Matrix &rset_weights, bool queries_equal_references, 
	    struct datanode *module_in) {

    // point to the incoming module
    module_ = module_in;

    // Set the flag for whether to perform leave-one-out computation.
    leave_one_out_ = fx_param_exists(module_in, "loo") &&
      (queries.ptr() == references.ptr());

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

    // Copy query dataset.
    if(queries_equal_references) {
      qset_.Alias(rset_);
    }
    else {
      qset_.Copy(queries);
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

    if(queries_equal_references) {
      qroot_ = rroot_;
      old_from_new_queries_.InitCopy(old_from_new_references_);
    }
    else {
      qroot_ = proximity::MakeGenMetricTree<Tree>(qset_, leaflen,
						  &old_from_new_queries_, 
						  NULL);
    }
    fx_timer_stop(NULL, "tree_d");
    
    // Initialize the density lists
    densities_l_.Init(qset_.n_cols());
    densities_e_.Init(qset_.n_cols());
    densities_u_.Init(qset_.n_cols());

    // Initialize the coverage probability vector.
    coverage_probabilities_.Init(20);

    // Initialize the error accounting stuff.
    used_error_.Init(qset_.n_cols());
    n_pruned_.Init(qset_.n_cols());
    
    // Initialize the space used for sorting.
    tmp_vector_for_sorting_.Init(leaflen);

    // Initialize the kernel.
    double bandwidth = fx_param_double_req(module_, "bandwidth");

    // initialize the series expansion object
    if(qset_.n_rows() <= 2) {
      ka_.Init(bandwidth, fx_param_int(module_, "order", 7), qset_.n_rows());
    }
    else if(qset_.n_rows() <= 3) {
      ka_.Init(bandwidth, fx_param_int(module_, "order", 5), qset_.n_rows());
    }
    else if(qset_.n_rows() <= 5) {
      ka_.Init(bandwidth, fx_param_int(module_, "order", 3), qset_.n_rows());
    }
    else if(qset_.n_rows() <= 6) {
      ka_.Init(bandwidth, fx_param_int(module_, "order", 1), qset_.n_rows());
    }
    else {
      ka_.Init(bandwidth, fx_param_int(module_, "order", 0), qset_.n_rows());
    }
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(module_, "fast_kde_output", 
			     "fast_kde_output.txt")) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", densities_e_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#include "dualtree_kde_impl.h"
#undef INSIDE_DUALTREE_KDE_H

#endif
