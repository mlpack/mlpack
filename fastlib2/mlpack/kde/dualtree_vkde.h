/** @file dualtree_vkde.h
 *
 *  This file contains an implementation of the variable-bandwidth
 *  kernel density estimation for a linkable library component. It
 *  implements a rudimentary depth-first dual-tree algorithm with
 *  finite difference, using the formalized GNP framework by Ryan and
 *  Garry.
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

#ifndef DUALTREE_VKDE_H
#define DUALTREE_VKDE_H

#define INSIDE_DUALTREE_VKDE_H

#include "fastlib/fastlib.h"
#include "contrib/dongryel/proximity_project/gen_metric_tree.h"
#include "dualtree_kde_common.h"
#include "kde_stat.h"
#include "mlpack/allknn/allknn.h"

/** @brief A computation class for dual-tree based variable-bandwidth
 *         kernel density estimation.
 *
 *  This class builds trees for input query and reference sets on Init.
 *  The KDE computation is then performed by calling Compute.
 *
 *  This class is only intended to compute once per instantiation.
 *
 *  Example use:
 *
 *  @code
 *    DualtreeVKde fast_kde;
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
template<typename TKernel>
class DualtreeVKde {
  
  friend class DualtreeKdeCommon;

 public:
  
  // our tree type using the VKdeStat
  typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, VKdeStat<TKernel> > Tree;
    
 private:

  ////////// Private Constants //////////

  /** @brief The number of initial samples to take per each query when
   *         doing Monte Carlo sampling.
   */
  static const int num_initial_samples_per_query_ = 25;

  static const int sample_multiple_ = 10;

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

  /** @brief The kernel objects, one for each reference point.
   */
  ArrayList<TKernel> kernels_;

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

  /** @brief The exhaustive base KDE case.
   */
  void DualtreeVKdeBase_(Tree *qnode, Tree *rnode, double probability);

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
  bool DualtreeVKdeCanonical_(Tree *qnode, Tree *rnode, double probability);

  /** @brief Pre-processing step - this wouldn't be necessary if the
   *         core fastlib supported a Init function for Stat objects
   *         that take more arguments.
   */
  void PreProcess(Tree *node, bool reference_side);

  /** @brief Post processing step.
   */
  void PostProcess(Tree *qnode);
    
 public:

  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  DualtreeVKde() {
    qroot_ = rroot_ = NULL;
  }

  /** @brief The default destructor which deletes the trees.
   */
  ~DualtreeVKde() { 
    
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

    // Set accuracy parameters.
    relative_error_ = fx_param_double(module_, "relative_error", 0.1);
    threshold_ = fx_param_double(module_, "threshold", 0) *
      kernels_[0].CalcNormConstant(qset_.n_rows());
    
    // initialize the lower and upper bound densities
    densities_l_.SetZero();
    densities_e_.SetZero();
    densities_u_.SetAll(rset_weight_sum_);

    // Set zero for error accounting stuff.
    used_error_.SetZero();
    n_pruned_.SetZero();

    // Reset prune statistics.
    num_finite_difference_prunes_ = num_monte_carlo_prunes_ = 0;

    printf("\nStarting variable KDE using %d neighbors...\n",
	   (int) fx_param_int_req(module_, "knn"));

    fx_timer_start(NULL, "fast_kde_compute");

    // Preprocessing step for initializing series expansion objects
    PreProcess(rroot_, true);
    if(qroot_ != rroot_) {
      PreProcess(qroot_, false);
    }
    
    // Preprocessing step for initializing the coverage probabilities.
    fx_timer_start(fx_root, "coverage_probability_precompute");
    double lower_percentile =
      (100.0 - fx_param_double(module_, "coverage_percentile", 100.0)) / 100.0;

    for(index_t j = 0; j < coverage_probabilities_.length(); j++) {
      coverage_probabilities_[j] = 
	DualtreeKdeCommon::OuterConfidenceInterval
	(ceil(qset_.n_cols()) * ceil(rset_.n_cols()), 
	 ceil(sample_multiple_ * (j + 1)), 1,
	 ceil(qset_.n_cols()) * ceil(rset_.n_cols()) * lower_percentile);
    }    
    fx_timer_stop(fx_root, "coverage_probability_precompute");
    coverage_probabilities_.PrintDebug();
    
    // Get the required probability guarantee for each query and call
    // the main routine.
    double probability = fx_param_double(module_, "probability", 1);
    DualtreeVKdeCanonical_(qroot_, rroot_, probability);

    // Postprocessing step for finalizing the sums.
    PostProcess(qroot_);
    fx_timer_stop(NULL, "fast_kde_compute");
    printf("\nFast KDE completed...\n");
    printf("Finite difference prunes: %d\n", num_finite_difference_prunes_);
    printf("Monte Carlo prunes: %d\n", num_monte_carlo_prunes_);

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

    // read in the number of points owned by a leaf
    int leaflen = fx_param_int(module_in, "leaflen", 20);

    // Copy reference dataset and reference weights and compute its
    // sum. rset_weight_sum_ should be the raw sum of the reference
    // weights, ignoring the possibly different normalizing constants
    // in the case of variable-bandwidth case.
    rset_.Copy(references);
    rset_weights_.Init(rset_weights.n_cols());
    rset_weight_sum_ = 0;
    for(index_t i = 0; i < rset_weights.n_cols(); i++) {
      rset_weights_[i] = rset_weights.get(0, i);
      rset_weight_sum_ += rset_weights_[i];
    }

    // Copy the query dataset.
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
    coverage_probabilities_.Init(10);

    // Initialize the error accounting stuff.
    used_error_.Init(qset_.n_cols());
    n_pruned_.Init(qset_.n_cols());

    // Initialize the kernels for each reference point.
    int knns = fx_param_int_req(module_, "knn");
    AllkNN all_knn;
    kernels_.Init(rset_.n_cols());
    all_knn.Init(rset_, 20, knns);
    ArrayList<index_t> resulting_neighbors;
    ArrayList<double> squared_distances;    

    fx_timer_start(fx_root, "bandwidth_initialization");
    all_knn.ComputeNeighbors(&resulting_neighbors, &squared_distances);

    for(index_t i = 0; i < squared_distances.size(); i += knns) {
      kernels_[i / knns].Init(sqrt(squared_distances[i + knns - 1]));
    }
    fx_timer_stop(fx_root, "bandwidth_initialization");

    // Renormalize the reference weights according to the bandwidths
    // that have been chosen.
    double min_norm_const = DBL_MAX;
    for(index_t i = 0; i < rset_weights_.length(); i++) {
      double norm_const = kernels_[i].CalcNormConstant(qset_.n_rows());
      min_norm_const = std::min(min_norm_const, norm_const);
    }
    for(index_t i = 0; i < rset_weights_.length(); i++) {
      double norm_const = kernels_[i].CalcNormConstant(qset_.n_rows());
      rset_weights_[i] *= (min_norm_const / norm_const);
    }

    // Compute normalization constant.
    mult_const_ = 1.0 / min_norm_const;
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

#include "dualtree_vkde_impl.h"
#undef INSIDE_DUALTREE_VKDE_H

#endif
