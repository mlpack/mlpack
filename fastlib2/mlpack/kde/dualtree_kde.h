/** @file dualtree_kde.h
 *
 *  This file contains an implementation of kernel density estimation
 *  for a linkable library component. It implements a rudimentary
 *  depth-first dual-tree algorithm with finite difference and
 *  series-expansion approximations, using the formalized GNP
 *  framework by Ryan and Garry. Currently, it supports a
 *  fixed-bandwidth, uniform weight kernel density estimation with no
 *  multi-bandwidth optimizations. We assume that users will be able
 *  to cross-validate for the optimal bandwidth using a black-box
 *  optimizer which is not implemented in this code.
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
  {"do_naive", FX_PARAM, FX_BOOL, NULL,
   "  Whether to perform naive computation as well.\n"},
  {"fast_kde_output", FX_PARAM, FX_STR, NULL,
   "  A file to receive the results of computation.\n"},
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
  
 public:

  // forward declaration of KdeStat class
  class KdeStat;
  
  // our tree type using the KdeStat
  typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, KdeStat > Tree;
  
  class KdeStat {
   public:
    
    /** @brief The lower bound on the densities for the query points
     *         owned by this node.
     */
    double mass_l_;
    
    /** @brief The upper bound on the densities for the query points
     *         owned by this node
     */
    double mass_u_;

    /** @brief Upper bound on the used error for the query points
     *         owned by this node.
     */
    double used_error_;

    /** @brief Lower bound on the number of reference points taken
     *         care of for query points owned by this node.
     */
    double n_pruned_;

    /** @brief The lower bound offset passed from above.
     */
    double postponed_l_;
    
    /** @brief Stores the portion pruned by finite difference.
     */
    double postponed_e_;

    /** @brief The upper bound offset passed from above.
     */
    double postponed_u_;

    /** @brief The total amount of error used in approximation for all query
     *         points that must be propagated downwards.
     */
    double postponed_used_error_;

    /** @brief The number of reference points that were taken care of
     *         for all query points under this node; this information
     *         must be propagated downwards.
     */
    double postponed_n_pruned_;

    /** @brief The far field expansion created by the reference points
     *         in this node.
     */
    typename TKernelAux::TFarFieldExpansion farfield_expansion_;
    
    /** @brief The local expansion stored in this node.
     */
    typename TKernelAux::TLocalExpansion local_expansion_;
    
    /** @brief The subspace associated with this node.
     */
    SubspaceStat subspace_;
    
    /** @brief Initialize the statistics.
     */
    void Init() {
      mass_l_ = 0;
      mass_u_ = 0;
      used_error_ = 0;
      n_pruned_ = 0;     
     
      postponed_l_ = 0;
      postponed_e_ = 0;
      postponed_u_ = 0;
      postponed_used_error_ = 0;
      postponed_n_pruned_ = 0;
    }
    
    void Init(const TKernelAux &ka) {
      farfield_expansion_.Init(ka);
      local_expansion_.Init(ka);
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
      Init();
      subspace_.Init(dataset, start, count);
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const KdeStat& left_stat,
	      const KdeStat& right_stat) {
      Init();
      subspace_.Init(dataset, start, count, left_stat.subspace_,
		     right_stat.subspace_);
    }
    
    void Init(const Vector& center, const TKernelAux &ka) {
      
      farfield_expansion_.Init(center, ka);
      local_expansion_.Init(center, ka);
      Init();
    }
    
    KdeStat() { }
    
    ~KdeStat() { }
    
  };
  
 private:

  ////////// Private Constants //////////

  /** @brief The number of initial samples to take per each query when
   *         doing Monte Carlo sampling.
   */
  static const int num_initial_samples_per_query_ = 25;

  ////////// Private Member Variables //////////

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module_;

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

  /** @brief The accuracy parameter specifying the relative error
   *         bound.
   */
  double tau_;

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

  /** @brief Checking whether it is able to prune the query and the
   *         reference pair using Monte Carlo sampling.
   */
  bool MonteCarloPrunable_(Tree *qnode, Tree *rnode, double probability,
			   DRange &dsqd_range, DRange &kernel_value_range, 
			   double &dl, double &de, double &du, 
			   double &used_error, double &n_pruned);

  /** @brief Checking for prunability of the query and the reference
   *         pair.
   */
  bool Prunable_(Tree *qnode, Tree *rnode, double probability,
		 DRange &dsqd_range, DRange &kernel_value_range, 
		 double &dl, double &de, double &du, double &used_error, 
		 double &n_pruned);

  /** @brief Determine which of the node to expand first.
   */
  void BestNodePartners(Tree *nd, Tree *nd1, Tree *nd2, double probability,
			Tree **partner1, double *probability1, Tree **partner2,
			double *probability2);

  /** @brief Canonical dualtree KDE case.
   */
  void DualtreeKdeCanonical_(Tree *qnode, Tree *rnode, double probability);

  /** @brief Pre-processing step - this wouldn't be necessary if the
   *         core fastlib supported a Init function for Stat objects
   *         that take more arguments.
   */
  void PreProcess(Tree *node);

  /** @brief Post processing step.
   */
  void PostProcess(Tree *qnode);

  public:

  // constructor/destructor
  DualtreeKde() {
    qroot_ = rroot_ = NULL;
  }

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
    mult_const_ = 1.0 / (ka_.kernel_.CalcNormConstant(qset_.n_rows()) *
			 rset_.n_cols());

    // Set accuracy parameters.
    tau_ = fx_param_double(module_, "relative_error", 0.1);
    threshold_ = fx_param_double(module_, "threshold", 0) *
      ka_.kernel_.CalcNormConstant(qset_.n_rows());
    
    // initialize the lower and upper bound densities
    densities_l_.SetZero();
    densities_e_.SetZero();
    densities_u_.SetAll(rset_.n_cols());

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

    // reshuffle the results to account for dataset reshuffling resulted
    // from tree constructions
    Vector tmp_q_results;
    tmp_q_results.Init(densities_e_.length());
    
    for(index_t i = 0; i < tmp_q_results.length(); i++) {
      tmp_q_results[old_from_new_queries_[i]] =
	densities_e_[i];
    }
    for(index_t i = 0; i < tmp_q_results.length(); i++) {
      densities_e_[i] = tmp_q_results[i];
    }

    // retrieve density estimates.
    get_density_estimates(results);
  }

  void Init(const Matrix &queries, const Matrix &references, 
	    bool queries_equal_references, struct datanode *module_in) {

    // point to the incoming module
    module_ = module_in;

    // read in the number of points owned by a leaf
    int leaflen = fx_param_int(module_in, "leaflen", 20);

    // copy reference dataset and reference weights. Currently only supports
    // uniformly weighted KDE...
    rset_.Copy(references);
    rset_weights_.Init(rset_.n_cols());
    rset_weights_.SetAll(1);

    // copy query dataset.
    if(queries_equal_references) {
      qset_.Alias(rset_);
    }
    else {
      qset_.Copy(queries);
    }

    // construct query and reference trees
    fx_timer_start(NULL, "tree_d");
    rroot_ = proximity::MakeGenMetricTree<Tree>(rset_, leaflen,
						&old_from_new_references_, 
						NULL);

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
    
    // initialize the density lists
    densities_l_.Init(qset_.n_cols());
    densities_e_.Init(qset_.n_cols());
    densities_u_.Init(qset_.n_cols());

    // Initialize the error accounting stuff.
    used_error_.Init(qset_.n_cols());
    n_pruned_.Init(qset_.n_cols());

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
