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

#include "mlpack/core.h"
#include "mlpack/core/tree/spacetree.hpp"
//#include "contrib/nslagle/series_expansion/farfield_expansion.h"
//#include "contrib/nslagle/series_expansion/local_expansion.h"
//#include "contrib/nslagle/series_expansion/mult_farfield_expansion.h"
//#include "contrib/nslagle/series_expansion/mult_local_expansion.h"
//#include "contrib/nslagle/series_expansion/kernel_aux.h"
#include "contrib/nslagle/proximity_project/gen_metric_tree.h"
//#include "contrib/nslagle/proximity_project/subspace_stat.h"
#include "dualtree_kde_common.h"
#include "kde_stat.h"

////////// Documentation stuffs //////////
PARAM_STRING_REQ ("data", "A file containing the reference data.","");
PARAM_STRING_REQ ("query", "A file containing the query data.","");

PARAM_DOUBLE ("bandwidth", "  The bandwidth parameter.","",0.5);
PARAM_FLAG ("do_naive", "  Whether to perform naive computation as well.","");
PARAM_STRING ("dwgts", "  A file that contains the weight of each point. If missing, will assume uniform weight","","");
PARAM_STRING ("fast_kde_output", "  A file to receive the results of computation.","","fast_kde_output.txt");
PARAM_STRING ("kernel", "  The type of kernel to use.","","");
PARAM_INT ("knn", "  The number of k-nearest neighbor to use for variable bandwidth.","",10);
PARAM_INT ("leaflen", "  The maximum number of leaves in the KD tree.","",20);
PARAM_FLAG ("loo", "  Whether to output the density estimates using leave-one-out.","");
PARAM_STRING ("mode", "  Fixed bandwidth or variable bandwidth mode.","","");
PARAM_FLAG ("multiplicative_expansion", "  Whether to do O(p^D) kernel expansion instead of O(D^p).","");
PARAM_DOUBLE ("probability", "  The probability guarantee that the relative error accuracy holds.","",1.00);
PARAM_DOUBLE ("relative_error", "  The required relative error accuracy.","",0.10);
PARAM_DOUBLE ("threshold", "  If less than this value, then absolute error bound.","",0.0);
PARAM_STRING ("scaling", "  The scaling option.","","");

PARAM_MODULE ("dualtree_kde",
  "Performs dual-tree kernel density estimate computation.");

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
 *    arma::vec results;
 *
 *    kde_module = fx_submodule(NULL, "kde", "kde_module");
 *    fast_kde.Init(queries, references, queries_equal_references,
 *                  kde_module);
 *
 *    // important to make sure that you don't call Init on results!
 *    fast_kde.Compute(&results);
 *  @endcode
 */

using namespace mlpack::tree;
using namespace mlpack::bound;

template<typename TKernelAux>
class DualtreeKde {

  friend class DualtreeKdeCommon;

 public:
  
  // our tree type using the KdeStat
  typedef tree::BinarySpaceTree<DBallBound < kernel::LMetric<2>, arma::vec>, KdeStat<TKernelAux> > Tree;
    
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
  arma::mat qset_;

  /** @brief The query tree.
   */
  Tree *qroot_;

  /** @brief The reference dataset.
   */
  arma::mat rset_;
  
  /** @brief The reference tree.
   */
  Tree *rroot_;

  /** @brief The reference weights.
   */
  arma::vec rset_weights_;

  /** @brief The running lower bound on the densities.
   */
  arma::vec densities_l_;

  /** @brief The computed densities.
   */
  arma::vec densities_e_;

  /** @brief The running upper bound on the densities.
   */
  arma::vec densities_u_;

  /** @brief The amount of used error for each query.
   */
  arma::vec used_error_;

  /** @brief The number of reference points taken care of for each
   *         query.
   */
  arma::vec n_pruned_;

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
  std::vector<size_t> old_from_new_queries_;
  
  /** @brief The permutation mapping indices of references_ to
   *         original order.
   */
  arma::Col<size_t> old_from_new_references_;

  ////////// Private Member Functions //////////

  void RefineBoundStatistics_(Tree *destination);

  /** @brief The exhaustive base KDE case.
   */
  void DualtreeKdeBase_(Tree *qnode, Tree *rnode, double probability);

  /** @brief Checking for prunability of the query and the reference
   *         pair using four types of pruning methods.
   */
  bool PrunableEnhanced_(Tree *qnode, Tree *rnode, double probability,
			 Range &dsqd_range, Range &kernel_value_range, 
			 double &dl, double &du,
			 double &used_error, double &n_pruned,
			 int &order_farfield_to_local,
			 int &order_farfield, int &order_local);
  
  double EvalUnnormOnSq_(size_t reference_point_index,
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
  void get_density_estimates(arma::vec *results) { 
    *results = arma::vec(densities_e_.size());
    
    for(size_t i = 0; i < densities_e_.size(); i++) {
      (*results)[i] = densities_e_[i];
    }
  }

  ////////// User Level Functions //////////

  void Compute(arma::vec *results) {

    // compute normalization constant
//    if(fx_param_exists(module_, "normalizing_dimension"))  {
//      NOTIFY("Using normalizing dimension of %d", 
//	     fx_param_int(module_, "normalizing_dimension", qset_.n_rows));
//      mult_const_ = 1.0 / ka_.kernel_.CalcNormConstant
//	(fx_param_int(module_, "normalizing_dimension", qset_.n_rows));
//    }
//    else {
//      NOTIFY("Using the default dimension of %d", qset_.n_rows);
      mult_const_ = 1.0 / ka_.kernel_.CalcNormConstant(qset_.n_rows);
//    }

    // Set accuracy parameters.
    relative_error_ = CLI::GetParam<double>("relative_error");
    threshold_ = CLI::GetParam<double>("threshold") *
      ka_.kernel_.CalcNormConstant(qset_.n_rows);
    
    // initialize the lower and upper bound densities
    densities_l_.zeros();
    densities_e_.zeros();
    densities_u_.fill(rset_weight_sum_);

    // Set zero for error accounting stuff.
    used_error_.zeros();
    n_pruned_.zeros();

    // Reset prune statistics.
    num_finite_difference_prunes_ = num_monte_carlo_prunes_ =
      num_farfield_to_local_prunes_ = num_farfield_prunes_ = 
      num_local_prunes_ = 0;

    printf("\nStarting fast KDE on bandwidth value of %g...\n",
	   sqrt(ka_.kernel_.bandwidth_sq()));
    CLI::StartTimer ("fast_kde_compute");

    // Preprocessing step for initializing series expansion objects
    PreProcess(rroot_);
    if(qroot_ != rroot_) {
      PreProcess(qroot_);
    }
    
    // Get the required probability guarantee for each query and call
    // the main routine.
    double probability = CLI::GetParam<double>("probability");
    DualtreeKdeCanonical_(qroot_, rroot_, probability);

    // Postprocessing step for finalizing the sums.
    PostProcess(qroot_);
    CLI::StopTimer("fast_kde_compute");
    printf("\nFast KDE completed...\n");
    printf("Finite difference prunes: %d\n", num_finite_difference_prunes_);
    printf("Monte Carlo prunes: %d\n", num_monte_carlo_prunes_);
    printf("F2L prunes: %d\n", num_farfield_to_local_prunes_);
    printf("F prunes: %d\n", num_farfield_prunes_);
    printf("L prunes: %d\n", num_local_prunes_);

    // Reshuffle the results to account for dataset reshuffling
    // resulted from tree constructions.
    arma::vec tmp_q_results(densities_e_.size());
    
    for(size_t i = 0; i < tmp_q_results.size(); i++) {
      tmp_q_results[old_from_new_queries_[i]] =
	densities_e_[i];
    }
    for(size_t i = 0; i < tmp_q_results.size(); i++) {
      densities_e_[i] = tmp_q_results[i];
    }

    // Retrieve density estimates.
    get_density_estimates(results);
  }

  void Init(const arma::mat &queries, const arma::mat &references,
	    const arma::mat &rset_weights, bool queries_equal_references, 
	    struct datanode *module_in) {

    // point to the incoming module
    module_ = module_in;

    // Set the flag for whether to perform leave-one-out computation.
    leave_one_out_ = CLI::HasParam("loo") &&
      (&queries == &references);

    // Read in the number of points owned by a leaf.
    int leaflen = CLI::GetParam<int>("leaflen");

    // Copy reference dataset and reference weights and compute its
    // sum.
    rset_ = arma::mat(references.n_rows, references.n_cols);
    for (size_t c = 0; c < references.n_cols; ++c)
    {
      for (size_t r = 0; r < references.n_rows; ++r)
      {
        rset_(r,c) = references(r,c);
      }
    }
    rset_weights_ = arma::vec(rset_weights.n_cols);
    rset_weight_sum_ = 0;
    for(size_t i = 0; i < rset_weights.n_cols; i++) {
      rset_weights_[i] = rset_weights(0, i);
      rset_weight_sum_ += rset_weights_[i];
    }

    // Copy query dataset.
    if(queries_equal_references) {
      qset_ = rset_;
    }
    else {
      qset_ = arma::mat(queries.n_rows, queries.n_cols);
      for (size_t c = 0; c < queries.n_cols; ++c)
      {
        for (size_t r = 0; r < queries.n_rows; ++r)
        {
          qset_(r,c) = queries(r,c);
        }
      }
    }

    // Construct query and reference trees. Shuffle the reference
    // weights according to the permutation of the reference set in
    // the reference tree.
    CLI::StartTimer("tree_d");
    rroot_ = proximity::MakeGenMetricTree<Tree>(rset_, leaflen,
						&old_from_new_references_, 
						NULL);
    DualtreeKdeCommon::ShuffleAccordingToPermutation
      (rset_weights_, old_from_new_references_);

    if(queries_equal_references) {
      qroot_ = rroot_;
      old_from_new_queries_ = old_from_new_references_;
    }
    else {
      qroot_ = proximity::MakeGenMetricTree<Tree>(qset_, leaflen,
						  &old_from_new_queries_, 
						  NULL);
    }
    CLI::StopTimer("tree_d");
    
    // Initialize the density lists
    densities_l_ = arma::vec(qset_.n_cols);
    densities_e_ = arma::vec(qset_.n_cols);
    densities_u_ = arma::vec(qset_.n_cols);

    // Initialize the error accounting stuff.
    used_error_ = arma::vec(qset_.n_cols);
    n_pruned_ = arma::vec(qset_.n_cols);

    // Initialize the kernel.
    double bandwidth = CLI::GetParam<double>("bandwidth");

    bool hasOrder = CLI::HasParam ("order");
    int order = -1;
    if (hasOrder)
    {
      order = CLI::GetParam<int> ("order");
    }

    // initialize the series expansion object
    if(qset_.n_rows <= 2) {
      if (!hasOrder)
      {
        order = 7;
      }
      ka_.Init(bandwidth, order, qset_.n_rows);
    }
    else if(qset_.n_rows <= 3) {
      if (!hasOrder)
      {
        order = 5;
      }
      ka_.Init(bandwidth, order, qset_.n_rows);
    }
    else if(qset_.n_rows <= 5) {
      if (!hasOrder)
      {
        order = 3;
      }
      ka_.Init(bandwidth, order, qset_.n_rows);
    }
    else if(qset_.n_rows <= 6) {
      if (!hasOrder)
      {
        order = 1;
      }
      ka_.Init(bandwidth, order, qset_.n_rows);
    }
    else {
      if (!hasOrder)
      {
        order = 0;
      }
      ka_.Init(bandwidth, order, qset_.n_rows);
    }
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = CLI::GetParam<std::string>("fast_kde_output")) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(size_t q = 0; q < qset_.n_cols; q++) {
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
