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
 *  The difference between dualtree_kde.h and kde.h is that this
 *  implementation satisfies the any-time bound criterion.
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

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/mult_farfield_expansion.h"
#include "mlpack/series_expansion/mult_local_expansion.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "contrib/dongryel/proximity_project/gen_metric_tree.h"

////////// Documentation stuffs //////////
const fx_entry_doc kde_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   "  A file containing reference data.\n"},
  {"query", FX_PARAM, FX_STR, NULL,
   "  A file containing query data (defaults to data).\n"},
  {"fast_kde_output", FX_PARAM, FX_STR, NULL,
   "  A file to receive the results of computation.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_entry_doc kde_entries[] = {
  {"bandwidth", FX_REQUIRED, FX_DOUBLE, NULL,
   "  The bandwidth parameter.\n"},
  {"do_naive", FX_PARAM, FX_BOOL, NULL,
   "  Whether to perform naive computation as well.\n"},
  {"multiplicative_expansion", FX_PARAM, FX_BOOL, NULL,
   "  Whether to do O(p^D) kernel expansion instead of O(D^p).\n"},
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
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const KdeStat& left_stat,
	      const KdeStat& right_stat) {
      Init();
    }
    
    void Init(const Vector& center, const TKernelAux &ka) {
      
      farfield_expansion_.Init(center, ka);
      local_expansion_.Init(center, ka);
      Init();
    }
    
    KdeStat() { }
    
    ~KdeStat() {}
    
  };
  
 private:

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

  /** @brief The permutation mapping indices of queries_ to original
   *         order.
   */
  ArrayList<index_t> old_from_new_queries_;
  
  /** @brief The permutation mapping indices of references_ to
   *         original order.
   */
  ArrayList<index_t> old_from_new_references_;
  
  /** @brief The exhaustive base KDE case.
   */
  void DualtreeKdeBase_(Tree *qnode, Tree *rnode) {

    // Clear the summary statistics of the current query node so that we
    // can refine it to better bounds.
    qnode->stat().mass_l_ = DBL_MAX;
    qnode->stat().mass_u_ = -DBL_MAX;
    qnode->stat().used_error_ = 0;
    qnode->stat().n_pruned_ = rset_.n_cols();

    // compute unnormalized sum
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      
      // incorporate the postponed information
      densities_l_[q] += qnode->stat().postponed_l_;
      densities_u_[q] += qnode->stat().postponed_u_;
      used_error_[q] += qnode->stat().postponed_used_error_;
      n_pruned_[q] += qnode->stat().postponed_n_pruned_;

      // get query point
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = rnode->begin(); r < rnode->end(); r++) {

	// get reference point
	const double *r_col = rset_.GetColumnPtr(r);

	// pairwise distance and kernel value
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double kernel_value = ka_.kernel_.EvalUnnormOnSq(dsqd);

	densities_l_[q] += kernel_value;
	densities_e_[q] += kernel_value;
	densities_u_[q] += kernel_value;
      } // end of iterating over each reference point.

      // each query point has taken care of all reference points.
      n_pruned_[q] += rnode->count();

      // subtract the number of reference points to undo the assumption made
      // in the function PreProcess.
      densities_u_[q] -= rnode->count();

      // Refine min and max summary statistics.
      qnode->stat().mass_l_ = std::min(qnode->stat().mass_l_, densities_l_[q]);
      qnode->stat().mass_u_ = std::max(qnode->stat().mass_u_, densities_u_[q]);
      qnode->stat().used_error_ = std::max(qnode->stat().used_error_,
					   used_error_[q]);
      qnode->stat().n_pruned_ = std::min(qnode->stat().n_pruned_, 
					 n_pruned_[q]);
    }

    // clear postponed information
    qnode->stat().postponed_l_ = qnode->stat().postponed_u_ = 0;
    qnode->stat().postponed_used_error_ = 0;
    qnode->stat().postponed_n_pruned_ = 0;
  }

  /** 
   * checking for prunability of the query and the reference pair using
   * four types of pruning methods
   */
  bool PrunableEnhanced_(Tree *qnode, Tree *rnode, DRange &dsqd_range,
			 DRange &kernel_value_range, double &dl, double &du,
			 double &used_error, double &n_pruned,
			 int &order_farfield_to_local,
			 int &order_farfield, int &order_local) {

    int dim = rset_.n_rows();

    // actual amount of error incurred per each query/ref pair
    double actual_err_farfield_to_local = 0;
    double actual_err_farfield = 0;
    double actual_err_local = 0;

    // estimated computational cost
    int cost_farfield_to_local = INT_MAX;
    int cost_farfield = INT_MAX;
    int cost_local = INT_MAX;
    int cost_exhaustive = (qnode->count()) * (rnode->count()) * dim;
    int min_cost = 0;

    // query node and reference node statistics
    KdeStat &qstat = qnode->stat();
    KdeStat &rstat = rnode->stat();
    
    // expansion objects
    typename TKernelAux::TFarFieldExpansion &farfield_expansion = 
      rstat.farfield_expansion_;
    typename TKernelAux::TLocalExpansion &local_expansion = 
      qstat.local_expansion_;

    // refine the lower bound using the new lower bound info
    double new_mass_l = qstat.mass_l_ + qstat.postponed_l_ + dl;
    double new_used_error = qstat.used_error_ + qstat.postponed_used_error_;
    double new_n_pruned = qstat.n_pruned_ + qstat.postponed_n_pruned_;
    double allowed_err = (tau_ * new_mass_l - new_used_error) /
      ((double) rroot_->count() - new_n_pruned);
    
    // get the order of approximations
    order_farfield_to_local = 
      farfield_expansion.OrderForConvertingToLocal
      (rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi, 
       allowed_err, &actual_err_farfield_to_local);
    order_farfield = 
      farfield_expansion.OrderForEvaluating(rnode->bound(), qnode->bound(),
					    dsqd_range.lo, dsqd_range.hi,
					    allowed_err, &actual_err_farfield);
    order_local = 
      local_expansion.OrderForEvaluating(rnode->bound(), qnode->bound(), 
					 dsqd_range.lo, dsqd_range.hi,
					 allowed_err, &actual_err_local);

    // update computational cost and compute the minimum
    if(order_farfield_to_local >= 0) {
      cost_farfield_to_local = (int) 
	ka_.sea_.FarFieldToLocalTranslationCost(order_farfield_to_local);
    }
    if(order_farfield >= 0) {
      cost_farfield = (int) 
	ka_.sea_.FarFieldEvaluationCost(order_farfield) * (qnode->count());
    }
    if(order_local >= 0) {
      cost_local = (int) 
	ka_.sea_.DirectLocalAccumulationCost(order_local) * (rnode->count());
    }

    min_cost = min(cost_farfield_to_local, 
		   min(cost_farfield, min(cost_local, cost_exhaustive)));

    if(cost_farfield_to_local == min_cost) {
      used_error = rnode->count() * actual_err_farfield_to_local;
      n_pruned = rnode->count();
      order_farfield = order_local = -1;
      num_farfield_to_local_prunes_++;
      return true;
    }

    if(cost_farfield == min_cost) {
      used_error = rnode->count() * actual_err_farfield;
      n_pruned = rnode->count();
      order_farfield_to_local = order_local = -1;
      num_farfield_prunes_++;
      return true;
    }

    if(cost_local == min_cost) {
      used_error = rnode->count() * actual_err_local;
      n_pruned = rnode->count();
      order_farfield_to_local = order_farfield = -1;
      num_local_prunes_++;
      return true;
    }

    order_farfield_to_local = order_farfield = order_local = -1;
    dl = du = used_error = n_pruned = 0;
    return false;
  }

  /** checking for prunability of the query and the reference pair */
  bool Prunable_(Tree *qnode, Tree *rnode, DRange &dsqd_range,
		 DRange &kernel_value_range, double &dl, double &de,
		 double &du, double &used_error, double &n_pruned) {

    // query node stat
    KdeStat &stat = qnode->stat();
    
    // number of reference points
    int num_references = rnode->count();

    // try pruning after bound refinement: first compute distance/kernel
    // value bounds
    dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
    dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
    kernel_value_range = ka_.kernel_.RangeUnnormOnSq(dsqd_range);

    // the new lower bound after incorporating new info
    dl = kernel_value_range.lo * num_references;
    de = 0.5 * num_references * 
      (kernel_value_range.lo + kernel_value_range.hi);
    du = (kernel_value_range.hi - 1) * num_references;

    // refine the lower bound using the new lower bound info
    double new_mass_l = stat.mass_l_ + stat.postponed_l_ + dl;
    double new_used_error = stat.used_error_ + stat.postponed_used_error_;
    double new_n_pruned = stat.n_pruned_ + stat.postponed_n_pruned_;

    double allowed_err = (tau_ * new_mass_l - new_used_error) *
      num_references / ((double) rroot_->count() - new_n_pruned);

    // this is error per each query/reference pair for a fixed query
    double m = 0.5 * (kernel_value_range.hi - kernel_value_range.lo);

    // this is total error for each query point
    used_error = m * num_references;
    
    // number of reference points for possible pruning.
    n_pruned = rnode->count();

    // check pruning condition
    return (used_error <= allowed_err);
  }

  /** determine which of the node to expand first */
  void BestNodePartners(Tree *nd, Tree *nd1, Tree *nd2, Tree **partner1,
			Tree **partner2) {
    
    double d1 = nd->bound().MinDistanceSq(nd1->bound());
    double d2 = nd->bound().MinDistanceSq(nd2->bound());

    if(d1 <= d2) {
      *partner1 = nd1;
      *partner2 = nd2;
    }
    else {
      *partner1 = nd2;
      *partner2 = nd1;
    }
  }

  /** canonical dualtree KDE case */
  void DualtreeKdeCanonical_(Tree *qnode, Tree *rnode) {
    
    // temporary variable for storing lower bound change.
    double dl = 0, de = 0, du = 0;
    int order_farfield_to_local = -1, order_farfield = -1, order_local = -1;

    // temporary variables for holding used error for pruning.
    double used_error = 0, n_pruned = 0;

    // temporary variable for holding distance/kernel value bounds
    DRange dsqd_range;
    DRange kernel_value_range;

    // try finite difference pruning first
    if(Prunable_(qnode, rnode, dsqd_range, kernel_value_range, dl, de, du,
		 used_error, n_pruned)) {
      qnode->stat().postponed_l_ += dl;
      qnode->stat().postponed_e_ += de;
      qnode->stat().postponed_u_ += du;
      qnode->stat().postponed_used_error_ += used_error;
      qnode->stat().postponed_n_pruned_ += n_pruned;
      num_finite_difference_prunes_++;
      return;
    }
    else if(PrunableEnhanced_(qnode, rnode, dsqd_range, kernel_value_range, 
			      dl, du, used_error, n_pruned, 
			      order_farfield_to_local,
			      order_farfield, order_local)) {
      
      // far field to local translation
      if(order_farfield_to_local >= 0) {
	rnode->stat().farfield_expansion_.TranslateToLocal
	  (qnode->stat().local_expansion_, order_farfield_to_local);
      }
      // far field pruning
      else if(order_farfield >= 0) {
	for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	  densities_e_[q] += 
	    rnode->stat().farfield_expansion_.EvaluateField(qset_, q, 
							    order_farfield);
	}
      }
      // local accumulation pruning
      else if(order_local >= 0) {
	qnode->stat().local_expansion_.AccumulateCoeffs(rset_, rset_weights_,
							rnode->begin(), 
							rnode->end(),
							order_local);
      }
      qnode->stat().postponed_l_ += dl;
      qnode->stat().postponed_u_ += du;
      qnode->stat().postponed_used_error_ += used_error;
      qnode->stat().postponed_n_pruned_ += n_pruned;
      return;
    }

    // for leaf query node
    if(qnode->is_leaf()) {
      
      // for leaf pairs, go exhaustive
      if(rnode->is_leaf()) {
	DualtreeKdeBase_(qnode, rnode);
	return;
      }
      
      // for non-leaf reference, expand reference node
      else {
	Tree *rnode_first = NULL, *rnode_second = NULL;
	BestNodePartners(qnode, rnode->left(), rnode->right(), &rnode_first,
			 &rnode_second);
	DualtreeKdeCanonical_(qnode, rnode_first);
	DualtreeKdeCanonical_(qnode, rnode_second);
	return;
      }
    }
    
    // for non-leaf query node
    else {
      
      // Push down postponed bound changes owned by the current query
      // node to the children of the query node and clear them.
      (qnode->left()->stat()).postponed_l_ += qnode->stat().postponed_l_;
      (qnode->right()->stat()).postponed_l_ += qnode->stat().postponed_l_;
      (qnode->left()->stat()).postponed_u_ += qnode->stat().postponed_u_;
      (qnode->right()->stat()).postponed_u_ += qnode->stat().postponed_u_;
      (qnode->left()->stat()).postponed_used_error_ += 
	qnode->stat().postponed_used_error_;
      (qnode->right()->stat()).postponed_used_error_ += 
	qnode->stat().postponed_used_error_;
      (qnode->left()->stat()).postponed_n_pruned_ += 
	qnode->stat().postponed_n_pruned_;
      (qnode->right()->stat()).postponed_n_pruned_ += 
	qnode->stat().postponed_n_pruned_;
      
      qnode->stat().postponed_l_ = qnode->stat().postponed_u_ = 0;
      qnode->stat().postponed_used_error_ = 0;
      qnode->stat().postponed_n_pruned_ = 0;

      // For a leaf reference node, expand query node
      if(rnode->is_leaf()) {
	Tree *qnode_first = NULL, *qnode_second = NULL;

	BestNodePartners(rnode, qnode->left(), qnode->right(), &qnode_first,
			 &qnode_second);
	DualtreeKdeCanonical_(qnode_first, rnode);
	DualtreeKdeCanonical_(qnode_second, rnode);
      }

      // for non-leaf reference node, expand both query and reference nodes
      else {
	Tree *rnode_first = NULL, *rnode_second = NULL;
	
	BestNodePartners(qnode->left(), rnode->left(), rnode->right(),
			 &rnode_first, &rnode_second);
	DualtreeKdeCanonical_(qnode->left(), rnode_first);
	DualtreeKdeCanonical_(qnode->left(), rnode_second);

	BestNodePartners(qnode->right(), rnode->left(), rnode->right(),
			 &rnode_first, &rnode_second);
	DualtreeKdeCanonical_(qnode->right(), rnode_first);
	DualtreeKdeCanonical_(qnode->right(), rnode_second);
      }

      // reaccumulate the summary statistics.
      qnode->stat().mass_l_ = std::min((qnode->left()->stat()).mass_l_ +
				       (qnode->left()->stat()).postponed_l_,
				       (qnode->right()->stat()).mass_l_ +
				       (qnode->right()->stat()).postponed_l_);
      qnode->stat().mass_u_ = std::max((qnode->left()->stat()).mass_u_ +
				       (qnode->left()->stat()).postponed_u_,
				       (qnode->right()->stat()).mass_u_ +
				       (qnode->right()->stat()).postponed_u_);
      qnode->stat().used_error_ = 
	std::max((qnode->left()->stat()).used_error_ +
		 (qnode->left()->stat()).postponed_used_error_,
		 (qnode->right()->stat()).used_error_ +
		 (qnode->right()->stat()).postponed_used_error_);
      qnode->stat().n_pruned_ = 
	std::min((qnode->left()->stat()).n_pruned_ +
		 (qnode->left()->stat()).postponed_n_pruned_,
		 (qnode->right()->stat()).n_pruned_ +
		 (qnode->right()->stat()).n_pruned_);
      return;
    } // end of the case: non-leaf query node.
  } // end of DualtreeKdeCanonical_

  /** 
   * pre-processing step - this wouldn't be necessary if the core
   * fastlib supported a Init function for Stat objects that take
   * more arguments.
   */
  void PreProcess(Tree *node) {

    // Initialize the center of expansions and bandwidth for series
    // expansion.
    Vector bounding_box_center;
    node->stat().Init(ka_);
    node->bound().CalculateMidpoint(&bounding_box_center);
    (node->stat().farfield_expansion_.get_center())->CopyValues
      (bounding_box_center);
    (node->stat().local_expansion_.get_center())->CopyValues
      (bounding_box_center);

    // initialize lower bound to 0
    node->stat().mass_l_ = 0;
    
    // set the upper bound to the number of reference points
    node->stat().mass_u_ = rset_.n_cols();

    node->stat().used_error_ = 0;
    node->stat().n_pruned_ = 0;

    // postponed lower and upper bound density changes to 0
    node->stat().postponed_l_ = node->stat().postponed_u_ = 0;

    // set the finite difference approximated amounts to 0
    node->stat().postponed_e_ = 0;

    // set the error incurred to 0
    node->stat().postponed_used_error_ = 0;

    // set the number of pruned reference points to 0
    node->stat().postponed_n_pruned_ = 0;
    
    // for non-leaf node, recurse
    if(!node->is_leaf()) {
     
      PreProcess(node->left());
      PreProcess(node->right());

      // translate multipole moments
      node->stat().farfield_expansion_.TranslateFromFarField
	(node->left()->stat().farfield_expansion_);
      node->stat().farfield_expansion_.TranslateFromFarField
	(node->right()->stat().farfield_expansion_);
    }
    else {
      
      // exhaustively compute multipole moments
      node->stat().farfield_expansion_.RefineCoeffs(rset_, rset_weights_,
						    node->begin(), node->end(),
						    ka_.sea_.get_max_order());
    }
  }

  /** post processing step */
  void PostProcess(Tree *qnode) {
    
    KdeStat &qstat = qnode->stat();

    // for leaf query node
    if(qnode->is_leaf()) {
      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	densities_l_[q] += qstat.postponed_l_;
	densities_e_[q] += qstat.local_expansion_.EvaluateField(qset_, q) +
	  qstat.postponed_e_;
	densities_u_[q] += qstat.postponed_u_;

	// normalize densities
	densities_l_[q] *= mult_const_;
	densities_e_[q] *= mult_const_;
	densities_u_[q] *= mult_const_;
      }
    }
    else {

      // push down approximations
      (qnode->left()->stat()).postponed_l_ += qstat.postponed_l_;
      (qnode->right()->stat()).postponed_l_ += qstat.postponed_l_;
      (qnode->left()->stat()).postponed_e_ += qstat.postponed_e_;
      (qnode->right()->stat()).postponed_e_ += qstat.postponed_e_;
      qstat.local_expansion_.TranslateToLocal
	(qnode->left()->stat().local_expansion_);
      qstat.local_expansion_.TranslateToLocal
	(qnode->right()->stat().local_expansion_);
      (qnode->left()->stat()).postponed_u_ += qstat.postponed_u_;
      (qnode->right()->stat()).postponed_u_ += qstat.postponed_u_;

      PostProcess(qnode->left());
      PostProcess(qnode->right());
    }
  }

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

  // getters and setters

  /** get the density estimate */
  void get_density_estimates(Vector *results) { 
    results->Init(densities_e_.length());
    
    for(index_t i = 0; i < densities_e_.length(); i++) {
      (*results)[i] = densities_e_[i];
    }
  }

  // interesting functions...

  void Compute(Vector *results) {

    // compute normalization constant
    mult_const_ = 1.0 / (ka_.kernel_.CalcNormConstant(qset_.n_rows()) *
			 rset_.n_cols());

    // set accuracy parameter
    tau_ = fx_param_double(module_, "relative_error", 0.01);
    
    // initialize the lower and upper bound densities
    densities_l_.SetZero();
    densities_e_.SetZero();
    densities_u_.SetAll(rset_.n_cols());

    // set zero for error accounting stuff
    used_error_.SetZero();
    n_pruned_.SetZero();

    // reset prune statistics.
    num_finite_difference_prunes_ = num_farfield_to_local_prunes_ =
      num_farfield_prunes_ = num_local_prunes_ = 0;

    printf("\nStarting fast KDE...\n");
    fx_timer_start(NULL, "fast_kde_compute");

    // preprocessing step for initializing series expansion objects
    PreProcess(rroot_);
    if(qroot_ != rroot_) {
      PreProcess(qroot_);
    }
    
    // call main routine
    DualtreeKdeCanonical_(qroot_, rroot_);

    // postprocessing step for finalizing the sums
    PostProcess(qroot_);
    fx_timer_stop(NULL, "fast_kde_compute");
    printf("\nFast KDE completed...\n");
    printf("Finite difference prunes: %d\n", num_finite_difference_prunes_);
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

  void Init(Matrix &queries, Matrix &references, 
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

    // initialize the error accounting stufff
    used_error_.Init(qset_.n_cols());
    n_pruned_.Init(qset_.n_cols());

    // initialize the kernel
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

    if((fname = fx_param_str(module_, "fast_kde_output", NULL)) != NULL) {
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

#endif
