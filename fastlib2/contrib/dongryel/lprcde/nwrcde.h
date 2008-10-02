/** @file nwrcde.h
 *
 *  This file contains an implementation of Nadaraya-Watson regression
 *  and conditional density estimation for a linkable library
 *  component. It implements a rudimentary depth-first dual-tree
 *  algorithm with finite difference and series-expansion
 *  approximations, using the formalized GNP framework by Ryan and
 *  Garry.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see nwrcde_main.cc
 *  @bug No known bugs.
 */

#ifndef NWRCDE_H
#define NWRCDE_H

#define INSIDE_NWRCDE_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/mult_farfield_expansion.h"
#include "mlpack/series_expansion/mult_local_expansion.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "contrib/dongryel/proximity_project/gen_metric_tree.h"
#include "contrib/dongryel/proximity_project/subspace_stat.h"

////////// Documentation stuffs //////////
const fx_entry_doc nwrcde_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   "  A file containing reference data.\n"},
  {"query", FX_PARAM, FX_STR, NULL,
   "  A file containing query data (defaults to data).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_entry_doc nwrcde_entries[] = {
  {"bandwidth", FX_PARAM, FX_DOUBLE, NULL,
   "  The bandwidth parameter.\n"},
  {"coverage_percentile", FX_PARAM, FX_DOUBLE, NULL,
   "  The upper percentile of the estimates for the error guarantee.\n"},
  {"do_naive", FX_PARAM, FX_BOOL, NULL,
   "  Whether to perform naive computation as well.\n"},
  {"output", FX_PARAM, FX_STR, NULL,
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

const fx_module_doc nwrcde_doc = {
  nwrcde_entries, NULL,
  "Performs dual-tree kernel density estimate computation.\n"
};

const fx_submodule_doc nwrcde_main_submodules[] = {
  {"nwrcde", &nwrcde_doc,
   "  Responsible for Nadaraya-Watson regression and conditional density
  estimate computation.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc nwrcde_main_doc = {
  nwrcde_main_entries, nwrcde_main_submodules,
  "This is the driver for the kernel density estimator.\n"
};


template<typename TKernel>
class NWRCde {

 public:
    
 private:

  ////////// Private Constants //////////

  ////////// Private Member Variables //////////

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module_;

  /** @brief The boolean flag to control the leave-one-out computation.
   */
  bool leave_one_out_;

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
  NWRCde() {
    qroot_ = rroot_ = NULL;
  }

  /** @brief The default destructor which deletes the trees.
   */
  ~NWRCde() { 
    
    if(qroot_ != rroot_ ) {
      delete qroot_; 
      delete rroot_; 
    } 
    else {
      delete rroot_;
    }

  }

  ////////// User Level Functions //////////

  void Compute(Vector *regression_estimates) {

    // Initialize the temporary sum accumulators to zero.
    nwr_numerator_sum_l_.SetZero();
    nwr_numerator_sum_e_.SetZero();
    
    
  }

  void Init(const Matrix &queries, const Matrix &references,
	    const Matrix &rset_weights, bool queries_equal_references, 
	    struct datanode *module_in) {

    
  }

  void PrintDebug();

};

#include "nwrcde_impl.h"
#undef INSIDE_NWRCDE_H

#endif
