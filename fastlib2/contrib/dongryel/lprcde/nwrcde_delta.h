#ifndef NWRCDE_DELTA_H
#define NWRCDE_DELTA_H

#include "fastlib/fastlib.h"

class NWRCdeDelta {
  
 public:
    
  /** @brief The approximation method types.
   */
  enum ApproximationType {
    
    /** @brief Exhaustive method.
     */
    EXHAUSTIVE,

    /** @brief Finite difference method.
     */
    FINITE_DIFFERENCE,

    /** @brief Far-to-local translation method.
     */
    FAR_TO_LOCAL,

    /** @brief Direct far-field evaluation method.
     */
    DIRECT_FARFIELD,
    
    /** @brief Direct local accumulation method.
     */
    DIRECT_LOCAL
  };

  class NWRCdeApproximation {

   public:

    /** @brief The decided approximation type.
     */
    ApproximationType approx_type;
    
    int order_farfield_to_local;

    int order_farfield;

    int order_local;

    int cost_farfield_to_local;

    int cost_farfield;

    int cost_local;

    int cost_exhaustive;

    int min_cost;

    double actual_err_farfield_to_local;

    double actual_err_farfield;
    
    double actual_err_local;
    
    double sum_l;
    double sum_e;
    double n_pruned;
    double used_error;

    OT_DEF_BASIC(NWRCdeApproximation) {
      OT_MY_OBJECT(approx_type);
      OT_MY_OBJECT(order_farfield_to_local);
      OT_MY_OBJECT(order_farfield);
      OT_MY_OBJECT(order_local);
      OT_MY_OBJECT(cost_farfield_to_local);
      OT_MY_OBJECT(cost_farfield);
      OT_MY_OBJECT(cost_local);
      OT_MY_OBJECT(cost_exhaustive);
      OT_MY_OBJECT(min_cost);
      OT_MY_OBJECT(actual_err_farfield_to_local);
      OT_MY_OBJECT(actual_err_farfield);
      OT_MY_OBJECT(actual_err_local);
      OT_MY_OBJECT(sum_l);
      OT_MY_OBJECT(sum_e);
      OT_MY_OBJECT(n_pruned);
      OT_MY_OBJECT(used_error);
    }

   public:

    template<typename TGlobal, typename QueryTree, typename ReferenceTree>
    void Init(const TGlobal &parameters, QueryTree *qnode,
	      ReferenceTree *rnode) {
      
      // Initialize the member variables.
      Reset();
      cost_exhaustive = parameters.rset.n_rows() * qnode->count() * 
	rnode->count();
    }

    void Reset() {
      approx_type = EXHAUSTIVE;
      order_farfield_to_local = -1;
      order_farfield = -1;
      order_local = -1;
      cost_farfield_to_local = INT_MAX;
      cost_farfield = INT_MAX;
      cost_local = INT_MAX;
      cost_exhaustive = INT_MAX;
      min_cost = INT_MAX;
      actual_err_farfield_to_local = DBL_MAX;
      actual_err_farfield = DBL_MAX;
      actual_err_local = DBL_MAX;
      sum_l = 0;
      sum_e = 0;
      n_pruned = 0;
      used_error = 0;
    }

  };

  DRange dsqd_range;
  DRange kernel_value_range;
  NWRCdeApproximation nwr_numerator;
  NWRCdeApproximation nwr_denominator;

  OT_DEF_BASIC(NWRCdeDelta) {
    OT_MY_OBJECT(dsqd_range);
    OT_MY_OBJECT(kernel_value_range);
    OT_MY_OBJECT(nwr_numerator);
    OT_MY_OBJECT(nwr_denominator);
  }

 public:
  
  template<typename TFarFieldExpansion, typename TLocalExpansion,
	   typename TGlobal, typename QueryTree, typename ReferenceTree, 
	   typename TErrorComponent>
  void DecideSeriesApproximationMethod
  (const TGlobal &parameters,
   QueryTree *qnode, TLocalExpansion &local_expansion, ReferenceTree *rnode,
   TFarFieldExpansion &farfield_expansion,
   const TErrorComponent &allowed_error,
   NWRCdeApproximation &approximation) {
    
    approximation.order_farfield_to_local =
      farfield_expansion.OrderForConvertingToLocal
      (rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi, 
       allowed_error.error_per_pair,
       &(approximation.actual_err_farfield_to_local));
    approximation.order_farfield = 
      farfield_expansion.OrderForEvaluating
      (rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi,
       allowed_error.error_per_pair,
       &(approximation.actual_err_farfield));
    approximation.order_local = 
      local_expansion.OrderForEvaluating
      (rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi,
       allowed_error.error_per_pair, 
       &(approximation.actual_err_local));
    
    if(approximation.order_farfield_to_local >= 0) {
      approximation.cost_farfield_to_local = (int) 
	parameters.kernel_aux.sea_.FarFieldToLocalTranslationCost
	(approximation.order_farfield_to_local);
    }
    if(approximation.order_farfield >= 0) {
      approximation.cost_farfield = (int) 
	parameters.kernel_aux.sea_.FarFieldEvaluationCost
	(approximation.order_farfield) * (qnode->count());
    }
    if(approximation.order_local >= 0) {
      approximation.cost_local = (int) 
	parameters.kernel_aux.sea_.DirectLocalAccumulationCost
	(approximation.order_local) * (rnode->count());
    }    
    approximation.min_cost = min(approximation.cost_farfield_to_local, 
				 min(approximation.cost_farfield, 
				     min(approximation.cost_local,
					 approximation.cost_exhaustive)));

    // Decide which method to use.
    if(approximation.cost_farfield_to_local == approximation.min_cost) {

      approximation.approx_type = FAR_TO_LOCAL;
      approximation.used_error = farfield_expansion.get_weight_sum() * 
	approximation.actual_err_farfield_to_local;
      approximation.n_pruned = farfield_expansion.get_weight_sum();
      approximation.order_farfield = approximation.order_local = -1;
      approximation.sum_e = 0;
    }
    
    if(approximation.cost_farfield == approximation.min_cost) {

      approximation.approx_type = DIRECT_FARFIELD;
      approximation.used_error = farfield_expansion.get_weight_sum() * 
	approximation.actual_err_farfield;
      approximation.n_pruned = farfield_expansion.get_weight_sum();
      approximation.order_farfield_to_local = approximation.order_local = -1;
      approximation.sum_e = 0;
    }
    
    if(approximation.cost_local == approximation.min_cost) {

      approximation.approx_type = DIRECT_LOCAL;
      approximation.used_error = farfield_expansion.get_weight_sum() * 
	approximation.actual_err_local;
      approximation.n_pruned = farfield_expansion.get_weight_sum();
      approximation.order_farfield_to_local = 
	approximation.order_farfield = -1;
      approximation.sum_e = 0;
    }
  }

  template<typename TGlobal, typename QueryTree, typename ReferenceTree,
	   typename TError>
  bool ComputeSeriesExpansion
  (const TGlobal &parameters, QueryTree *qnode, ReferenceTree *rnode,
   const TError &allowed_error) {
    
    // Decide methods for both the numerator and the denominator.
    DecideSeriesApproximationMethod
      (parameters, qnode, qnode->stat().nwr_numerator_local_expansion, rnode, 
       rnode->stat().nwr_numerator_farfield_expansion,
       allowed_error.nwr_numerator, nwr_numerator);
    DecideSeriesApproximationMethod
      (parameters, qnode, qnode->stat().nwr_denominator_local_expansion, rnode,
       rnode->stat().nwr_denominator_farfield_expansion,
       allowed_error.nwr_denominator, nwr_denominator);

    return (nwr_numerator.approx_type != EXHAUSTIVE) &&
      (nwr_denominator.approx_type != EXHAUSTIVE);
  }
  
  template<typename TGlobal, typename QueryTree, typename ReferenceTree>
  void ComputeFiniteDifference
  (const TGlobal &parameters, QueryTree *qnode, ReferenceTree *rnode) {

    double finite_difference_error;

    dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
    dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
    kernel_value_range = 
      parameters.kernel_aux.kernel_.RangeUnnormOnSq(dsqd_range);

    finite_difference_error = 0.5 * kernel_value_range.width();

    // Compute the bound delta changes based on the kernel value range.
    nwr_numerator.sum_l = rnode->stat().sum_of_target_values * 
      kernel_value_range.lo;
    nwr_denominator.sum_l = rnode->count() * kernel_value_range.lo;
    nwr_numerator.sum_e = rnode->stat().sum_of_target_values *
      kernel_value_range.mid();
    nwr_denominator.sum_e = rnode->count() * kernel_value_range.mid();
    nwr_numerator.n_pruned = rnode->stat().sum_of_target_values;
    nwr_denominator.n_pruned = rnode->count();
    nwr_numerator.used_error = rnode->stat().sum_of_target_values * 
      finite_difference_error;
    nwr_denominator.used_error = rnode->count() * finite_difference_error;
  }
  
  template<typename TKernelAux, typename TGlobal,
	   typename QueryTree, typename ReferenceTree>
  void SetZero(const TGlobal &parameters, QueryTree *qnode,
	       ReferenceTree *rnode) {

    dsqd_range.Reset(DBL_MAX, -DBL_MAX);
    kernel_value_range.Reset(DBL_MAX, -DBL_MAX);
    nwr_numerator.Init(parameters, qnode, rnode);
    nwr_denominator.Init(parameters.qnode, rnode);
  }
  
};

#endif
