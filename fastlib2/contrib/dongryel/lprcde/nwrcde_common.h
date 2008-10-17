#ifndef NWRCDE_COMMON_H
#define NWRCDE_COMMON_H

#include "fastlib/fastlib.h"

#include "nwrcde_delta.h"
#include "nwrcde_error.h"
#include "nwrcde_global.h"
#include "nwrcde_query_summary.h"

class NWRCdeCommon {

 public:

  /** @brief Shuffles a vector according to a given permutation.
   *
   *  @param v The vector to be shuffled.
   *  @param permutation The permutation.
   */
  static void ShuffleAccordingToPermutation
  (Vector &v, const ArrayList<index_t> &permutation) {
    
    Vector v_tmp;
    v_tmp.Init(v.length());
    for(index_t i = 0; i < v_tmp.length(); i++) {
      v_tmp[i] = v[permutation[i]];
    }
    v.CopyValues(v_tmp);
  }

  template<typename Tree1, typename Tree2>
  static void Heuristic
  (Tree1 *nd, Tree2 *nd1, Tree2 *nd2, double probability, 
   Tree2 **partner1, double *probability1, 
   Tree2 **partner2, double *probability2) {
    
    double d1 = nd->bound().MinDistanceSq(nd1->bound());
    double d2 = nd->bound().MinDistanceSq(nd2->bound());
    
    // Prioritized traversal based on the squared distance bounds.
    if(d1 <= d2) {
      *partner1 = nd1;
      *probability1 = sqrt(probability);
      *partner2 = nd2;
      *probability2 = sqrt(probability);
    }
    else {
      *partner1 = nd2;
      *probability1 = sqrt(probability);
      *partner2 = nd1;
      *probability2 = sqrt(probability);
    }
  }

  template<typename TKernelAux, typename QueryTree, typename ReferenceTree>
  static bool ConsiderPairExact
  (const NWRCdeGlobal<TKernelAux, ReferenceTree> &parameters,
   QueryTree *qnode, ReferenceTree *rnode, double probability, 
   NWRCdeDelta &delta) {
    
    // Refine the lower bound using the new lower bound info.
    NWRCdeQuerySummary new_summary;
    new_summary.InitCopy(qnode->stat().summary);
    new_summary.ApplyPostponed(qnode->stat().postponed);
    new_summary.ApplyDelta(delta);
    
    // Compute the allowable error.
    NWRCdeError allowed_error;
    allowed_error.ComputeAllowableError(parameters, new_summary, rnode);

    // If the error bound is satisfied by the hard error bound, it is
    // safe to prune.
    return (!isnan(allowed_error.nwr_numerator_error)) && 
      (!isnan(allowed_error.nwr_denominator_error)) &&
      (delta.nwr_numerator_used_error <= allowed_error.nwr_numerator_error) &&
      (delta.nwr_denominator_used_error <= 
       allowed_error.nwr_denominator_error);
  }

};

#endif
