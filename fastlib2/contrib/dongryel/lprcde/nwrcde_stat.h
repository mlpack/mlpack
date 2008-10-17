#ifndef NWRCDE_STAT_H
#define NWRCDE_STAT_H

#include "nwrcde_query_postponed.h"
#include "nwrcde_query_summary.h"

template<typename TKernelAux>
class NWRCdeQueryStat {
 public:

  NWRCdeQueryPostponed<TKernelAux> postponed;

  NWRCdeQuerySummary summary;

  OT_DEF_BASIC(NWRCdeQueryStat) {
    OT_MY_OBJECT(postponed);
    OT_MY_OBJECT(summary);
  }

 public:

  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const NWRCdeQueryStat& left_stat, 
	    const NWRCdeQueryStat& right_stat) {
  }
    
};

template<typename TKernelAux>
class NWRCdeReferenceStat {
 public:

  /** @brief The far field expansion created by the reference points
   *         in this node.
   */
  typename TKernelAux::TFarFieldExpansion farfield_expansion;

  double sum_of_target_values;

  OT_DEF_BASIC(NWRCdeReferenceStat) {
    OT_MY_OBJECT(farfield_expansion);
    OT_MY_OBJECT(sum_of_target_values);
  }

 public:

  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const NWRCdeReferenceStat& left_stat, 
	    const NWRCdeReferenceStat& right_stat) {
    
  }

  template<typename TBound>
  void PostInitCommon(const TBound &bounding_primitive,
		      const TKernelAux &kernel_aux_in) {
    
    // Initialize the center of expansions and bandwidth for series
    // expansion.
    Vector bounding_box_center;
    farfield_expansion.Init(kernel_aux_in);
    bounding_primitive.CalculateMidpoint(&bounding_box_center);
    (farfield_expansion.get_center())->CopyValues(bounding_box_center);
  }

  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for a leaf node.
   */
  template<typename TBound>
  void PostInit(const TBound &bounding_primitive,
		const TKernelAux &kernel_aux_in,
		const Matrix &reference_set, const Vector& targets, 
		index_t start, index_t count) {

    PostInitCommon(bounding_primitive, kernel_aux_in);

    // Exhaustively compute multipole moments.
    farfield_expansion.AccumulateCoeffs(reference_set, targets,
					start, start + count,
					kernel_aux_in.sea_.get_max_order());

    sum_of_target_values = 0;
    for(index_t i = start; i < start + count; i++) {
      sum_of_target_values += targets[i];
    }
  }
    
  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for an internal node.
   */
  template<typename TBound>
  void PostInit(const TBound &bounding_primitive,
		const TKernelAux &kernel_aux_in,
		const Vector &targets, index_t start, index_t count, 
		const NWRCdeReferenceStat& left_stat, 
		const NWRCdeReferenceStat& right_stat) {

    PostInitCommon(bounding_primitive, kernel_aux_in);
    
    // Translate the moments up from the two children's moments.
    farfield_expansion.TranslateFromFarField(left_stat.farfield_expansion);
    farfield_expansion.TranslateFromFarField(right_stat.farfield_expansion);

    sum_of_target_values = left_stat.sum_of_target_values +
      right_stat.sum_of_target_values;
  }

};

#endif
