#ifndef NWRCDE_STAT_H
#define NWRCDE_STAT_H

#include "nwrcde_query_postponed.h"
#include "nwrcde_query_summary.h"

class NWRCdeQueryStat {
 public:

  NWRCdeQueryPostponed postponed;

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

class NWRCdeReferenceStat {
 public:

  double sum_of_target_values;

  OT_DEF_BASIC(NWRCdeReferenceStat) {
    OT_MY_OBJECT(sum_of_target_values);
  }

 public:

  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const NWRCdeReferenceStat& left_stat, 
	    const NWRCdeReferenceStat& right_stat) {
    
  }

  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for a leaf node.
   */
  void PostInit(const Vector& targets, index_t start, index_t count) {
    sum_of_target_values = 0;
    for(index_t i = start; i < start + count; i++) {
      sum_of_target_values += targets[i];
    }
  }
    
  /** @brief Computes the sum of the target values owned by the
   *         reference statistics for an internal node.
   */
  void PostInit(const Vector &targets, index_t start, index_t count, 
		const NWRCdeReferenceStat& left_stat, 
		const NWRCdeReferenceStat& right_stat) {
    sum_of_target_values = left_stat.sum_of_target_values +
      right_stat.sum_of_target_values;
  }

};

#endif
