#ifndef FMM_STAT_H
#define FMM_STAT_H

#include "mlpack/series_expansion/inverse_pow_dist_farfield_expansion.h"
#include "mlpack/series_expansion/inverse_pow_dist_local_expansion.h"

class FmmStat {
 public:

  /** @brief The far field expansion created by the reference points
   *         in this node.
   */
  InversePowDistFarFieldExpansion farfield_expansion_;
    
  /** @brief The local expansion stored in this node.
   */
  InversePowDistLocalExpansion local_expansion_;
    
  FmmStat() { }
    
  ~FmmStat() { }
    
};

#endif
