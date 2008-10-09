#ifndef NWRCDE_QUERY_POSTPONED_H
#define NWRCDE_QUERY_POSTPONED_H

class NWRCdeQueryPostponed {
  
 public:
  
  double nwr_numerator_sum_l;
  double nwr_numerator_sum_e;
  double nwr_denominator_sum_l;
  double nwr_denominator_sum_e;
  double n_pruned;
  
 public:
  
  void Init(int num_queries) {
    
    // Reset the postponed quantities to zero.
    SetZero();
  }
  
  void SetZero() {
    nwr_numerator_sum_l = 0;
    nwr_numerator_sum_e = 0;
    nwr_denominator_sum_l = 0;
    nwr_denominator_sum_e = 0;
    n_pruned = 0;
  }
  
};

#endif
