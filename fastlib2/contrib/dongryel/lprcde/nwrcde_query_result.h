#ifndef NWRCDE_QUERY_RESULT_H
#define NWRCDE_QUERY_RESULT_H

class NWRCdeQueryResult {
  
 public:
  
  Vector nwr_numerator_sum_l;
  Vector nwr_numerator_sum_e;
  Vector nwr_denominator_sum_l;
  Vector nwr_denominator_sum_e;
  Vector n_pruned;
  
 public:
  
  void Init(int num_queries) {
    
    // Initialize the space.
    nwr_numerator_sum_l.Init(num_queries);
    nwr_numerator_sum_e.Init(num_queries);
    nwr_denominator_sum_l.Init(num_queries);
    nwr_denominator_sum_e.Init(num_queries);
    n_pruned.Init(num_queries);
    
    // Reset the sums to zero.
    SetZero();
  }
  
  void SetZero() {
    nwr_numerator_sum_l.SetZero();
    nwr_numerator_sum_e.SetZero();
    nwr_denominator_sum_l.SetZero();
    nwr_denominator_sum_e.SetZero();
    n_pruned.SetZero();
  }
  
};

#endif
