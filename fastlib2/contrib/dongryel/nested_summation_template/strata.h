#ifndef STRATA_H
#define STRATA_H

class Strata {

 public:

  index_t total_num_stratum;

  ArrayList<index_t> num_samples_for_each_stratum;

  ArrayList<index_t> percentage_of_terms_in_each_stratum;

  index_t total_num_samples_so_far;

  Matrix statistics_for_each_stratum;

  index_t total_samples_to_allocate;

  ArrayList<index_t> output_allocation_for_each_stratum;

};

#endif
