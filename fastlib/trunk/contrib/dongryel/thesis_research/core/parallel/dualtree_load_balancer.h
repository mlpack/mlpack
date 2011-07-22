/** @file dualtree_load_balancer.h
 *
 *  @author Dongryeol Lee (dongryel@.cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DUALTREE_LOAD_BALANCER_H
#define CORE_PARALLEL_DUALTREE_LOAD_BALANCER_H

#include <boost/mpi.hpp>

namespace core {
namespace parallel {
class DualtreeLoadBalancer {
  public:

    template<typename TreeType>
    static void Compute(
      boost::mpi::communicator &comm,
      const std::vector<TreeType *> &local_query_subtrees,
      const std::vector <
      std::vector <
      std::pair<int, int> > > &essential_reference_subtrees_to_send,
      const std::vector <
      std::vector <
      std::pair<int, int> > > &reference_subtrees_to_receive,
      int num_reference_subtrees_to_receive) {

      // Figure out the subtree distribution by doing an all-gather.
      boost::scoped_array<int> query_subtree_distribution(
        new int[ comm.size()]);
      boost::scoped_array<int> distribution_prefix_sum_before(
        new int[ comm.size() + 1]);
      boost::mpi::all_gather(
        comm,
        static_cast<int>(local_query_subtrees.size()),
        query_subtree_distribution.get());
      distribution_prefix_sum_before[0] = 0;
      for(int i = 1; i < comm.size(); i++) {
        distribution_prefix_sum_before[i] =
          distribution_prefix_sum_before[i - 1] +
          query_subtree_distribution[i - 1];
      }
      int total_num_query_subtrees =
        distribution_prefix_sum_before[ comm.size() - 1 ] +
        query_subtree_distribution[ comm.size() - 1 ];
      distribution_prefix_sum_before[ comm.size()] = total_num_query_subtrees;

      // Query subtree distribution.
      int num_query_subtree_per_process =
        total_num_query_subtrees / comm.size();
      boost::scoped_array<int> distribution_prefix_sum_after(
        new int[comm.size() + 1]);
      distribution_prefix_sum_after[0] = 0;
      int remainder = total_num_query_subtrees % comm.size();

      int process_id = -1;
      for(int i = 0; i < comm.size(); i++) {
        int portion = num_query_subtree_per_process;
        if(i < remainder) {
          portion++;
        }
        distribution_prefix_sum_after[i + 1] =
          portion + distribution_prefix_sum_after[i];
        if(distribution_prefix_sum_before[ comm.rank()] >=
            distribution_prefix_sum_after[i] &&
            distribution_prefix_sum_before[ comm.rank()] <
            distribution_prefix_sum_after[i + 1]) {
          process_id = i;
        }
      }

      // Each process assigns its local query subtrees. Right now,
      // just assign an equal number of query subtrees across all MPI
      // processes.
      std::vector<int> local_query_subtree_assignments(
        local_query_subtrees.size(), 0);
      for(unsigned int i = 0; i < local_query_subtrees.size(); i++) {
        int local_query_subtree_global_id =
          i + distribution_prefix_sum_before[ comm.rank()];
        if(local_query_subtree_global_id >=
            distribution_prefix_sum_after[process_id + 1]) {
          process_id++;
        }
        local_query_subtree_assignments[i] = process_id;
      }
    }
};
}
}

#endif
