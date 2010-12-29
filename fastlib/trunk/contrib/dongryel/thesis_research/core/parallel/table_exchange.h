/** @file table_exchange.h
 *
 *  A class to do a set of all-to-all table exchanges.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_TABLE_EXCHANGE_H
#define CORE_PARALLEL_TABLE_EXCHANGE_H

#include <boost/mpi.hpp>

namespace core {
namespace parallel {
class TableExchange {
  public:

    template<typename TableType, typename SubTableListType>
    static void AllToAll(
      boost::mpi::communicator &world,
      int max_num_levels_to_serialize,
      TableType &local_table,
      const std::vector< std::vector< std::pair<int, int> > > &receive_requests,
      std::vector< SubTableListType > *received_subtables) {

      // The gathered request lists to send to each process.
      std::vector< std::vector< std::pair<int, int> > > send_requests;

      // Each process gathers the list of requests: (node
      // begin, node count) pairs.
      boost::mpi::all_to_all(
        world, receive_requests, send_requests);

      // Prepare the list of subtables, and do another all_to_all.
      std::vector< SubTableListType > send_subtables;
      send_subtables.resize(send_requests.size());
      for(unsigned int j = 0; j < send_requests.size(); j++) {
        for(unsigned int i = 0; i < send_requests[j].size(); i++) {
          int begin = send_requests[j][i].first;
          int count = send_requests[j][i].second;
          send_subtables[j].push_back(
            &local_table,
            local_table.get_tree()->FindByBeginCount(begin, count),
            max_num_levels_to_serialize);
        }
      }
      received_subtables->resize(world.size());
      for(unsigned int j = 0; j < receive_requests.size(); j++) {
        for(unsigned int i = 0; i < receive_requests[j].size(); i++) {
          (*received_subtables)[j].push_back(j, max_num_levels_to_serialize);
        }
      }
      boost::mpi::all_to_all(world, send_subtables, *received_subtables);
    }
};
};
};

#endif
