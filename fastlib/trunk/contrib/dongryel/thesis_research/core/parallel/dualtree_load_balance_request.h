/** @file dualtree_load_balance_request.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DUALTREE_LOAD_BALANCE_REQUEST_H
#define CORE_PARALLEL_DUALTREE_LOAD_BALANCE_REQUEST_H

#include <boost/intrusive_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace core {
namespace parallel {

template <
typename DistributedTableType,
         typename TaskPriorityQueueType >
class DistributedDualtreeTaskQueue;

template <
typename DistributedTableType,
         typename TaskPriorityQueueType >
class DualtreeLoadBalanceRequest {
  private:

    typedef typename DistributedTableType::SubTableType SubTableType;

    typedef typename SubTableType::SubTableIDType SubTableIDType;

    typedef class DistributedDualtreeTaskQueue <
      DistributedTableType,
        TaskPriorityQueueType > DistributedDualtreeTaskQueueType;

    typedef
    typename DistributedDualtreeTaskQueueType::QuerySubTableLockListType QuerySubTableLockListType;

    std::vector< SubTableIDType > existing_query_subtables_;

    unsigned long int remaining_local_computation_;

    unsigned long int remaining_extra_points_to_hold_;

  public:

    unsigned long int remaining_local_computation() const {
      return remaining_local_computation_;
    }

    unsigned long int remaining_extra_points_to_hold() const {
      return remaining_extra_points_to_hold_;
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      int num_ids_to_serialize = existing_query_subtables_.size();
      ar & remaining_local_computation_;
      ar & remaining_extra_points_to_hold_;
      ar & num_ids_to_serialize;
      for(unsigned int i = 0; i < existing_query_subtables_.size(); i++) {
        ar & existing_query_subtables_[i].get<0>();
        ar & existing_query_subtables_[i].get<1>();
        ar & existing_query_subtables_[i].get<2>();
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      ar & remaining_local_computation_;
      ar & remaining_extra_points_to_hold_;
      int num_ids_to_load;
      ar & num_ids_to_load;
      existing_query_subtables_.resize(num_ids_to_load);
      for(int i = 0; i < num_ids_to_load; i++) {
        ar & existing_query_subtables_[i].get<0>();
        ar & existing_query_subtables_[i].get<1>();
        ar & existing_query_subtables_[i].get<2>();
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    DualtreeLoadBalanceRequest() {
      remaining_local_computation_ = 0;
      remaining_extra_points_to_hold_ = 0;
    }

    void Init(
      const std::vector <
      boost::intrusive_ptr< SubTableType > > &query_subtables_in,
      const QuerySubTableLockListType &checked_out_query_subtables_in,
      unsigned long int remaining_local_computation_in,
      unsigned long int remaining_extra_points_to_hold_in) {

      // Set the current estimated remaining local work.
      remaining_local_computation_ = remaining_local_computation_in;

      // Set the maximum number of points to receive.
      remaining_extra_points_to_hold_ = remaining_extra_points_to_hold_in;

      // Copy the subtable IDs.
      for(unsigned int i = 0; i < query_subtables_in.size(); i++) {
        existing_query_subtables_.push_back(
          query_subtables_in[i]->subtable_id());
      }
      for(typename QuerySubTableLockListType::const_iterator it =
            checked_out_query_subtables_in.begin();
          it != checked_out_query_subtables_in.end(); it++) {
        existing_query_subtables_.push_back((*it)->subtable_id());
      }
    }
};
}
}

#endif
