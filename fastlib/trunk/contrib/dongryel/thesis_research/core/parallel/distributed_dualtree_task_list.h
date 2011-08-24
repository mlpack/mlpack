/** @file distributed_dualtree_task_list.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H

#include <boost/serialization/serialization.hpp>
#include <map>
#include <vector>
#include "core/parallel/distributed_dualtree_task.h"
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template<typename TaskPriorityQueueType>
class DistributedDualtreeTaskList {

  public:

    typedef typename TaskPriorityQueueType::value_type TaskType;

    typedef typename TaskType::TableType TableType;

    typedef core::table::SubTable<TableType> SubTableType;

    typedef typename TableType::TreeType TreeType;

  private:

    // std::map< boost::tuple<int, int, int>, boost::tuple<int, int, int> > id_to_position_map_;

    unsigned long int remaining_extra_points_to_hold_;

    std::vector<SubTableType> sub_tables_;

  private:

    bool FindSubTable_(int rank_in, int begin_in, int count_in) {
      return true;
    }

    /** @brief Returns true if the subtable can be transferred within
     *         the limit.
     */
    bool push_back_(
      TableType *test_subtable_in, TreeType *test_start_node_in) {
      if(FindSubTable_(
            test_subtable_in->rank(), test_start_node_in->begin(),
            test_start_node_in->count())) {

      }
      else if(test_start_node_in->count() <= remaining_extra_points_to_hold_) {
        sub_tables_.resize(sub_tables_.size() + 1);
        sub_tables_.Alias();
        remaining_extra_points_to_hold_ -= test_start_node_in->count();
        return true;
      }
      return false;
    }

  public:

    DistributedDualtreeTaskList() {
      remaining_extra_points_to_hold_ = 0;
    }

    void Init(unsigned long int remaining_extra_points_to_hold_in) {
      remaining_extra_points_to_hold_ = remaining_extra_points_to_hold_in;
    }

    bool push_back() {
      return true;
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the number of subtables transferred.
      int num_subtables = sub_tables_.size();
      ar & num_subtables;
      if(num_subtables > 0) {
        for(int i = 0; i < num_subtables; i++) {
          ar & sub_tables_[i];
        }
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of subtables transferred.
      int num_subtables;
      ar & num_subtables;

      if(num_subtables > 0) {
        sub_tables_.resize(num_subtables);
        for(int i = 0; i < num_subtables; i++) {
          ar & sub_tables_[i];
        }
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif
