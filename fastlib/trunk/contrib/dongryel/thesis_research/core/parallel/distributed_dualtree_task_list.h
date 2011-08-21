/** @file distributed_dualtree_task_list.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_LIST_H

#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include "core/parallel/distributed_dualtree_task.h"
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template<typename TaskType>
class DistributedDualtreeTaskList {

  public:

    typedef typename TaskType::TableType TableType;

    typedef core::table::SubTable<TableType> SubTableType;

  private:

    boost::scoped_array<SubTableType> sub_tables_;

    int num_subtables_;

  public:

    DistributedDualtreeTaskList() {
      num_subtables_ = 0;
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of subtables transferred.

    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

};
}
}

#endif
