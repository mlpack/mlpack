/** @file distributed_dualtree_task.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H

#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template<typename IncomingTableType, typename QueryResultType>
class DistributedDualtreeTask {

  public:

    typedef IncomingTableType TableType;

    typedef core::table::SubTable<TableType> SubTableType;

    typedef typename TableType::TreeType TreeType;

  private:

    SubTableType query_subtable_;

    QueryResultType *query_result_;

    SubTableType reference_subtable_;

    int cache_id_;

    double priority_;

  public:

    TreeType *query_start_node() {
      return query_subtable_.start_node();
    }

    TreeType *reference_start_node() {
      return reference_subtable_.start_node();
    }

    void set_query_start_node(TreeType *query_start_node_in) {
      query_subtable_.set_start_node(query_start_node_in);
    }

    void Init(
      SubTableType &query_subtable_in,
      QueryResultType *query_result_in,
      SubTableType &reference_subtable_in,
      double priority_in) {

      query_subtable_.Alias(query_subtable_in);
      query_result_ = query_result_in;
      reference_subtable_.Alias(reference_subtable_in);
      priority_ = priority_in;
    }

    void operator=(const DistributedDualtreeTask &task_in) {
      DistributedDualtreeTask *task_modifiable =
        const_cast<DistributedDualtreeTask *>(&task_in);
      this->Init(
        task_modifiable->query_subtable(),
        task_modifiable->query_result(),
        task_modifiable->reference_subtable(),
        task_in.priority());
    }

    DistributedDualtreeTask(const DistributedDualtreeTask &task_in) {
      this->operator=(task_in);
    }

    DistributedDualtreeTask(
      SubTableType &query_subtable_in,
      QueryResultType *query_result_in,
      SubTableType &reference_subtable_in,
      double priority_in) {
      this->Init(
        query_subtable_in, query_result_in, reference_subtable_in, priority_in);
    }

    DistributedDualtreeTask() {
      query_result_ = NULL;
      priority_ = 0.0;
    }

    SubTableType &query_subtable() {
      return query_subtable_;
    }

    QueryResultType *query_result() {
      return query_result_;
    }

    SubTableType &reference_subtable() {
      return reference_subtable_;
    }

    int reference_subtable_cache_block_id() const {
      return reference_subtable_.cache_block_id();
    }

    double priority() const {
      return priority_;
    }
};
}
}

#endif
