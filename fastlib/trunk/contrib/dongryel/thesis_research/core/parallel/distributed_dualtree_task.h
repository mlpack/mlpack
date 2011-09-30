/** @file distributed_dualtree_task.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H

#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template<typename IncomingTableType>
class DistributedDualtreeTask {

  public:

    typedef IncomingTableType TableType;

    typedef typename TableType::QueryResultType QueryResultType;

    typedef core::table::SubTable<TableType> SubTableType;

    typedef typename TableType::TreeType TreeType;

  private:

    SubTableType query_subtable_;

    SubTableType reference_subtable_;

    double priority_;

  public:

    unsigned long int work() const {
      return (this->query_start_node()->count()) *
             (this->reference_start_node()->count());
    }

    TreeType *query_start_node() const {
      return query_subtable_.start_node();
    }

    TreeType *reference_start_node() const {
      return reference_subtable_.start_node();
    }

    void set_query_start_node(TreeType *query_start_node_in) {
      query_subtable_.set_start_node(query_start_node_in);
    }

    void Init(
      SubTableType &query_subtable_in,
      SubTableType &reference_subtable_in,
      double priority_in) {

      // Alias the incoming query subtable.
      query_subtable_.Init(
        query_subtable_in.table(), query_subtable_in.start_node(), false);
      query_subtable_.set_query_result(* query_subtable_in.query_result());
      query_subtable_.set_cache_block_id(
        query_subtable_in.cache_block_id());
      query_subtable_.set_id_to_position_map(
        query_subtable_in.id_to_position_map());
      query_subtable_.set_position_to_id_map(
        query_subtable_in.position_to_id_map());

      // Alias the incoming reference subtable.
      reference_subtable_.Init(
        reference_subtable_in.table(),
        reference_subtable_in.start_node(), false);
      reference_subtable_.set_cache_block_id(
        reference_subtable_in.cache_block_id());
      reference_subtable_.set_id_to_position_map(
        reference_subtable_in.id_to_position_map());
      reference_subtable_.set_position_to_id_map(
        reference_subtable_in.position_to_id_map());
      priority_ = priority_in;
    }

    void operator=(const DistributedDualtreeTask &task_in) {
      DistributedDualtreeTask *task_modifiable =
        const_cast<DistributedDualtreeTask *>(&task_in);
      this->Init(
        task_modifiable->query_subtable(),
        task_modifiable->reference_subtable(),
        task_in.priority());
    }

    DistributedDualtreeTask(const DistributedDualtreeTask &task_in) {
      this->operator=(task_in);
    }

    DistributedDualtreeTask(
      SubTableType &query_subtable_in,
      SubTableType &reference_subtable_in,
      double priority_in) {
      this->Init(
        query_subtable_in, reference_subtable_in, priority_in);
    }

    DistributedDualtreeTask() {
      priority_ = 0.0;
    }

    SubTableType &query_subtable() {
      return query_subtable_;
    }

    QueryResultType *query_result() {
      return query_subtable_.query_result();
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
