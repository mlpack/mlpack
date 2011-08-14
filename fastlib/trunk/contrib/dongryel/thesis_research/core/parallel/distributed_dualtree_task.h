/** @file distributed_dualtree_task.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H

namespace core {
namespace parallel {

template<typename TableType, typename TreeType, typename QueryResultType>
class DistributedDualtreeTask {
  private:

    TableType *query_table_;

    TreeType *query_start_node_;

    QueryResultType *query_result_;

    TableType *reference_table_;

    TreeType *reference_start_node_;

    int cache_id_;

    double priority_;

  public:

    void set_query_start_node(TreeType *query_start_node_in) {
      query_start_node_ = query_start_node_in;
    }

    void Init(
      TableType *query_table_in,
      TreeType *query_start_node_in,
      QueryResultType *query_result_in,
      TableType *reference_table_in,
      TreeType *reference_start_node_in,
      int cache_id_in, double priority_in) {

      query_table_ = query_table_in;
      query_start_node_ = query_start_node_in;
      query_result_ = query_result_in;
      reference_table_ = reference_table_in;
      reference_start_node_ = reference_start_node_in;
      cache_id_ = cache_id_in;
      priority_ = priority_in;
    }

    void operator=(const DistributedDualtreeTask &task_in) {
      DistributedDualtreeTask *task_modifiable =
        const_cast<DistributedDualtreeTask *>(&task_in);
      this->Init(
        task_modifiable->query_table(),
        task_modifiable->query_start_node(),
        task_modifiable->query_result(),
        task_modifiable->reference_table(),
        task_modifiable->reference_start_node(),
        task_in.cache_id(), task_in.priority());
    }

    DistributedDualtreeTask(const DistributedDualtreeTask &task_in) {
      this->operator=(task_in);
    }

    DistributedDualtreeTask(
      TableType *query_table_in,
      TreeType *query_start_node_in,
      QueryResultType *query_result_in,
      TableType *reference_table_in,
      TreeType *reference_start_node_in,
      int cache_id_in, double priority_in) {
      this->Init(
        query_table_in, query_start_node_in, query_result_in,
        reference_table_in, reference_start_node_in,
        cache_id_in, priority_in);
    }

    DistributedDualtreeTask() {
      query_table_ = NULL;
      query_start_node_ = NULL;
      query_result_ = NULL;
      reference_table_ = NULL;
      reference_start_node_ = NULL;
      cache_id_ = 0;
      priority_ = 0.0;
    }

    TableType *query_table() {
      return query_table_;
    }

    TreeType *query_start_node() {
      return query_start_node_;
    }

    QueryResultType *query_result() {
      return query_result_;
    }

    TableType *reference_table() {
      return reference_table_;
    }

    TreeType *reference_start_node() {
      return reference_start_node_;
    }

    int cache_id() const {
      return cache_id_;
    }

    double priority() const {
      return priority_;
    }
};
}
}

#endif
