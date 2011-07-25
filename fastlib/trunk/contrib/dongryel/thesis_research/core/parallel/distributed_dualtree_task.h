/** @file distributed_dualtree_task.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H

namespace core {
namespace parallel {

template<typename TableType, typename TreeType>
class DistributedDualtreeTask {
  private:

    TreeType *query_start_node_;

    TableType *reference_table_;

    TreeType *reference_start_node_;

    int cache_id_;

    double priority_;

  public:

    void Init(
      TreeType *query_start_node_in,
      TableType *reference_table_in, TreeType *reference_start_node_in,
      int cache_id_in, double priority_in) {

      query_start_node_ = query_start_node_in;
      reference_table_ = reference_table_in;
      reference_start_node_ = reference_start_node_in;
      cache_id_ = cache_id_in;
      priority_ = priority_in;
    }

    void operator=(const DistributedDualtreeTask &task_in) {
      this->Init(
        const_cast<DistributedDualtreeTask &>(task_in).query_start_node(),
        const_cast<DistributedDualtreeTask &>(task_in).reference_table(),
        const_cast<DistributedDualtreeTask &>(task_in).reference_start_node(),
        task_in.cache_id(), task_in.priority());
    }

    DistributedDualtreeTask(const DistributedDualtreeTask &task_in) {
      this->operator=(task_in);
    }

    DistributedDualtreeTask(
      TreeType *query_start_node_in,
      TableType *reference_table_in, TreeType *reference_start_node_in,
      int cache_id_in, double priority_in) {
      this->Init(
        query_start_node_in, reference_table_in, reference_start_node_in,
        cache_id_in, priority_in);
    }

    DistributedDualtreeTask() {
      query_start_node_ = NULL;
      reference_table_ = NULL;
      reference_start_node_ = NULL;
      cache_id_ = 0;
      priority_ = 0.0;
    }

    TreeType *query_start_node() {
      return query_start_node_;
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
