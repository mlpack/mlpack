/** @file distributed_dualtree_task.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_TASK_H

#include <deque>
#include <vector>

namespace core {
namespace parallel {

template<typename TreeType, typename TaskPriorityQueueType>
class DistributedDualtreeTask {
  private:

    std::vector< TreeType *> local_query_trees_;

    std::deque<bool> local_query_trees_lock_;

    std::vector< TaskPriorityQueueType > tasks_;

    std::deque< bool > do_split_after_dequeue_;

  private:

    void split_subtree_(int local_query_tree_id) {

      // After splitting, the current index will have the left child
      // and the right child will be appended to the end of the list
      // of trees, plus duplicating the reference tasks along the way.
      if(! local_query_trees_[ local_query_tree_id]->is_leaf()) {
      }
    }

  public:

    void Init() {
    }
};
}
}

#endif
