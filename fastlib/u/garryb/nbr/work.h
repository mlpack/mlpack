/**
 * @file work.h
 *
 * Work-queues, load balancing, and the like.
 */

#ifndef NBR_WORK_H
#define NBR_WORK_H

#include "rpc.h"

/**
 * Generic work-queue interface.
 */
class WorkQueueInterface {
  FORBID_COPY(WorkQueueInterface);

 public:
  WorkQueueInterface() {}
  virtual ~WorkQueueInterface() {}

  virtual void GetWork(ArrayList<index_t> *work) = 0;
};

template<typename Node>
class SimpleWorkQueue
    : public WorkQueueInterface {
 private:
  MinHeap<double, index_t> tasks_;

 public:
  void Init(CacheArray<Node> *array, index_t n_grains) {
    index_t root_index = 0;
    const Node *root = array->StartRead(root_index);
    tasks_.Init();
    AddWork_(array, root_index, (root->count() + n_grains - 1) / n_grains);
    array->StopRead(root_index);
  }

  virtual void GetWork(ArrayList<index_t> *work) {
    if (tasks_.size() == 0) {
      work->Init(0);
    } else {
      work->Init(1);
      work->set(0, tasks_.Pop());
    }
  }

 private:
  void AddWork_(CacheArray<Node> *array, index_t grain_size, index_t node_i) {
    const Node *node = array->StartRead(node_i);

    if (node->count() <= grain_size || node->is_leaf()) {
      tasks_.Add(-1.0 * node->count(), node_i);
    } else {
      for (index_t k = 0; k < 2; k++) {
        SplitNode(narray, grain_size, node->child(k));
      }
    }

    array->StopRead(node_i);
  }
};

// Making work queues work over the net.

struct WorkRequest {
  enum { GIVE_ME_WORK } operation;
}; 

struct WorkResponse {
  ArrayList<index_t> work_items;
};
class RemoteWorkQueueBackend
    : public RemoteObjectBackend<WorkRequest, WorkResponse> {
 private:
  WorkQueueInterface *inner_;

 public:
  void Init(WorkQueueInterface *inner_work_queue) {
    inner_ = inner_work_queue;
  }

  virtual void HandleRequest(
      const WorkRequest& request, WorkResponse *response) {
    DEBUG_ASSERT(request.operation == WorkRequest::GIVE_ME_WORK);
    inner_->GetWork(&response->work_items);
  }
};

class RemoteWorkQueue
    : public WorkQueueInterface {
 private:
  RemoteObjectStub<WorkRequest, WorkResponse> stub_;

 public:
  void Init(int channel, int destination) {
    stub_.Init(channel, destination);
  }

  void GetWork(ArrayList<index_t> *work_items) {
    WorkRequest request;
    request.operation = WorkRequest::GIVE_ME_WORK;

    stub_.Lock();
    WorkResponse *response = stub_.Request(request);
    work_items->Copy(response->work_items);
    stub_.Unlock();
  }
};

#endif
