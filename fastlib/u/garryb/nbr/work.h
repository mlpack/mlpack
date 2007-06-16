/**
 * @file work.h
 *
 * Work-queues, load balancing, and the like.
 */

#ifndef NBR_WORK_H
#define NBR_WORK_H

#include "cache.h"
#include "cachearray.h"

#include "base/cc.h"
#include "col/arraylist.h"
#include "col/heap.h"

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

class LockedWorkQueue {
  FORBID_COPY(LockedWorkQueue);

 private:
  WorkQueueInterface *inner_;
  
 public:
  LockedWorkQueue(WorkQueueInterface *inner) : inner_(inner) {}
  virtual ~LockedWorkQueue() { delete inner_; }

  virtual void GetWork(ArrayList<index_t> *work);
};

template<typename Node>
class SimpleWorkQueue
    : public WorkQueueInterface {
  FORBID_COPY(SimpleWorkQueue);

 private:
  MinHeap<double, index_t> tasks_;

 public:
  SimpleWorkQueue() {}
  virtual ~SimpleWorkQueue() {}

  void Init(CacheArray<Node> *array, index_t n_grains);
  
  index_t n_grains() const {
    return tasks_.size();
  }

  virtual void GetWork(ArrayList<index_t> *work);

 private:
  void AddWork_(CacheArray<Node> *array, index_t grain_size, index_t node_i);
};

template<typename Node>
void SimpleWorkQueue<Node>::Init(CacheArray<Node> *array, index_t n_grains) {
  index_t root_index = 0;
  const Node *root = array->StartRead(root_index);
  tasks_.Init();
  AddWork_(array, (root->count() + n_grains - 1) / n_grains, root_index);
  array->StopRead(root_index);
}

template<typename Node>
void SimpleWorkQueue<Node>::GetWork(ArrayList<index_t> *work) {
  if (tasks_.size() == 0) {
    work->Init(0);
  } else {
    work->Init(1);
    (*work)[0] = tasks_.Pop();
  }
}

template<typename Node>
void SimpleWorkQueue<Node>::AddWork_(
    CacheArray<Node> *array, index_t grain_size, index_t node_i) {
  const Node *node = array->StartRead(node_i);

  if (node->count() <= grain_size || node->is_leaf()) {
    tasks_.Put(-1.0 * node->count(), node_i);
  } else {
    for (index_t k = 0; k < 2; k++) {
      AddWork_(array, grain_size, node->child(k));
    }
  }

  array->StopRead(node_i);
}

// Making work queues work over the net.
#ifdef USE_MPI
#include "rpc.h"

struct WorkRequest {
  enum Operation { GIVE_ME_WORK } operation;

  OT_DEF(WorkRequest) {
    OT_MY_OBJECT(operation);
  }
}; 

struct WorkResponse {
  ArrayList<index_t> work_items;

  OT_DEF(WorkResponse) {
    OT_MY_OBJECT(work_items);
  }
};

class RemoteWorkQueueBackend
    : public RemoteObjectBackend<WorkRequest, WorkResponse> {
 private:
  WorkQueueInterface *inner_;

 public:
  void Init(WorkQueueInterface *inner_work_queue);

  virtual void HandleRequest(
      const WorkRequest& request, WorkResponse *response);
};

class RemoteWorkQueue
    : public WorkQueueInterface {
  FORBID_COPY(RemoteWorkQueue);

 private:
  int channel_;
  int destination_;

 public:
  RemoteWorkQueue() {}
  virtual ~RemoteWorkQueue() {}
  
  void Init(int channel, int destination);

  void GetWork(ArrayList<index_t> *work_items);
};
#endif

#endif
