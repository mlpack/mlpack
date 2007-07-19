/**
 * @file work.h
 *
 * Work-queues, load balancing, and the like.
 */

#ifndef NBR_WORK_H
#define NBR_WORK_H

#include "rpc.h"
#include "cache.h"
#include "cachearray.h"

#include "base/cc.h"
#include "col/arraylist.h"
#include "col/heap.h"

//------------------------------------------------------------------------

/**
 * Generic work-queue interface.
 */
class WorkQueueInterface {
  FORBID_COPY(WorkQueueInterface);

 public:
  WorkQueueInterface() {}
  virtual ~WorkQueueInterface() {}

  virtual void GetWork(int process, ArrayList<index_t> *work) = 0;
};

//------------------------------------------------------------------------

class LockedWorkQueue : public WorkQueueInterface {
  FORBID_COPY(LockedWorkQueue);

 private:
  WorkQueueInterface *inner_;
  Mutex mutex_;
  
 public:
  LockedWorkQueue(WorkQueueInterface *inner) : inner_(inner) {}
  virtual ~LockedWorkQueue() { delete inner_; }

  virtual void GetWork(int process, ArrayList<index_t> *work) {
    mutex_.Lock();
    inner_->GetWork(process, work);
    mutex_.Unlock();
  }
};

//------------------------------------------------------------------------

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

  virtual void GetWork(int process, ArrayList<index_t> *work);

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
void SimpleWorkQueue<Node>::GetWork(int process, ArrayList<index_t> *work) {
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

//------------------------------------------------------------------------

template<typename Node>
class CentroidWorkQueue
    : public WorkQueueInterface {
  FORBID_COPY(CentroidWorkQueue);

 private:
  struct ProcessWorkQueue {
    index_t n_centers;
    Vector sum_centers;
    MinHeap<double, index_t> work_items;
  };

 private:
  CacheArray<Node> *tree_;
  ArrayList<ProcessWorkQueue> processes_;
  index_t max_grain_size_;

 public:
  CentroidWorkQueue() {}
  virtual ~CentroidWorkQueue() {}

  void Init(CacheArray<Node> *tree_in, index_t n_grains);

  index_t n_grains() const {
    return tasks_.size();
  }

  virtual void GetWork(int process_num, ArrayList<index_t> *work);

 private:
  void AddWork_(CacheArray<Node> *array, index_t grain_size, index_t node_i);
};

template<typename Node>
void CentroidWorkQueue<Node>::Init(CacheArray<Node> *tree) {
  tree_ = tree_in;
  processes_.Init(rpc::n_peers());
}

template<typename Node>
void CentroidWorkQueue<Node>::GetWork(int process_num, ArrayList<index_t> *work) {
  ProcessWorkQueue *queue = &processes_[process];
  Vector center;

  center.Copy(queue->sum_centers);
  centroid.Scale(1.0 / queue->n_centers);

  find work item

  Vector midpoint;
  node->bound().CalculateMidpoint(&midpoint);
  la::Add(midpoint, &queue->sum_centers);
  queue->n_centers++;
}

template<typename Node>
void CentroidWorkQueue<Node>::AddWork_(
    CacheArray<Node> *array, index_t grain_size, index_t node_i) {
}

//------------------------------------------------------------------------

struct WorkRequest {
  enum Operation { GIVE_ME_WORK } operation;
  int process;

  OT_DEF(WorkRequest) {
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(process);
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

  void GetWork(int process, ArrayList<index_t> *work_items);
};

#endif
