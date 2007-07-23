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

#include "fastlib/fastlib_int.h"

//------------------------------------------------------------------------

/**
 * Generic work-queue interface.
 *
 * Consider renaming this to "Scheduler" or "Work Scheduler" because it's
 * not exactly a queue -- it can be any kind of scheduler.
 */
class WorkQueueInterface {
  FORBID_COPY(WorkQueueInterface);

 public:
  WorkQueueInterface() {}
  virtual ~WorkQueueInterface() {}

  /**
   * Gets work items to do -- may get one or multiple items.
   *
   * Work items are labelled by an index.  For dual-tree algorithms,
   * this is probably the index in the array of tree nodes of the
   * node to operate on.
   */
  virtual void GetWork(int process, ArrayList<index_t> *work) = 0;

  /**
   * Report any relevant statistics to fastexec.
   */
  virtual void Report(struct datanode *module);
};

//------------------------------------------------------------------------

/**
 * A wrapper for a work-queue that ensures that multiple
 * threads aren't going to trample on each other.
 */
class LockedWorkQueue : public WorkQueueInterface {
  FORBID_COPY(LockedWorkQueue);

 private:
  WorkQueueInterface *inner_;
  Mutex mutex_;
  
 public:
  /**
   * Wraps another work-queue.
   *
   * Will delete the other work-queue when done.
   */
  LockedWorkQueue(WorkQueueInterface *inner) : inner_(inner) {}
  virtual ~LockedWorkQueue() { delete inner_; }

  virtual void GetWork(int process, ArrayList<index_t> *work) {
    mutex_.Lock();
    inner_->GetWork(process, work);
    mutex_.Unlock();
  }
  
  virtual void Report(struct datanode *module) {
    inner_->Report(module);
  }
};

//------------------------------------------------------------------------

/**
 * Work queue that randomly assigns tasks to processors, starting with the
 * largest query node first.  This works fine for small-scale parallelism,
 * but when multiple computers are involved, something that is
 * spatial-locality-aware like CentroidWorkQueue would be desirable.
 */
template<typename Node>
class SimpleWorkQueue
    : public WorkQueueInterface {
  FORBID_COPY(SimpleWorkQueue);

 private:
  MinHeap<double, index_t> tasks_;

 public:
  SimpleWorkQueue() {}
  virtual ~SimpleWorkQueue() {}

  /**
   * Creates a work-queue with the specified minimum
   * number of work items (k-hat).
   *
   * This does not keep a reference to array.
   */
  void Init(CacheArray<Node> *array, index_t n_grains);
  
  index_t n_grains() const {
    return tasks_.size();
  }

  virtual void GetWork(int process, ArrayList<index_t> *work);
  
  void Report(struct datanode *module);

 private:
  void AddWork_(CacheArray<Node> *array, index_t grain_size, index_t node_i);
};

template<typename Node>
void SimpleWorkQueue<Node>::Report(struct datanode *module) {
  fx_format_result(module, "n_grains", "%"LI"d", n_grains());
}

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

#include "spbounds.h"

template<typename Node>
class CentroidWorkQueue
    : public WorkQueueInterface {
  FORBID_COPY(CentroidWorkQueue);

 private:
  struct InternalNode {
   public:
    typename Node::Bound bound;
    enum { NONE, SOME, ALL } assignment;
    bool is_leaf;
    index_t index;
    index_t count;
    index_t child_indices[2];
    InternalNode *children[2];
    InternalNode *parent;
   
   public:
    ~InternalNode() {
      if (children[0]) {
        delete children[0];
      }
      if (children[1]) {
        delete children[1];
      }
    }
  };

  struct ProcessWorkQueue {
    index_t n_centers;
    Vector sum_centers;
    MinHeap<double, InternalNode*> work_items;
  };

 private:
  CacheArray<Node> *tree_;
  ArrayList<ProcessWorkQueue> processes_;
  InternalNode *root_;
  index_t max_grain_size_;
  index_t n_tasks_;
  index_t n_overflows_;
  index_t n_preferred_;
  index_t n_overflow_points_;
  index_t n_assigned_points_;

 public:
  CentroidWorkQueue() {}
  virtual ~CentroidWorkQueue() {
    delete root_;
  }

  /**
   * Creates a work-queue with the specified minimum number of
   * grains (k-hat).
   *
   * This *WILL* keep a referenece to the tree.  Do not call
   * GetWork after the tree has been destroyed.
   */
  void Init(CacheArray<Node> *tree_in, index_t n_grains);

  index_t n_grains() const {
    return n_tasks_;
  }

  virtual void GetWork(int process_num, ArrayList<index_t> *work);

  index_t n_overflows() const {
    return n_overflows_;
  }
  index_t n_preferred() const {
    return n_preferred_;
  }

  virtual void Report(struct datanode *datanode);

 private:
  void AddWork_(CacheArray<Node> *array, index_t grain_size, index_t node_i);

  /** Creates the domain decomposition */
  void DistributeInitialWork_(
      InternalNode *node, int process_begin, int process_end);
  
  bool IsSmallEnough_(const InternalNode *node) {
    return node->count <= max_grain_size_ || node->is_leaf;
  }

  InternalNode *Child_(InternalNode *node, int i) {
    if (!node->is_leaf && node->children[i] == NULL) {
      node->children[i] = MakeNode_(node->child_indices[i], node);
    }
    return node->children[i];
  }

  InternalNode *MakeNode_(index_t index, InternalNode *parent);
  
};

template<typename Node>
typename CentroidWorkQueue<Node>::InternalNode *
    CentroidWorkQueue<Node>::MakeNode_(index_t index, InternalNode *parent) {
  const Node *orig_node = tree_->StartRead(index);
  InternalNode *node = new InternalNode();
  node->bound.Copy(orig_node->bound());
  node->is_leaf = orig_node->is_leaf();
  node->index = index;
  node->count = orig_node->count();
  for (int i = 0; i < 2; i++) {
    node->child_indices[i] = node->is_leaf ? -1 : orig_node->child(i);
    node->children[i] = NULL;
  }
  node->parent = parent;
  if (parent) {
    node->assignment = parent->assignment;
  } else {
    node->assignment = InternalNode::NONE;
  }
  tree_->StopRead(index);
  return node;
}

template<typename Node>
void CentroidWorkQueue<Node>::Init(CacheArray<Node> *tree, index_t n_grains) {
  tree_ = tree;
  root_ = MakeNode_(0, NULL);
  max_grain_size_ = root_->count / n_grains;
  processes_.Init(rpc::n_peers());
  n_tasks_ = 0;
  n_overflows_ = 0;
  n_overflow_points_ = 0;
  n_assigned_points_ = 0;
  n_preferred_ = 0;

  DistributeInitialWork_(root_, 0, rpc::n_peers());

  // Ensure we don't access the tree after we start the algorithm -- if we
  // do, we risk making a call out to the network while within the network
  // thread (which will cause the program to hang).
  tree_ = NULL;
}

template<typename Node>
void CentroidWorkQueue<Node>::DistributeInitialWork_(
    InternalNode *node, int process_begin, int process_end) {
  int n_processs = process_end - process_begin;

  if (n_processs == 1 || node->is_leaf) {
    // Prime each processor's centroid with the centroid of this block.
    // Note there will probably only be 1 processor unless the pathological
    // case where we're trying to subdivide a leaf between processors.
    Vector center;
    node->bound.CalculateMidpoint(&center);

    for (int i = process_begin; i < process_end; i++) {
      ProcessWorkQueue *queue = &processes_[process_begin];
      queue->n_centers = 1;
      queue->sum_centers.Copy(center);
      queue->work_items.Init();
    }

    ProcessWorkQueue *queue = &processes_[process_begin];
    ArrayList<InternalNode*> node_stack;
    node_stack.Init();
    *node_stack.AddBack() = node;

    // In this queue, prioritize by distance from the center of the node so
    // that we start at the center and work outwards.
    while (node_stack.size() != 0) {
      InternalNode *cur = *node_stack.PopBackPtr();
      double distance = cur->bound.MinDistanceSq(center);

      if (IsSmallEnough_(cur)) {
        queue->work_items.Put(distance, cur);
        n_tasks_++;
      } else {
        *node_stack.AddBack() = Child_(cur, 0);
        *node_stack.AddBack() = Child_(cur, 1);
      }
    }
  } else {
    InternalNode *left = Child_(node, 0);
    InternalNode *right = Child_(node, 1);
    int process_boundary = process_begin +
        int(nearbyint(double(n_processs) * left->count / node->count));
    DistributeInitialWork_(left, process_begin, process_boundary);
    DistributeInitialWork_(right, process_boundary, process_end);
  }
}

template<typename Node>
void CentroidWorkQueue<Node>::GetWork(int process_num, ArrayList<index_t> *work) {
  InternalNode *found_node;
  ProcessWorkQueue *queue = &processes_[process_num];

  found_node = NULL;

  while (found_node == NULL && !queue->work_items.is_empty()) {
    InternalNode *node = queue->work_items.Pop();
    if (node->assignment == InternalNode::NONE) {
      found_node = node;
    }
  }

  if (!found_node) {
    Vector center;
    center.Copy(queue->sum_centers);
    la::Scale(1.0 / queue->n_centers, &center);

    MinHeap<double, InternalNode*> prio;

    prio.Init();
    prio.Put(0, root_);

    // Single-tree nearest-node search
    while (!prio.is_empty()) {
      InternalNode *node = prio.Pop();
      if (node->assignment != InternalNode::ALL) {
        if (node->count <= max_grain_size_
            || !node->children[0] || !node->children[1]) {
          // We can't explore a node that is missing children or whose
          // count is too large.
          if (node->assignment == InternalNode::NONE) {
            found_node = node;
            n_overflow_points_ += found_node->count;
            n_overflows_++;
            break;
          }
        } else {
          DEBUG_ASSERT(!node->is_leaf);
          for (int i = 0; i < 2; i++) {
            InternalNode *child = node->children[i];
            prio.Put(node->bound.MinDistanceSq(center), child);
          }
        }
      }
    }
  } else {
    n_preferred_++;
  }

  if (found_node == NULL) {
    work->Init();
  } else {
    ArrayList<InternalNode*> stack;
    stack.Init();
    *stack.AddBack() = found_node;

    // Show user-friendly status messages every 5% increment
    n_assigned_points_ += found_node->count;
    int interval = 20;
    if ((n_assigned_points_ - found_node->count) * interval / root_->count
        != n_assigned_points_ * interval / root_->count) {
      fprintf(stderr,
          "--------------- %02d%% of work has been scheduled --------------\n",
          n_assigned_points_ * 100 / root_->count);
    }

    // Mark all children as complete
    while (stack.size() != 0) {
      InternalNode *c = *stack.PopBackPtr();
      c->assignment = InternalNode::ALL;
      if (c->children[0]) {
        *stack.AddBack() = c->children[0];
        *stack.AddBack() = c->children[1];
      }
    }

    // Mark my parents as partially or fully complete
    for (InternalNode *p = found_node->parent; p != NULL; p = p->parent) {
      DEBUG_ASSERT(!p->is_leaf);
      if (p->children[0]->assignment == InternalNode::ALL
          && p->children[1]->assignment == InternalNode::ALL) {
        p->assignment = InternalNode::ALL;
      } else {
        p->assignment = InternalNode::SOME;
      }
    }

    work->Init(1);
    (*work)[0] = found_node->index;

    Vector midpoint;
    found_node->bound.CalculateMidpoint(&midpoint);
    la::AddTo(midpoint, &queue->sum_centers);
    queue->n_centers++;
  }
}

template<typename Node>
void CentroidWorkQueue<Node>::Report(struct datanode *module) {
  fx_format_result(module, "n_grains", "%"LI"d",
      n_preferred_ + n_overflows_);
  fx_format_result(module, "overflow_grain_ratio", "%.4f",
      1.0 * n_overflows_ / (n_preferred_ + n_overflows_));
  fx_format_result(module, "n_overflows", "%"LI"d",
      n_overflows_);
  fx_format_result(module, "overflow_point_ratio", "%.4f",
      1.0 * n_overflow_points_ / root_->count);
}

//------------------------------------------------------------------------

struct WorkRequest {
  enum Operation { GIVE_ME_WORK } operation;
  int process;

  bool requires_response() const { return true; }

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
