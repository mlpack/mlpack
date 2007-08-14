/**
 * @file work.h
 *
 * Work-queues, load balancing, and the like.
 */

#ifndef THOR_WORK_H
#define THOR_WORK_H

#include "rpc.h"
#include "cache.h"
#include "cachearray.h"
#include "thortree.h"

#include "col/heap.h"
#include "col/arraylist.h"
#include "la/uselapack.h"
#include "tree/bounds.h"

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
  typedef TreeGrain Grain;

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
  virtual void GetWork(int process, ArrayList<Grain> *work) = 0;

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

  virtual void GetWork(int process, ArrayList<Grain> *work) {
    mutex_.Lock();
    inner_->GetWork(process, work);
    mutex_.Unlock();
  }
  
  virtual void Report(struct datanode *module) {
    inner_->Report(module);
  }
};

//------------------------------------------------------------------------

template<typename Node>
class CentroidWorkQueue
    : public WorkQueueInterface {
  FORBID_COPY(CentroidWorkQueue);

 private:
  typedef typename ThorTreeDecomposition<Node>::DecompNode DecompNode;
  enum Status { NONE, SOME, ALL };
  typedef ThorSkeletonNode<Node, Status> InternalNode;

  struct ProcessWorkQueue {
    index_t n_centers;
    index_t max_grain_size;
    Vector sum_centers;
    MinHeap<double, InternalNode*> work_items;
    
    OT_DEF(ProcessWorkQueue) {
      OT_MY_OBJECT(n_centers);
      OT_MY_OBJECT(max_grain_size);
      OT_MY_OBJECT(sum_centers);
      OT_MY_OBJECT(work_items);
    }
  };

 private:
  CacheArray<Node> *tree_;
  ArrayList<ProcessWorkQueue> processes_;
  InternalNode *root_;
  int n_threads_;
  double granularity_;
  index_t n_grains_;
  index_t n_overflows_;
  index_t n_preferred_;
  index_t n_overflow_points_;
  index_t n_assigned_points_;
  bool no_overflow_;

 public:
  CentroidWorkQueue() {}
  virtual ~CentroidWorkQueue() {
    delete root_;
  }

  /**
   * Creates a work-queue with the specified minimum number of
   * grains (k-hat).
   *
   * This will NOT keep a permanent reference to the tree.
   */
  void Init(CacheArray<Node> *tree_in, const DecompNode *decomp_root,
      int n_threads, datanode *module);

  index_t n_grains() const {
    return n_grains_;
  }

  virtual void GetWork(int process_num, ArrayList<Grain> *work);

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
  void DistributeInitialWork_(const DecompNode *decomp_node,
      InternalNode *node);
};

template<typename Node>
void CentroidWorkQueue<Node>::Init(CacheArray<Node> *tree,
    const DecompNode *decomp_root, int n_threads, datanode *module) {
  granularity_ = fx_param_double(module, "granularity", 12);
  no_overflow_ = fx_param_bool(module, "no_overflow", false);
  tree_ = tree;
  n_threads_ = n_threads;
  root_ = new InternalNode(NONE, tree,
      decomp_root->index(), decomp_root->end_index());
  processes_.Init(rpc::n_peers());
  DEBUG_ASSERT_MSG(decomp_root->info().begin_rank == 0
      && decomp_root->info().end_rank == rpc::n_peers(),
      "Can't handle incomplete decompositions (yet)");
  n_grains_ = 0;
  n_overflows_ = 0;
  n_overflow_points_ = 0;
  n_assigned_points_ = 0;
  n_preferred_ = 0;

  DistributeInitialWork_(decomp_root, root_);

  // Ensure we don't access the tree after we start the algorithm.
  tree_ = NULL;
}

template<typename Node>
void CentroidWorkQueue<Node>::DistributeInitialWork_(
    const DecompNode *decomp_node, InternalNode *node) {
  int begin_rank = decomp_node->info().begin_rank;
  int end_rank = decomp_node->info().end_rank;
  int n_processs = end_rank - begin_rank;

  if (n_processs == 1 || node->is_leaf()) {
    // Prime each processor's centroid with the centroid of this block.
    // Note there will probably only be 1 processor unless the pathological
    // case where we're trying to subdivide a leaf between processors.
    // If we're subdividing a leaf between processors, then some processors
    // won't have any work to do...
    Vector center;
    node->node().bound().CalculateMidpoint(&center);
    index_t max_grain_size = math::RoundInt(
        node->count() / granularity_ / n_threads_);

    for (int i = begin_rank; i < end_rank; i++) {
      ProcessWorkQueue *queue = &processes_[i];
      queue->max_grain_size = max_grain_size;
      queue->n_centers = 1;
      queue->sum_centers.Copy(center);
      queue->work_items.Init();
    }

    ProcessWorkQueue *queue = &processes_[begin_rank];
    ArrayList<InternalNode*> node_stack;
    node_stack.Init();
    *node_stack.AddBack() = node;

    // Subdivide the node further if possible.
    while (node_stack.size() != 0) {
      InternalNode *cur = *node_stack.PopBackPtr();
      double distance = cur->node().bound().MinDistanceSq(center);

      if (cur->is_leaf() || cur->count() <= max_grain_size) {
        queue->work_items.Put(distance, cur);
        n_grains_++;
      } else {
        for (index_t k = 0; k < Node::CARDINALITY; k++) {
          *node_stack.AddBack() = cur->GetChild(tree_, k);
        }
      }
    }
  } else {
    for (index_t k = 0; k < Node::CARDINALITY; k++) {
      DistributeInitialWork_(decomp_node->child(k), node->GetChild(tree_, k));
    }
  }
}

template<typename Node>
void CentroidWorkQueue<Node>::GetWork(int process_num, ArrayList<Grain> *work) {
  InternalNode *found_node;
  ProcessWorkQueue *queue = &processes_[process_num];

  found_node = NULL;

  while (found_node == NULL && !queue->work_items.is_empty()) {
    InternalNode *node = queue->work_items.Pop();
    if (node->info() == NONE) {
      found_node = node;
    }
  }

  if (!found_node) {
    if (!no_overflow_) {
      Vector center;
      center.Copy(queue->sum_centers);
      la::Scale(1.0 / queue->n_centers, &center);

      MinHeap<double, InternalNode*> prio;

      prio.Init();
      prio.Put(0, root_);

      // Single-tree nearest-node search
      while (!prio.is_empty()) {
        InternalNode *node = prio.Pop();
        if (node->info() != ALL) {
          if (!node->is_complete()) {
            // We can't explore a node that is missing children or whose
            // count is too large.
            if (node->info() == NONE) {
              found_node = node;
              n_overflow_points_ += found_node->count();
              n_overflows_++;
              break;
            }
          } else {
            DEBUG_ASSERT(node->is_complete());
            for (int i = 0; i < Node::CARDINALITY; i++) {
              InternalNode *child = node->child(i);
              prio.Put(child->node().bound().MinDistanceSq(center), child);
            }
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
    // Show user-friendly status messages every 5% increment
    index_t count = found_node->count();
    n_assigned_points_ += count;

    percent_indicator("scheduled", n_assigned_points_, root_->count());

    // Mark all children as complete (non-recursive version)
    ArrayList<InternalNode*> stack;
    stack.Init();
    *stack.AddBack() = found_node;
    while (stack.size() != 0) {
      InternalNode *c = *stack.PopBackPtr();
      c->info() = ALL;
      for (index_t k = 0; k < Node::CARDINALITY; k++) {
        InternalNode *c_child = c->child(k);
        if (c_child) {
          *stack.AddBack() = c_child;
        }
      }
    }

    // Mark my parents as partially or fully complete
    for (InternalNode *p = found_node->parent(); p != NULL; p = p->parent()) {
      DEBUG_ASSERT(p->is_complete());
      p->info() = ALL;
      for (int k = 0; k < Node::CARDINALITY; k++) {
        if (p->child(k)->info() != ALL) {
          p->info() = SOME;
          break;
        }
      }
    }

    work->Init(1);
    Grain *grain = &(*work)[0];
    grain->node_index = found_node->index();
    grain->node_end_index = found_node->end_index();
    grain->point_begin_index = found_node->node().begin();
    grain->point_end_index = found_node->node().end();

    Vector midpoint;
    found_node->node().bound().CalculateMidpoint(&midpoint);
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
      1.0 * n_overflow_points_ / root_->count());
}

//------------------------------------------------------------------------

struct WorkRequest {
  enum Operation { GIVE_ME_WORK } operation;
  int process;

  bool requires_response() const { return true; }

  OT_DEF_BASIC(WorkRequest) {
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(process);
  }
}; 

struct WorkResponse {
  ArrayList<WorkQueueInterface::Grain> work_items;

  OT_DEF_BASIC(WorkResponse) {
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

  void GetWork(int process, ArrayList<Grain> *work_items);
};

#endif
