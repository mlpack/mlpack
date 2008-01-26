/**
 * @file sched.h
 *
 * Work-queues, load balancing, and the like.
 */

#ifndef THOR_WORK_H
#define THOR_WORK_H

#include "rpc.h"
#include "cache.h"
#include "cachearray.h"
#include "thortree.h"

#include "fastlib/col/heap.h"
#include "fastlib/col/arraylist.h"
#include "fastlib/la/uselapack.h"
#include "fastlib/tree/bounds.h"

//------------------------------------------------------------------------

/**
 * Generic work-queue interface.
 *
 * Consider renaming this to "Scheduler" or "Work Scheduler" because it's
 * not exactly a queue -- it can be any kind of scheduler.
 */
class SchedulerInterface {
  FORBID_ACCIDENTAL_COPIES(SchedulerInterface);

 public:
  typedef TreeGrain Grain;

 public:
  SchedulerInterface() {}
  virtual ~SchedulerInterface() {}

  /**
   * Gets work items to do -- may get one or multiple items.
   *
   * Work items are labelled by an index.  For dual-tree algorithms,
   * this is probably the index in the array of tree nodes of the
   * node to operate on.
   *
   * @param rank the rank of the machine requesting work
   * @param work where the work will be stored
   */
  virtual void GetWork(int rank, ArrayList<Grain> *work) = 0;

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
class LockedScheduler : public SchedulerInterface {
  FORBID_ACCIDENTAL_COPIES(LockedScheduler);

 private:
  SchedulerInterface *inner_;
  Mutex mutex_;
  
 public:
  /**
   * Wraps another work-queue.
   *
   * Will delete the other work-queue when done.
   */
  LockedScheduler(SchedulerInterface *inner) : inner_(inner) {}
  virtual ~LockedScheduler() { delete inner_; }

  virtual void GetWork(int rank, ArrayList<Grain> *work) {
    mutex_.Lock();
    inner_->GetWork(rank, work);
    mutex_.Unlock();
  }

  virtual void Report(struct datanode *module) {
    inner_->Report(module);
  }
};

//------------------------------------------------------------------------

template<typename Node>
class CentroidScheduler
    : public SchedulerInterface {
  FORBID_ACCIDENTAL_COPIES(CentroidScheduler);

 private:
  /** Convenience typedef for the decomposition node. */
  typedef typename ThorTreeDecomposition<Node>::DecompNode DecompNode;

  /**
   * Completion status of part of the tree.
   */
  enum Status {
    NONE, //< None of this branch has been scheduled
    SOME, //< At least one part of this branch has been scheduled, but not all
    ALL //< This branch has been completely scheduled
  };

  /** Skeleton node to keep trach of scheduling. */
  typedef ThorSkeletonNode<Node, Status> InternalNode;

  /** The status for one particular machine. */
  struct ProcessScheduler {
    index_t n_centers;
    index_t max_grain_size;
    Vector sum_centers;
    MinHeap<double, InternalNode*> work_items;
    
    OT_DEF(ProcessScheduler) {
      OT_MY_OBJECT(n_centers);
      OT_MY_OBJECT(max_grain_size);
      OT_MY_OBJECT(sum_centers);
      OT_MY_OBJECT(work_items);
    }
  };

 private:
  CacheArray<Node> *tree_;
  ArrayList<ProcessScheduler> rankes_;
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
  CentroidScheduler() {}
  virtual ~CentroidScheduler() {
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

  /** Gets the number of work grains. */
  index_t n_grains() const {
    return n_grains_;
  }

  virtual void GetWork(int rank_num, ArrayList<Grain> *work);

  /**
   * Gets the number of grains that were not assigned to the original machine.
   */
  index_t n_overflows() const {
    return n_overflows_;
  }
  /**
   * Gets the number of grains that were assigned to the originally scheduled
   * or preferred machine.
   */
  index_t n_preferred() const {
    return n_preferred_;
  }

  virtual void Report(struct datanode *datanode);

 private:
  /** Adds a work item. */
  void AddWork_(CacheArray<Node> *array, index_t grain_size, index_t node_i);

  /** Creates the domain decomposition */
  void DistributeInitialWork_(const DecompNode *decomp_node,
      InternalNode *node);
};

//------------------------------------------------------------------------

struct WorkRequest {
  enum Operation { GIVE_ME_WORK } operation;
  int rank;

  bool requires_response() const { return true; }

  OT_DEF_BASIC(WorkRequest) {
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(rank);
  }
}; 

struct WorkResponse {
  ArrayList<SchedulerInterface::Grain> work_items;

  OT_DEF_BASIC(WorkResponse) {
    OT_MY_OBJECT(work_items);
  }
};

class RemoteSchedulerBackend
    : public RemoteObjectBackend<WorkRequest, WorkResponse> {
 private:
  SchedulerInterface *inner_;

 public:
  void Init(SchedulerInterface *inner_work_queue);

  virtual void HandleRequest(
      const WorkRequest& request, WorkResponse *response);
};

class RemoteScheduler
    : public SchedulerInterface {
  FORBID_ACCIDENTAL_COPIES(RemoteScheduler);

 private:
  int channel_;
  int destination_;

 public:
  RemoteScheduler() {}
  virtual ~RemoteScheduler() {}
  
  void Init(int channel, int destination);

  void GetWork(int rank, ArrayList<Grain> *work_items);
};

//------------------------------------------------------------------------

#include "sched_impl.h"

#endif
