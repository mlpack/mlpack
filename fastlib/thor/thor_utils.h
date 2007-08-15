/**
 * @file thor_utils.h
 *
 * Top-level THOR utilities that take care of more high-level actions.
 *
 * This contains functions to read data and to execute a generalized N-body
 * problem in parallel.
 */

#ifndef THOR_UTILS_H
#define THOR_UTILS_H

#include "kdtree.h"
#include "work.h"
#include "rpc.h"

#include "par/thread.h"
#include "par/task.h"

#include "distribcache.h"

namespace thor {

/**
 * Class used to run a multi-threaded dual-tree algorithm.
 *
 * It is probably easier to use thor::RpcDualTree even if you are on just
 * one machine, and it's no more or less efficient either way.
 *
 * Template paramters:
 * @li GNP - the THOR-compatible generalized N-body class
 * @li Solver - a solver such as DualTreeDepthFirst&lt;GNP&gt;
 */
template<typename GNP, typename Solver>
class ThreadedDualTreeSolver {
 private:
  struct WorkerTask : public Task {
    ThreadedDualTreeSolver *solver;
    WorkerTask(ThreadedDualTreeSolver *solver_in) : solver(solver_in) { }
    void Run() { solver->ThreadBody_(); delete this; }
  };

 private:
  int rank_;
  const typename GNP::Param *param_;
  WorkQueueInterface *work_queue_;
  DistributedCache *q_points_cache_;
  DistributedCache *q_nodes_cache_;
  DistributedCache *r_points_cache_;
  DistributedCache *r_nodes_cache_;
  DistributedCache *q_results_cache_;
  typename GNP::GlobalResult global_result_;
  Mutex mutex_;

 public:
  /**
   * Runs a multi-threaded dual-tree problem.
   *
   * @param n_threads the number of threads to run on this machine
   * @param work_queue_in the work queue to get work items from
   * @param param the parameter object for the GNP
   * @param q_points_cache_in the cache containing query points
   * @param q_nodes_cache_in the cache containing the query tree
   * @param r_points_cache_in the cache containing reference points
   * @param r_nodes_cache_in the cache containing the reference tree
   * @param q_results_cache_in the cache containing per-query results.
   */
  void Doit(index_t n_threads, int rank, WorkQueueInterface *work_queue_in,
      const typename GNP::Param& param,
      DistributedCache *q_points_cache_in, DistributedCache *q_nodes_cache_in,
      DistributedCache *r_points_cache_in, DistributedCache *r_nodes_cache_in,
      DistributedCache *q_results_cache_in);

  /**
   * Gets the GNP's global-result of the entire computation.
   */
  const typename GNP::GlobalResult& global_result() const {
    return global_result_;
  }

 private:
  void ThreadBody_();
};

/**
 * An rpc-style reductor suitable for condensing global results.
 *
 * Template parameter GNP is the generalized N-body problem containing
 * nested classes Param and GlobalResult.
 */
template<typename GNP>
class GlobalResultReductor {
 private:
  const typename GNP::Param *param_;

 public:
  /** Initializes the reductor for a given parameter object. */
  void Init(const typename GNP::Param* param_in) {
    param_ = param_in;
  }

  /**
   * Reduces two elements.
   *
   * @param right a new element to merge
   * @param left the element to merge into
   */
  void Reduce(typename GNP::GlobalResult& right,
      typename GNP::GlobalResult* left) const {
    left->Accumulate(*param_, right);
  }
};

/**
 * Reads points in a data set, just for the master machine.
 *
 * This does all the actual reading into the cache but doesn't do any
 * syncing.  It's recommended to just use thor::ReadPoints.
 *
 * Template parameters:
 * @li @c Point - a conformant point type (see gnp.h)
 * @li @c Param - a parameter object used for initializing points
 */
template<typename Point, typename Param>
index_t ReadPointsMaster(
    const Param& param, int points_channel,
    const char *filename, int block_size_kb, double megs,
    DistributedCache *points_cache);

/**
 * Reads data points from a file into a data set.
 *
 * The Point object must contain the suitable Init and Set method as does
 * the class ThorVectorPoint in gnp.h.
 *
 * This takes in a module, such that the module's root parameter is the
 * filename.  There is optionally another parameter "block_size_kb" which
 * is the maximum block size in kilobytes, and "megs" which is the minimum
 * number of megabytes to dedicate to the cache.
 *
 * Aborts program on error.
 *
 * Template parameters:
 * @li @c Point - a conformant point type (see gnp.h)
 * @li @c Param - a parameter object used for initializing points
 *
 * @param param the parameter object used for initializing the points
 * @param points_channel an rpc channel number that can be used for points
 * @param extra_channel an extra channel used internally for communication
 * @param module parameters (see above)
 * @param points_cache an uninitialized cache that will contain the points
 *
 * @return the number of points read
 */
template<typename Point, typename Param>
index_t ReadPoints(
    const Param& param, int points_channel, int extra_channel,
    datanode *module, DistributedCache *points_cache);

/**
 * Does a distributed dual-tree computation.
 *
 * @param module where to get tuning parameters from and store results in
 * @param base_channel the begin of a range of 10 free channels
 * @param param the gnp parameters
 * @param q the query tree
 * @param r the reference tree
 * @param q_results the query results
 * @param global_result_pp (output) if non-NULL, the global result will be
 *        allocated via @c new and stored here only on the master machine
 */
template<typename GNP, typename SerialSolver, typename QTree, typename RTree>
void RpcDualTree(datanode *module, int base_channel,
    const typename GNP::Param& param, QTree *q, RTree *r,
    DistributedCache *q_results,
    typename GNP::GlobalResult **global_result_pp);

/**
 * A "cookie-cutter" main for monochromatic dual tree problems.
 *
 * The @c gnp_name parameter specifies a short name of the problem solved.
 * If @c gnp_name is @c kde, then for example the @c Param object will
 * be initialized with the datanode in @c kde, so on the command line you
 * might type @c kde/bandwidth=1.
 *
 * Other parameters:
 * @li @c n_threads - the number of threads (defaults to 2)
 * @li @c data - the data file, with sub-parameters explained in
 *     thor::ReadPoints such as @c data/block_size_kb or @c data/megs
 * @li @c tree - parameters for tree building, such as @c tree/block_size_kb,
 *     @c tree/megs, and @c tree/leaf_size.
 *
 * @param module where to read parameters (explained above)
 * @param gnp_name a short textual name of the GNP
 */
template<typename GNP, typename Solver>
void MonochromaticDualTreeMain(datanode *module, const char *gnp_name);

}; // end namespace

#include "thor_utils_impl.h"

#endif
