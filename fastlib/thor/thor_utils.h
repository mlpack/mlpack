#ifndef THOR_UTILS_H
#define THOR_UTILS_H

#include "kdtree.h"
#include "work.h"
#include "rpc.h"

#include "par/thread.h"
#include "par/task.h"

#include "distribcache.h"

// TODO: These classes all need comments

namespace thor_utils {

template<typename GNP, typename Solver>
class ThreadedDualTreeSolver {
 private:
  class WorkerTask : public Task {
   private:
    ThreadedDualTreeSolver *base_;

   public:
    WorkerTask(ThreadedDualTreeSolver *base_solver)
        : base_(base_solver)
      { }

    void Run() {
      while (1) {
        ArrayList<WorkQueueInterface::Grain> work;

        base_->mutex_.Lock();
        base_->work_queue_->GetWork(base_->process_, &work);
        base_->mutex_.Unlock();

        if (work.size() == 0) {
          break;
        }

        for (index_t i = 0; i < work.size(); i++) {
          Solver solver;

          //fprintf(stderr, "- Grain with %"LI"d points starting at %"LI"d\n",
          //    work[i].n_points(),
          //    work[i].point_begin_index);

          solver.Doit(*base_->param_,
              work[i].node_index, work[i].node_end_index,
              base_->q_points_cache_, base_->q_nodes_cache_,
              base_->r_points_cache_, base_->r_nodes_cache_,
              base_->q_results_cache_);

          base_->mutex_.Lock();
          base_->global_result_.Accumulate(
              *base_->param_, solver.global_result());
          base_->mutex_.Unlock();
        }
      }
      delete this;
    }
  };

 private:
  int process_;
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
  static void Solve(
      datanode *module,
      const typename GNP::Param& param_in,
      CacheArray<typename GNP::QPoint> *q_points_array_in,
      CacheArray<typename GNP::QNode> *q_nodes_array_in,
      CacheArray<typename GNP::RPoint> *r_points_array_in,
      CacheArray<typename GNP::RNode> *r_nodes_array_in,
      CacheArray<typename GNP::QResult> *q_results_array_in) {
    index_t n_threads = fx_param_int(module, "n_threads", 2);
    index_t n_grains = fx_param_int(module, "n_grains",
        n_threads == 1 ? 1 : (n_threads * 3));
    SimpleWorkQueue<typename GNP::QNode> simple_work_queue;
    simple_work_queue.Init(q_nodes_array_in, n_grains);
    fx_format_result(module, "n_grains_actual", "%"LI"d",
        simple_work_queue.n_grains());

    fx_timer_start(module, "all_threads");

    ThreadedDualTreeSolver solver;
    solver.Doit(
        n_threads, 0, &simple_work_queue,
        param_in,
        q_points_array_in->cache(), q_nodes_array_in->cache(),
        r_points_array_in->cache(), r_nodes_array_in->cache(),
        q_results_array_in->cache());

    fx_timer_stop(module, "all_threads");
  }

  void Doit(
      index_t n_threads,
      int process,
      WorkQueueInterface *work_queue_in,
      const typename GNP::Param& param,
      DistributedCache *q_points_cache_in, DistributedCache *q_nodes_cache_in,
      DistributedCache *r_points_cache_in, DistributedCache *r_nodes_cache_in,
      DistributedCache *q_results_cache_in) {
    param_ = &param;
    work_queue_ = work_queue_in;
    process_ = process;

    q_points_cache_ = q_points_cache_in;
    q_nodes_cache_ = q_nodes_cache_in;
    r_points_cache_ = r_points_cache_in;
    r_nodes_cache_ = r_nodes_cache_in;
    q_results_cache_ = q_results_cache_in;

    ArrayList<Thread*> threads;
    threads.Init(n_threads);

    global_result_.Init(*param_);

    for (index_t i = 0; i < n_threads; i++) {
      threads[i] = new Thread();
      threads[i]->Init(new WorkerTask(this));
      threads[i]->Start();
    }

    for (index_t i = 0; i < n_threads; i++) {
      threads[i]->WaitStop();
      delete threads[i];
    }
  }

  const typename GNP::GlobalResult& global_result() const {
    return global_result_;
  }

  typename GNP::GlobalResult& global_result() {
    return global_result_;
  }
};

template<typename GNP>
class GlobalResultReductor {
 private:
  const typename GNP::Param *param_;

 public:
  void Init(const typename GNP::Param* param_in) {
    param_ = param_in;
  }

  void Reduce(typename GNP::GlobalResult& right,
      typename GNP::GlobalResult* left) const {
    left->Accumulate(*param_, right);
  }
};


/**
 * Does a distributed dual-tree computation.
 *
 * @param module where to get tuning parameters from and store results in
 * @param base_channel the begin of a range of 10 free channels
 * @param param the gnp parameters
 * @param q the query tree
 * @param r the reference tree
 * @param q_results the query results
 * @param global_result (output) if non-NULL, the global result will be
 *        allocated via new() and stored here only on the master machine
 */
template<typename GNP, typename SerialSolver, typename QTree, typename RTree>
void RpcDualTree(
    datanode *module,
    int base_channel,
    const typename GNP::Param& param,
    QTree *q,
    RTree *r,
    DistributedCache *q_results,
    typename GNP::GlobalResult **global_result) {
  int n_threads = fx_param_int(module, "n_threads", 2);
  RemoteWorkQueueBackend *work_backend = NULL;
  WorkQueueInterface *work_queue;

  if (rpc::is_root()) {
    // Make a static work queue
    CentroidWorkQueue<typename GNP::QNode> *actual_work_queue =
        new CentroidWorkQueue<typename GNP::QNode>;
    CacheArray<typename GNP::QNode> q_nodes_array;
    q_nodes_array.Init(&q->nodes(), BlockDevice::M_READ);
    actual_work_queue->Init(&q_nodes_array,
        q->decomposition().root(), n_threads, module);
    work_queue = new LockedWorkQueue(actual_work_queue);

    work_backend = new RemoteWorkQueueBackend();
    work_backend->Init(work_queue);
    rpc::Register(base_channel + 0, work_backend);
  } else {
    RemoteWorkQueue *remote_work_queue = new RemoteWorkQueue();
    remote_work_queue->Init(base_channel + 0, 0);
    work_queue = remote_work_queue;
  }

  rpc::Barrier(base_channel + 1);

  fx_timer_start(fx_submodule(module, NULL, "gnp"), "all_machines");
  ThreadedDualTreeSolver<GNP, SerialSolver> solver;
  solver.Doit(
      n_threads, rpc::rank(), work_queue, param,
      &q->points(), &q->nodes(),
      &r->points(), &r->nodes(),
      q_results);
  rpc::Barrier(base_channel + 1);
  fx_timer_stop(fx_submodule(module, NULL, "gnp"), "all_machines");

  fx_timer_start(fx_submodule(module, NULL, "gnp"), "write_results");
  q->points().StartSync();
  q->nodes().StartSync();
  if (r != q) {
    r->points().StartSync();
    r->nodes().StartSync();
  }
  q_results->StartSync();
  q->points().WaitSync(fx_submodule(module, NULL, "io/gnp/q_points"));
  q->nodes().WaitSync(fx_submodule(module, NULL, "io/gnp/q_nodes"));
  if (r != q) {
    r->points().WaitSync(fx_submodule(module, NULL, "io/gnp/r_points"));
    r->nodes().WaitSync(fx_submodule(module, NULL, "io/gnp/r_nodes"));
  }
  q_results->WaitSync(fx_submodule(module, NULL, "io/gnp/q_results"));
  fx_timer_stop(fx_submodule(module, NULL, "gnp"), "write_results");

  GlobalResultReductor<GNP> global_result_reductor;
  global_result_reductor.Init(&param);
  rpc::Reduce(base_channel + 5,
      global_result_reductor, &solver.global_result());

  work_queue->Report(fx_submodule(module, NULL, "work_queue"));
  solver.global_result().Report(param,
      fx_submodule(module, NULL, "global_result"));

  if (rpc::is_root()) {
    delete work_backend;
    if (global_result) {
      *global_result = new typename GNP::GlobalResult(solver.global_result());
    }
  } else {
    if (global_result) {
      *global_result = NULL;
    }
  }

  delete work_queue;
}

/**
 * A "cookie-cutter" main for monochromatic dual tree problems.
 */
template<typename GNP, typename Solver>
void MonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  const size_t MEGABYTE = 1048576;
  const int DATA_CHANNEL = 110;
  const int Q_RESULTS_CHANNEL = 120;
  const int GNP_CHANNEL = 200;

  rpc::Init();

  if (!rpc::is_root()) {
    fx_silence();
  }

  size_t q_results_mb = fx_param_int(module, "q_results_mb", 1000);
  fx_submodule(module, NULL, "io"); // influnce output order

  ThorKdTree<typename GNP::Param, typename GNP::QPoint, typename GNP::QNode>
      data;
  DistributedCache q_results;
  typename GNP::Param *param = new typename GNP::Param();
  param->Init(fx_submodule(module, gnp_name, gnp_name));

  fx_timer_start(module, "data");
  data.Init(&param, 0, DATA_CHANNEL, fx_submodule(module, "data", "data"));
  fx_timer_stop(module, "data");

  typename GNP::QResult default_result;
  default_result.Init(*param);
  data.InitDistributedCache(Q_RESULTS_CHANNEL, default_result,
        size_t(q_results_mb) * MEGABYTE, &q_results);

  typename GNP::GlobalResult *global_result;
  RpcDualTree<GNP, Solver>(module, GNP_CHANNEL, *param,
      &data, &data, &q_results, &global_result);
  delete global_result;
  delete param;

  rpc::Done();
}

}; // end namespace

#endif
