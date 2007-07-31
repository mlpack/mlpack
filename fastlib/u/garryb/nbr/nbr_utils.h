#ifndef NBR_UTILS_H
#define NBR_UTILS_H

#include "kdtree.h"
#include "work.h"
#include "rpc.h"

#include "par/thread.h"
#include "par/task.h"

#include "distribcache.h"

// TODO: These classes all need comments

namespace nbr_utils {

template<typename Param, typename Point, typename Node>
class StatFixer {
 public:
  static void Fix(const Param &param,
      CacheArray<Point> *points, CacheArray<Node> *nodes) {
    StatFixer fixer;
    fixer.Doit(&param, points, nodes);
  }

 private:
  const Param *param_;
  CacheArray<Point> points_;
  CacheArray<Node> nodes_;

 public:
  void Doit(const Param *param,
      CacheArray<Point> *points, CacheArray<Node> *nodes);

 private:
  void FixRecursively_(index_t node_index);
};

template<typename Param, typename Point, typename Node>
void StatFixer<Param, Point, Node>::Doit(
    const Param *param, CacheArray<Point> *points, CacheArray<Node> *nodes) {
  param_ = param;
  points_.Init(points, BlockDevice::M_READ);
  nodes_.Init(nodes, BlockDevice::M_MODIFY);
  FixRecursively_(0);
  nodes_.Flush();
  points_.Flush();
}

template<typename Param, typename Point, typename Node>
void StatFixer<Param, Point, Node>::FixRecursively_(index_t node_index) {
  CacheWrite<Node> node(&nodes_, node_index);

  node->stat().Reset(*param_);

  if (!node->is_leaf()) {
    for (index_t k = 0; k < 2; k++) {
      index_t child_index = node->child(k);

      FixRecursively_(child_index);

      CacheRead<Node> child(&nodes_, child_index);
      node->stat().Accumulate(*param_, child->stat(),
          child->bound(), child->count());
    }
    node->stat().Postprocess(*param_, node->bound(),
        node->count());
  } else {
    CacheReadIter<Point> point(&points_, node->begin());

    for (index_t i = 0; i < node->count(); i++, point.Next()) {
      node->stat().Accumulate(*param_, *point);
    }
  }

  node->stat().Postprocess(*param_, node->bound(), node->count());
}

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

          base_->mutex_.Lock();
          struct datanode *submodule = fx_submodule(base_->module_,
              "solver", "grain_%d", work[i].point_begin_index);
          base_->mutex_.Unlock();

          fprintf(stderr, "- Grain with %"LI"d points starting at %"LI"d\n",
              work[i].n_points(),
              work[i].point_begin_index);

          solver.Doit(submodule, *base_->param_,
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

      fprintf(stderr, "- Thread Done\n");
      delete this;
    }
  };

 private:
  datanode *module_;
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
    index_t n_threads = fx_param_int(module, "n_threads", 1);
    index_t n_grains = fx_param_int(module, "n_grains",
        n_threads == 1 ? 1 : (n_threads * 3));
    SimpleWorkQueue<typename GNP::QNode> simple_work_queue;
    simple_work_queue.Init(q_nodes_array_in, n_grains);
    fx_format_result(module, "n_grains_actual", "%"LI"d",
        simple_work_queue.n_grains());

    ThreadedDualTreeSolver solver;
    solver.Doit(module,
        n_threads, 0, &simple_work_queue,
        param_in,
        q_points_array_in->cache(), q_nodes_array_in->cache(),
        r_points_array_in->cache(), r_nodes_array_in->cache(),
        q_results_array_in->cache());
  }

  void Doit(
      datanode *module,
      index_t n_threads,
      int process,
      WorkQueueInterface *work_queue_in,
      const typename GNP::Param& param,
      DistributedCache *q_points_cache_in,
      DistributedCache *q_nodes_cache_in,
      DistributedCache *r_points_cache_in,
      DistributedCache *r_nodes_cache_in,
      DistributedCache *q_results_cache_in
      ) {
    module_ = module;
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

    fx_timer_start(module, "all_threads");

    for (index_t i = 0; i < n_threads; i++) {
      threads[i] = new Thread();
      threads[i]->Init(new WorkerTask(this));
      threads[i]->Start();
    }

    for (index_t i = 0; i < n_threads; i++) {
      threads[i]->WaitStop();
      delete threads[i];
    }

    fx_timer_stop(module, "all_threads");

    global_result_.Report(*param_,
        fx_submodule(module, NULL, "global_result"));
  }
  
  const typename GNP::GlobalResult& global_result() const {
    return global_result_;
  }
  typename GNP::GlobalResult& global_result() {
    return global_result_;
  }
};

///**
// * Dual-tree main for monochromatic problems.
// *
// * A bichromatic main isn't that much harder to write, it's just kind of
// * tedious -- we will save this for later.
// */
//template<typename GNP, typename SerialSolver>
//void MonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
//  typename GNP::Param param;
//
//  param.Init(fx_submodule(module, gnp_name, gnp_name));
//
//  TempCacheArray<typename GNP::QPoint> data_points;
//  TempCacheArray<typename GNP::QNode> data_nodes;
//  TempCacheArray<typename GNP::QResult> q_results;
//
//  index_t n_block_points = fx_param_int(
//      module, "n_block_points", 512);
//  index_t n_block_nodes = fx_param_int(
//      module, "n_block_nodes", 128);
//
//  datanode *data_module = fx_submodule(module, "data", "data");
//  fx_timer_start(module, "read");
//
//  Matrix data_matrix;
//  MUST_PASS(data::Load(fx_param_str_req(data_module, ""), &data_matrix));
//  typename GNP::QPoint default_point;
//  default_point.vec().Init(data_matrix.n_rows());
//  param.BootstrapMonochromatic(&default_point, data_matrix.n_cols());
//  data_points.Init(default_point, data_matrix.n_cols(), n_block_points);
//  for (index_t i = 0; i < data_matrix.n_cols(); i++) {
//    CacheWrite<typename GNP::QPoint> point(&data_points, i);
//    point->vec().CopyValues(data_matrix.GetColumnPtr(i));
//  }
//
//  fx_timer_stop(module, "read");
//
//  typename GNP::QNode data_example_node;
//  data_example_node.Init(data_matrix.n_rows(), param);
//  data_nodes.Init(data_example_node, 0, n_block_nodes);
//  KdTreeHybridBuilder
//      <typename GNP::QPoint, typename GNP::QNode, typename GNP::Param>
//      ::Build(data_module, param, 0, data_matrix.n_cols(),
//          &data_points, &data_nodes);
//
//  // Create our array of results.
//  typename GNP::QResult default_result;
//  default_result.Init(param);
//  q_results.Init(default_result, data_points.end_index(),
//      data_points.n_block_elems());
//
//  // TODO: Send global result
//  ThreadedDualTreeSolver<GNP, SerialSolver>::Solve(
//      module, param,
//      &data_points, &data_nodes,
//      &data_points, &data_nodes,
//      &q_results);
//}
//
///*
//- add the code for loading the tree
//- standardize channel numbers
//- write code that detects and runs the server
//*/

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

template<typename GNP, typename SerialSolver, typename QTree, typename RTree>
void RpcDualTree(
    datanode *module,
    int base_channel,
    const typename GNP::Param& param,
    QTree *q,
    RTree *r,
    DistributedCache *q_results,
    typename GNP::GlobalResult **global_result) {
  fprintf(stderr, "nbr_utils(%d): starting the GNP\n", rpc::rank());
  int n_threads = fx_param_int(module, "n_threads", 1);
  datanode *local_module = fx_submodule(module, "solver", "gnp/local");
  RemoteWorkQueueBackend *work_backend = NULL;
  WorkQueueInterface *work_queue;

  if (rpc::is_root()) {
    // Make a static work queue
    CentroidWorkQueue<typename GNP::QNode> *actual_work_queue =
        new CentroidWorkQueue<typename GNP::QNode>;
    int n_grains = fx_param_int(module, "n_grains",
        n_threads * rpc::n_peers() * 12);
    CacheArray<typename GNP::QNode> q_nodes_array;
    q_nodes_array.Init(&q->nodes(), BlockDevice::M_READ);
    actual_work_queue->Init(&q_nodes_array, n_grains);
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
      local_module,
      n_threads, rpc::rank(), work_queue, param,
      &q->points(), &q->nodes(),
      &r->points(), &r->nodes(),
      q_results);
  rpc::Barrier(base_channel + 1);
  fx_timer_stop(fx_submodule(module, NULL, "gnp"), "all_machines");
  if (rpc::is_root()) {
    fprintf(stderr, "nbr_utils(%d): All results computed, writing results\n");
  }
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
    *global_result = new typename GNP::GlobalResult(solver.global_result());
  } else {
    *global_result = NULL;
  }

  delete work_queue;
}

template<typename GNP, typename Solver>
class RpcMonochromaticDualTreeRunner {
 private:
  static const size_t MEGABYTE = 1048576;

  static const int MASTER_RANK = 0;

  static const int BARRIER_CHANNEL = 100;
  static const int DATA_CHANNEL = 110;
  static const int Q_RESULTS_CHANNEL = 120;

  static const int GNP_CHANNEL = 150;

  static const int GLOBAL_RESULT_CHANNEL = 145;

 private:
  datanode *module_;
  datanode *data_module_;
  const char *gnp_name_;

  typename GNP::Param *param_;
  index_t dim_;
  SpKdTree<typename GNP::Param, typename GNP::QPoint, typename GNP::QNode> data_;
  DistributedCache q_results_;

 public:
  RpcMonochromaticDualTreeRunner() {
  }
  ~RpcMonochromaticDualTreeRunner() {
    delete param_;
  }

  void Doit(datanode *module, const char *gnp_name);
  void Init(datanode *module, const char *gnp_name);
  void DoGNP();

 private:
  void InitResults_(size_t q_results_mb);
};

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::InitResults_(
    size_t q_results_mb) {
  // also, set up the results array
  typename GNP::QResult default_result;
  default_result.Init(*param_);
  CacheArray<typename GNP::QResult>::InitDistributedCacheMaster(
      Q_RESULTS_CHANNEL, data_.points_block(), default_result,
      size_t(q_results_mb) * MEGABYTE,
      &q_results_);
  CacheArray<typename GNP::QResult> q_results_array;
  q_results_array.Init(&q_results_, BlockDevice::M_CREATE);
  q_results_array.AllocD(rpc::rank(), data_.n_points());
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::Init(
    datanode *module, const char *gnp_name) {
  module_ = module;
  gnp_name_ = gnp_name;

  size_t q_results_mb = fx_param_int(module_, "q_results_mb", 1000);
  fx_submodule(module_, NULL, "io"); // influnce output order

  param_ = new typename GNP::Param();
  param_->Init(fx_submodule(module_, gnp_name_, gnp_name_));

  fx_timer_start(module_, "data");
  data_.Init(&param_, 0, DATA_CHANNEL, fx_submodule(module_, "data", "data"));
  fx_timer_stop(module_, "data");

  if (rpc::is_root()) {
    InitResults_(q_results_mb);
  } else {
    q_results_.InitWorker(Q_RESULTS_CHANNEL,
        size_t(q_results_mb) * MEGABYTE,
        new CacheArrayBlockHandler<typename GNP::QResult>);
  }

  q_results_.StartSync();
  q_results_.WaitSync();
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::DoGNP() {
  typename GNP::GlobalResult *global_result;
  RpcDualTree<GNP, Solver>(module_, GNP_CHANNEL, *param_,
      &data_, &data_, &q_results_,
      &global_result);
  delete global_result;
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::Doit(
    datanode *module, const char *gnp_name) {
  rpc::Init();

  if (!rpc::is_root()) {
    fx_silence();
  }

  Init(module, gnp_name);
  DoGNP();
  rpc::Done();
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  RpcMonochromaticDualTreeRunner<GNP, Solver> runner;
  runner.Doit(module, gnp_name);
}

};


#endif
