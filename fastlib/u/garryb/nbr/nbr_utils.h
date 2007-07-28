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

          fprintf(stderr, "- Grain %"LI"d with %d points, %d..%d\n",
              work[i].point_begin_index, work[i].n_points(),
              work[i].node_index, work[i].node_end_index);

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

template<typename GNP, typename Solver>
class RpcMonochromaticDualTreeRunner {
 private:
  /**
   * The configuration of the problem.
   */
  struct Config {
    typename GNP::Param param;
    int n_threads;
    int data_points_mb;
    int data_nodes_mb;
    int q_results_mb;

    void Copy(const Config& other) {
      *this = other;
    }

    OT_DEF(Config) {
      OT_MY_OBJECT(param);
      OT_MY_OBJECT(n_threads);
      OT_MY_OBJECT(data_points_mb);
      OT_MY_OBJECT(data_nodes_mb);
      OT_MY_OBJECT(q_results_mb);
    }
  };

  struct Master {
    DataGetterBackend<Config> config_backend;
    RemoteWorkQueueBackend work_backend;
  };

 private:
  static const size_t MEGABYTE = 1048576;

  static const int MASTER_RANK = 0;

  static const int BARRIER_CHANNEL = 100;
  static const int DATA_POINTS_CHANNEL = 110;
  static const int DATA_NODES_CHANNEL = 111;
  static const int Q_RESULTS_CHANNEL = 112;
  static const int PARAM_CHANNEL = 120;
  static const int CONFIG_CHANNEL = 121;
  static const int WORK_CHANNEL = 122;

  static const int IOSTATS_POINTS_CHANNEL = 130;
  static const int IOSTATS_NODES_CHANNEL = 131;
  static const int IOSTATS_RESULTS_CHANNEL = 132;

  static const int GLOBAL_RESULT_CHANNEL = 135;

 private:
  datanode *module_;
  datanode *data_module_;
  const char *gnp_name_;

  Config config_;
  WorkQueueInterface *work_queue_;
  index_t n_points_;
  index_t dim_;
  Master *master_;
  DistributedCache data_points_;
  DistributedCache data_nodes_;
  DistributedCache q_results_;

 public:
  RpcMonochromaticDualTreeRunner() {
    master_ = NULL;
    work_queue_ = NULL;
  }
  ~RpcMonochromaticDualTreeRunner() {
    if (master_) {
      delete master_;
    }
    if (work_queue_) {
      delete work_queue_;
    }
  }

  void Doit(datanode *module, const char *gnp_name);

 private:
  // insert prototypes
  void Preinit_();
  void ReadData_();
  void MakeTree_();
  void SetupMaster_();
  void SetupConfig_();
};

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::Preinit_() {
  if (!rpc::is_root()) {
    //String my_fx_scope;
    //my_fx_scope.InitSprintf("rank%d", rpc::rank());
    //fx_scope(my_fx_scope.c_str());
    fx_silence();
  }
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::ReadData_() {
  index_t n_block_points = fx_param_int(module_, "n_block_points", 1024);

  data_module_ = fx_submodule(module_, "data", "data");

  fprintf(stderr, "master: Reading data\n");
  fx_timer_start(module_, "read");
  Matrix data_matrix;
  MUST_PASS(data::Load(fx_param_str_req(data_module_, ""), &data_matrix));
  fx_timer_stop(module_, "read");

  n_points_ = data_matrix.n_cols();
  dim_ = data_matrix.n_rows();

  fprintf(stderr, "master: Copying data to the cache\n");
  fx_timer_start(module_, "copy");
  typename GNP::QPoint default_point;
  default_point.vec().Init(dim_);
  config_.param.BootstrapMonochromatic(&default_point, n_points_);

  CacheArray<typename GNP::QPoint>::InitDistributedCacheMaster(
      DATA_POINTS_CHANNEL, n_block_points,
      config_.data_points_mb * MEGABYTE, default_point,
      &data_points_);
  CacheArray<typename GNP::QPoint> data_points_array;
  data_points_array.Init(&data_points_, BlockDevice::M_CREATE);
  data_points_array.AllocD(rpc::rank(), n_points_);

  for (index_t i = 0; i < n_points_; i++) {
    CacheWrite<typename GNP::QPoint> point(&data_points_array, i);
    point->vec().CopyValues(data_matrix.GetColumnPtr(i));
  }
  fx_timer_stop(module_, "copy");

  // also, set up the results array
  typename GNP::QResult default_result;
  default_result.Init(config_.param);
  CacheArray<typename GNP::QResult>::InitDistributedCacheMaster(
      Q_RESULTS_CHANNEL, n_block_points,
      config_.q_results_mb * MEGABYTE, default_result,
      &q_results_);
  CacheArray<typename GNP::QResult> q_results_array;
  q_results_array.Init(&q_results_, BlockDevice::M_CREATE);
  q_results_array.AllocD(rpc::rank(), n_points_);
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::MakeTree_() {
  index_t n_block_nodes = fx_param_int(module_, "n_block_nodes", 512);

  fprintf(stderr, "master: Building tree\n");
  fx_timer_start(module_, "tree");
  typename GNP::QNode data_example_node;
  data_example_node.Init(dim_, config_.param);
  CacheArray<typename GNP::QNode>::InitDistributedCacheMaster(
      DATA_NODES_CHANNEL, n_block_nodes,
      config_.data_nodes_mb * MEGABYTE, data_example_node,
      &data_nodes_);
  KdTreeHybridBuilder
      <typename GNP::QPoint, typename GNP::QNode, typename GNP::Param>
      ::Build(data_module_, config_.param, 0, n_points_,
          &data_points_, &data_nodes_);
  fx_timer_stop(module_, "tree");
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::SetupConfig_() {
  // Set up and export the config object
  config_.n_threads = fx_param_int(module_, "n_threads", 1);
  config_.data_points_mb = fx_param_int(module_, "data_points_mb", 400);
  config_.data_nodes_mb = fx_param_int(module_, "data_nodes_mb", 100);
  config_.q_results_mb = fx_param_int(module_, "q_results_mb", 100);
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::SetupMaster_() {
  master_ = new Master();
  master_->config_backend.Init(&config_);
  rpc::Register(CONFIG_CHANNEL, &master_->config_backend);

  // Make a static work queue
  CentroidWorkQueue<typename GNP::QNode> *actual_work_queue =
      new CentroidWorkQueue<typename GNP::QNode>;
  int n_grains = fx_param_int(module_, "n_grains",
      config_.n_threads * rpc::n_peers() * 12);
  CacheArray<typename GNP::QNode> data_nodes_array;
  data_nodes_array.Init(&data_nodes_, BlockDevice::M_READ);
  actual_work_queue->Init(&data_nodes_array, n_grains);
  fx_format_result(module_, "n_grains_actual", "%"LI"d",
      actual_work_queue->n_grains());
  work_queue_ = new LockedWorkQueue(actual_work_queue);

  master_->work_backend.Init(work_queue_);
  rpc::Register(WORK_CHANNEL, &master_->work_backend);
}

class IoStatsReductor {
 public:
  void Reduce(const IoStats& right, IoStats* left) const {
    left->Add(right);
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

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeRunner<GNP, Solver>::Doit(
    datanode *module, const char *gnp_name) {
  module_ = module;
  gnp_name_ = gnp_name;

  rpc::Init();
  Preinit_();

  fx_timer_start(module_, "load_data");
  if (rpc::is_root()) {
    SetupConfig_();
    fprintf(stderr, "nbr_utils(%d): reading in data, making tree, and distributing data\n",
        rpc::rank());
    config_.param.Init(fx_submodule(module_, gnp_name_, gnp_name_));
    ReadData_();
    MakeTree_();

    SetupMaster_();
  } else {
    data_points_.InitWorker(DATA_POINTS_CHANNEL,
        config_.data_points_mb * MEGABYTE,
        new CacheArrayBlockHandler<typename GNP::QPoint>);
    data_nodes_.InitWorker(DATA_NODES_CHANNEL,
        config_.data_nodes_mb * MEGABYTE,
        new CacheArrayBlockHandler<typename GNP::QNode>);
    q_results_.InitWorker(Q_RESULTS_CHANNEL,
        config_.q_results_mb * MEGABYTE,
        new CacheArrayBlockHandler<typename GNP::QResult>);

    rpc::GetRemoteData(CONFIG_CHANNEL, MASTER_RANK, &config_);

    RemoteWorkQueue *remote_work_queue = new RemoteWorkQueue();
    remote_work_queue->Init(WORK_CHANNEL, MASTER_RANK);
    work_queue_ = remote_work_queue;
  }

  data_points_.StartSync();
  data_nodes_.StartSync();
  q_results_.StartSync();
  data_points_.WaitSync();
  data_nodes_.WaitSync();
  q_results_.WaitSync();
  fx_timer_stop(module_, "load_data");

  //data_points_.ReportStats(true, fx_submodule(module_, NULL, "config_points"));
  //data_nodes_.ReportStats(true, fx_submodule(module_, NULL, "config_nodes"));
  //q_results_.ReportStats(true, fx_submodule(module_, NULL, "config_results"));

  fprintf(stderr, "nbr_utils(%d): starting the GNP\n", rpc::rank());

  fx_timer_start(module_, "all_machines");
  ThreadedDualTreeSolver<GNP, Solver> solver;
  solver.Doit(
      fx_submodule(module_, "solver", "local"),
      config_.n_threads, rpc::rank(), work_queue_, config_.param,
      &data_points_, &data_nodes_,
      &data_points_, &data_nodes_,
      &q_results_);
  q_results_.StartSync();
  q_results_.WaitSync();
  fx_timer_stop(module_, "all_machines");

  //rpc::Reduce(IOSTATS_POINTS_CHANNEL, IoStatsReductor(), &data_points_.stats());
  //rpc::Reduce(IOSTATS_NODES_CHANNEL, IoStatsReductor(), &data_nodes_.stats());
  //rpc::Reduce(IOSTATS_RESULTS_CHANNEL, IoStatsReductor(), &q_results_.stats());

  GlobalResultReductor<GNP> global_result_reductor;
  global_result_reductor.Init(&config_.param);
  rpc::Reduce(GLOBAL_RESULT_CHANNEL,
      global_result_reductor, &solver.global_result());

  //data_points_.ReportStats(true, fx_submodule(module_, NULL, "gnp_points"));
  //data_nodes_.ReportStats(true, fx_submodule(module_, NULL, "gnp_nodes"));
  //q_results_.ReportStats(true, fx_submodule(module_, NULL, "gnp_results"));

  work_queue_->Report(fx_submodule(module_, NULL, "work_queue"));
  solver.global_result().Report(config_.param,
      fx_submodule(module_, NULL, "global_result"));

  rpc::Done();
}

template<typename GNP, typename Solver>
void RpcMonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  RpcMonochromaticDualTreeRunner<GNP, Solver> runner;
  runner.Doit(module, gnp_name);
}

};


#endif
