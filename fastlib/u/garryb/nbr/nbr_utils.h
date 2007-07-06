#ifndef NBR_UTILS_H
#define NBR_UTILS_H

#include "kdtree.h"
#include "work.h"
#include "rpc.h"

#include "par/thread.h"
#include "par/task.h"

#ifdef USE_MPI
#include "netcache.h"
#endif

namespace nbr_utils {

template<typename Param, typename Point, typename Node>
class StatFixer {
 public:
  static void Fix(const Param &param,
      CacheArray<Point> *points, CacheArray<Node> *nodes) {
    StatFixer fixer;
    fixer.InitFix(&param, points, nodes);
  }

 private:
  const Param *param_;
  CacheArray<Point> points_;
  CacheArray<Node> nodes_;

 public:
  void InitFix(const Param *param,
      CacheArray<Point> *points, CacheArray<Node> *nodes);

 private:
  void FixRecursively_(index_t node_index);
};

template<typename Param, typename Point, typename Node>
void StatFixer<Param, Point, Node>::InitFix(
    const Param *param, CacheArray<Point> *points, CacheArray<Node> *nodes) {
  param_ = param;
  points_.Init(points, BlockDevice::READ);
  nodes_.Init(nodes, BlockDevice::MODIFY);
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
    CacheReadIterator<Point> point(&points_, node->begin());

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
        ArrayList<index_t> work;

        base_->mutex_.Lock();
        base_->work_queue_->GetWork(&work);
        base_->mutex_.Unlock();

        if (work.size() == 0) {
          break;
        }

        for (index_t i = 0; i < work.size(); i++) {
          index_t q_root_index = work[i];
          Solver solver;

          base_->mutex_.Lock();
          struct datanode *submodule = fx_submodule(base_->module_,
              "solver", "grain_%d", work[i]);
          base_->mutex_.Unlock();

          solver.InitSolve(submodule, *base_->param_, q_root_index,
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
  datanode *module_;
  const typename GNP::Param *param_;
  WorkQueueInterface *work_queue_;
  SmallCache *q_points_cache_;
  SmallCache *q_nodes_cache_;
  SmallCache *r_points_cache_;
  SmallCache *r_nodes_cache_;
  SmallCache *q_results_cache_;
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
    solver.InitSolve(module,
        n_threads, &simple_work_queue,
        param_in,
        q_points_array_in->cache(), q_nodes_array_in->cache(),
        r_points_array_in->cache(), r_nodes_array_in->cache(),
        q_results_array_in->cache());
  }

  void InitSolve(
      datanode *module,
      index_t n_threads,
      WorkQueueInterface *work_queue_in,
      const typename GNP::Param& param,
      SmallCache *q_points_cache_in,
      SmallCache *q_nodes_cache_in,
      SmallCache *r_points_cache_in,
      SmallCache *r_nodes_cache_in,
      SmallCache *q_results_cache_in
      ) {
    module_ = module;
    param_ = &param;
    work_queue_ = work_queue_in;

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
};

/**
 * Dual-tree main for monochromatic problems.
 *
 * A bichromatic main isn't that much harder to write, it's just kind of
 * tedious -- we will save this for later.
 */
template<typename GNP, typename SerialSolver>
void MonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  typename GNP::Param param;

  param.Init(fx_submodule(module, gnp_name, gnp_name));

  TempCacheArray<typename GNP::QPoint> data_points;
  TempCacheArray<typename GNP::QNode> data_nodes;
  TempCacheArray<typename GNP::QResult> q_results;

  index_t n_block_points = fx_param_int(
      module, "n_block_points", 1024);
  index_t n_block_nodes = fx_param_int(
      module, "n_block_nodes", 128);

  datanode *data_module = fx_submodule(module, "data", "data");

  fx_timer_start(module, "read");

  Matrix data_matrix;
  MUST_PASS(data::Load(fx_param_str_req(data_module, ""), &data_matrix));
  typename GNP::QPoint default_point;
  default_point.vec().Init(data_matrix.n_rows());
  param.BootstrapMonochromatic(&default_point, data_matrix.n_cols());
  data_points.Init(default_point, data_matrix.n_cols(), n_block_points);
  for (index_t i = 0; i < data_matrix.n_cols(); i++) {
    CacheWrite<typename GNP::QPoint> point(&data_points, i);
    point->vec().CopyValues(data_matrix.GetColumnPtr(i));
  }

  fx_timer_stop(module, "read");

  typename GNP::QNode data_example_node;
  data_example_node.Init(data_matrix.n_rows(), param);
  data_nodes.Init(data_example_node, 0, n_block_nodes);
  KdTreeMidpointBuilder
      <typename GNP::QPoint, typename GNP::QNode, typename GNP::Param>
      ::Build(data_module, param, &data_points, &data_nodes);

  // Create our array of results.
  typename GNP::QResult default_result;
  default_result.Init(param);
  q_results.Init(default_result, data_points.end_index(),
      data_points.n_block_elems());

  ThreadedDualTreeSolver<GNP, SerialSolver>::Solve(
      module, param,
      &data_points, &data_nodes,
      &data_points, &data_nodes,
      &q_results);
}

/*
- add the code for loading the tree
- standardize channel numbers
- write code that detects and runs the server
*/

//problem problem - no network thread!

#ifdef USE_MPI


template<typename GNP, typename Solver>
class MpiMonochromaticDualTreeRunner {
 private:
  struct Config {
    int n_threads;

    void Copy(const Config& other) {
      *this = other;
    }

    OT_DEF(Config) {
      OT_MY_OBJECT(n_threads);
    }
  };

  struct Master {
    DataGetterBackend<Config> config_backend;
    DataGetterBackend<typename GNP::Param> param_backend;
    RemoteWorkQueueBackend work_backend;
  };

 private:
  static const int MASTER_RANK = 0;
  static const int DATA_POINTS_CHANNEL = 10;
  static const int DATA_NODES_CHANNEL = 11;
  static const int Q_RESULTS_CHANNEL = 12;
  static const int PARAM_CHANNEL = 20;
  static const int CONFIG_CHANNEL = 21;
  static const int WORK_CHANNEL = 22;

 private:
  datanode *module_;
  const char *gnp_name_;

  typename GNP::Param param_;
  Config config_;
  WorkQueueInterface *work_queue_;
  int my_rank_;
  int n_machines_;
  index_t n_points_;
  Master *master_;
  SimpleDistributedCacheArray<typename GNP::QPoint> data_points_;
  SimpleDistributedCacheArray<typename GNP::QNode> data_nodes_;
  SimpleDistributedCacheArray<typename GNP::QResult> q_results_;
  RpcServer server_;

 public:
  MpiMonochromaticDualTreeRunner() {
    master_ = NULL;
    work_queue_ = NULL;
  }
  ~MpiMonochromaticDualTreeRunner() {
    if (master_) {
      delete master_;
    }
    if (work_queue_) {
      delete work_queue_;
    }
  }

  void Main(datanode *module, const char *gnp_name);

 private:
  // insert prototypes
  void Preinit_();
  void ReadData_();
  void MakeTree_();
  void SetupMaster_();
};

template<typename GNP, typename Solver>
void MpiMonochromaticDualTreeRunner<GNP, Solver>::Preinit_() {
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &n_machines_);

  if (my_rank_ != MASTER_RANK) {
    String my_fx_scope;

    my_fx_scope.InitSprintf("rank%d", my_rank_);
    fx_scope(my_fx_scope.c_str());
  }
}

template<typename GNP, typename Solver>
void MpiMonochromaticDualTreeRunner<GNP, Solver>::ReadData_() {
  index_t n_block_points = fx_param_int(module_, "n_block_points", 1024);
  datanode *data_module = fx_submodule(module_, "data", "data");

  fx_timer_start(module_, "read");
  Matrix data_matrix;
  MUST_PASS(data::Load(fx_param_str_req(data_module, ""), &data_matrix));
  fx_timer_stop(module_, "read");

  n_points_ = data_matrix.n_cols();

  fx_timer_start(module_, "copy");
  typename GNP::QPoint default_point;
  default_point.vec().Init(data_matrix.n_rows());
  param_.BootstrapMonochromatic(&default_point, n_points_);
  data_points_.InitMaster(default_point, data_matrix.n_cols(), n_block_points);

  for (index_t i = 0; i < n_points_; i++) {
    CacheWrite<typename GNP::QPoint> point(&data_points_, i);
    point->vec().CopyValues(data_matrix.GetColumnPtr(i));
  }
  fx_timer_stop(module_, "copy");
}

template<typename GNP, typename Solver>
void MpiMonochromaticDualTreeRunner<GNP, Solver>::MakeTree_() {
  index_t n_block_nodes = fx_param_int(module_, "n_block_nodes", 128);
  datanode *data_module = fx_submodule(module_, "data", "data");

  fx_timer_start(module_, "tree");
  typename GNP::QNode data_example_node;
  data_example_node.Init(n_points_, param_);
  data_nodes_.InitMaster(data_example_node, 0, n_block_nodes);
  KdTreeMidpointBuilder
      <typename GNP::QPoint, typename GNP::QNode, typename GNP::Param>
      ::Build(data_module, param_, &data_points_, &data_nodes_);
  fx_timer_stop(module_, "tree");
}

template<typename GNP, typename Solver>
void MpiMonochromaticDualTreeRunner<GNP, Solver>::SetupMaster_() {
  master_ = new Master();

  // Set up and export the config object
  config_.n_threads = fx_param_int(module_, "n_threads", 1);
  master_->config_backend.Init(&config_);
  server_.Register(CONFIG_CHANNEL, &master_->config_backend);

  // Set up and export the dual-tree algorithm Param object
  master_->param_backend.Init(&param_);
  server_.Register(PARAM_CHANNEL, &master_->param_backend);

  // Make a static work queue
  SimpleWorkQueue<typename GNP::QNode> *simple_work_queue =
      new SimpleWorkQueue<typename GNP::QNode>;
  int n_grains = fx_param_int(module_, "n_grains",
      config_.n_threads * n_machines_ * 3);
  simple_work_queue->Init(&data_nodes_, n_grains);
  fx_format_result(module_, "n_grains_actual", "%"LI"d",
      simple_work_queue->n_grains());
  work_queue_ = new LockedWorkQueue(simple_work_queue);

  master_->work_backend.Init(work_queue_);
  server_.Register(WORK_CHANNEL, &master_->work_backend);
}

template<typename GNP, typename Solver>
void MpiMonochromaticDualTreeRunner<GNP, Solver>::Main(
    datanode *module, const char *gnp_name) {
  module_ = module;
  gnp_name_ = gnp_name;

  Preinit_();

  server_.Init();
  server_.Start();

  data_points_.Configure(DATA_POINTS_CHANNEL, my_rank_, n_machines_);
  server_.Register(data_points_.channel(), data_points_.server());
  data_nodes_.Configure(DATA_NODES_CHANNEL, my_rank_, n_machines_);
  server_.Register(data_nodes_.channel(), data_nodes_.server());
  q_results_.Configure(Q_RESULTS_CHANNEL, my_rank_, n_machines_);
  server_.Register(q_results_.channel(), q_results_.server());

  fx_timer_start(module_, "master_init");
  if (my_rank_ == MASTER_RANK) {
    param_.Init(fx_submodule(module_, gnp_name_, gnp_name_));
    ReadData_();
    MakeTree_();

    typename GNP::QResult default_result;
    default_result.Init(param_);
    q_results_.InitMaster(default_result, n_points_, data_points_.n_block_elems());

    SetupMaster_();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  fx_timer_stop(module_, "master_init");

  fx_timer_start(module_, "worker_init");
  if (my_rank_ != MASTER_RANK) {
    data_points_.InitWorker();
    data_nodes_.InitWorker();
    q_results_.InitWorker();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  fx_timer_stop(module_, "worker_init");

  fx_timer_start(module_, "flush_data");
  data_points_.FlushClear(BlockDevice::READ);
  data_nodes_.FlushClear(BlockDevice::READ);
  q_results_.FlushClear(BlockDevice::MODIFY);
  MPI_Barrier(MPI_COMM_WORLD);
  fx_timer_stop(module_, "flush_data");

  fx_timer_start(module_, "distribute_config");
  if (my_rank_ != MASTER_RANK) {
    GetRemoteData(CONFIG_CHANNEL, MASTER_RANK, &config_);
    GetRemoteData(PARAM_CHANNEL, MASTER_RANK, &param_);

    RemoteWorkQueue *remote_work_queue = new RemoteWorkQueue();
    remote_work_queue->Init(WORK_CHANNEL, MASTER_RANK);
    work_queue_ = remote_work_queue;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fx_timer_stop(module_, "distribute_config");

  fx_timer_start(module_, "all_machines");
  ThreadedDualTreeSolver<GNP, Solver> solver;
  solver.InitSolve(
      fx_submodule(module_, "solver", "solver"),
      config_.n_threads, work_queue_, param_,
      data_points_.cache(), data_nodes_.cache(),
      data_points_.cache(), data_nodes_.cache(),
      q_results_.cache());
  MPI_Barrier(MPI_COMM_WORLD);
  fx_timer_stop(module_, "all_machines");

  fx_timer_start(module_, "flush_results");
  q_results_.FlushClear(BlockDevice::READ);
  MPI_Barrier(MPI_COMM_WORLD);
  fx_timer_stop(module_, "flush_results");

  server_.Stop();
}

template<typename GNP, typename Solver>
void MpiMonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  MpiMonochromaticDualTreeRunner<GNP, Solver> runner;
  runner.Main(module, gnp_name);
}

#endif

// template<typename GNP, typename Solver>
// void MpiDualTreeMain(datanode *module, const char *gnp_name) {
//   const int MASTER_RANK = 0;
//   const int PARAM_CHANNEL = 2;
//   const int WORK_CHANNEL = 3;
//   const int Q_POINTS_CHANNEL = 4;
//   const int Q_NODES_CHANNEL = 5;
//   const int R_POINTS_CHANNEL = 6;
//   const int R_NODES_CHANNEL = 7;
//   const int Q_RESULTS_CHANNEL = 8;
//   const int CONFIG_CHANNEL = 9;
//   typename GNP::Param param;
//   int my_rank_;
//   int n_machines;
//   int n_workers_total;
//
//   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
//   MPI_Comm_size(MPI_COMM_WORLD, &n_machines);
//
//   n_workers_total = n_machines - 1;
//
//   if (my_rank_ == MASTER_RANK) {
//     param.Init(fx_submodule(module, gnp_name, gnp_name));
//
//     TempCacheArray<typename GNP::Point> q_points;
//     TempCacheArray<typename GNP::QNode> q_nodes;
//     TempCacheArray<typename GNP::Point> r_points;
//     TempCacheArray<typename GNP::RNode> r_nodes;
//     TempCacheArray<typename GNP::QResult> q_results_;
//
//     nbr_utils::LoadKdTree(fx_submodule(module, "q", "q"),
//         &param, &q_points, &q_nodes);
//     nbr_utils::LoadKdTree(fx_submodule(module, "r", "r"),
//         &param, &r_points, &r_nodes);
//
//     MpiDualTreeConfig config;
//     config.n_threads = fx_param_int(module, "n_threads", 1);
//     config.monochromatic = fx_param_bool(module, "monochromatic", 1);
//
//     typename GNP::QResult default_result;
//     default_result.Init(param);
//     q_results_.Init(default_result, q_points.end_index(),
//         q_points.n_block_elems());
//
//     SimpleWorkQueue<typename GNP::QNode> work_queue;
//     int n_grains = fx_param_int(module, "n_grains",
//         config.n_threads * n_workers_total * 3);
//     work_queue.Init(&q_nodes, n_grains);
//     fx_format_result(module, "n_grains_actual", "%d", work_queue.n_grains());
//
//
//     RemoteWorkQueueBackend work_queue_backend;
//     work_queue_backend.Init(&work_queue);
//
//     RemoteBlockDeviceBackend q_points_backend;
//     q_points_backend.Init(q_points.cache());
//     RemoteBlockDeviceBackend q_nodes_backend;
//     q_nodes_backend.Init(q_nodes.cache());
//     RemoteBlockDeviceBackend r_points_backend;
//     r_points_backend.Init(r_points.cache());
//     RemoteBlockDeviceBackend r_nodes_backend;
//     r_nodes_backend.Init(r_nodes.cache());
//     RemoteBlockDeviceBackend q_results_backend;
//     q_results_backend.Init(q_results_.cache());
//
//     DataGetterBackend<typename GNP::Param> param_backend;
//     param_backend.Init(&param);
//
//     DataGetterBackend<MpiDualTreeConfig> config_backend;
//     config_backend.Init(&config);
//
//     RemoteObjectServer server;
//     server.Init();
//
//     server.Register(PARAM_CHANNEL, &param_backend);
//     server.Register(WORK_CHANNEL, &work_queue_backend);
//     server.Register(Q_POINTS_CHANNEL, &q_points_backend);;
//     server.Register(Q_NODES_CHANNEL, &q_nodes_backend);;
//     server.Register(R_POINTS_CHANNEL, &r_points_backend);;
//     server.Register(R_NODES_CHANNEL, &r_nodes_backend);;
//     server.Register(Q_RESULTS_CHANNEL, &q_results_backend);;
//     server.Register(CONFIG_CHANNEL, &config_backend);
//
//     fx_timer_start(module, "server");
//     server.Loop(n_workers_total);
//     fx_timer_stop(module, "server");
//
//     q_points_backend.Report(fx_submodule(module, NULL, "backends/q_points"));
//     q_nodes_backend.Report(fx_submodule(module, NULL, "backends/q_nodes"));
//     r_points_backend.Report(fx_submodule(module, NULL, "backends/r_points"));
//     r_nodes_backend.Report(fx_submodule(module, NULL, "backends/r_nodes"));
//     q_results_backend.Report(fx_submodule(module, NULL, "backends/q_results_"));
//   } else {
//     String my_fx_scope;
//
//     my_fx_scope.InitSprintf("rank%d", my_rank_);
//     fx_scope(my_fx_scope.c_str());
//
//     RemoteObjectServer::Connect(MASTER_RANK);
//
//     RemoteDataGetter<MpiDualTreeConfig> config_getter;
//     MpiDualTreeConfig config;
//     config_getter.Init(CONFIG_CHANNEL, MASTER_RANK);
//     config_getter.GetData(&config);
//
//     RemoteDataGetter<typename GNP::Param> param_getter;
//     param_getter.Init(PARAM_CHANNEL, MASTER_RANK);
//     param_getter.GetData(&param);
//
//     RemoteWorkQueue work_queue;
//     work_queue.Init(WORK_CHANNEL, MASTER_RANK);
//
//     RemoteBlockDevice q_points_device;
//     RemoteBlockDevice q_nodes_device;
//     RemoteBlockDevice r_points_device;
//     RemoteBlockDevice r_nodes_device;
//     RemoteBlockDevice q_results_device;
//
//     q_points_device.Init(Q_POINTS_CHANNEL, MASTER_RANK);
//     q_nodes_device.Init(Q_NODES_CHANNEL, MASTER_RANK);
//     r_points_device.Init(R_POINTS_CHANNEL, MASTER_RANK);
//     r_nodes_device.Init(R_NODES_CHANNEL, MASTER_RANK);
//     q_results_device.Init(Q_RESULTS_CHANNEL, MASTER_RANK);
//
//     SmallCache q_points_array;
//     SmallCache q_nodes_array;
//     SmallCache *r_points_array;
//     SmallCache r_nodes_array;
//     SmallCache q_results_array;
//
//     q_points_array.Init(&q_points_device,
//         new CacheArraySchema<typename GNP::Point>,
//         BlockDevice::READ);
//     q_nodes_array.Init(&q_nodes_device,
//         new CacheArraySchema<typename GNP::QNode>,
//         BlockDevice::READ);
//     if (!config.monochromatic) {
//       r_points_array = new SmallCache();
//       r_points_array->Init(&r_points_device,
//           new CacheArraySchema<typename GNP::Point>,
//           BlockDevice::READ);
//     } else {
//       r_points_array = &q_points_array;
//     }
//     r_nodes_array.Init(&r_nodes_device,
//         new CacheArraySchema<typename GNP::RNode>,
//         BlockDevice::READ);
//     q_results_array.Init(&q_results_device,
//         new CacheArraySchema<typename GNP::QResult>,
//         BlockDevice::MODIFY);
//
//     ThreadedDualTreeSolver<GNP, Solver> solver;
//     solver.InitSolve(
//         fx_submodule(module, "solver", "solver"), param,
//         config.n_threads, &work_queue,
//         &q_points_array, &q_nodes_array,
//         r_points_array, &r_nodes_array,
//         &q_results_array);
//
//     RemoteObjectServer::Disconnect(MASTER_RANK);
//
//     if (!config.monochromatic) {
//       delete r_points_array;
//     }
//   }
// }
// #endif

//  /**
//   * Loads vectors from a file into a point array.
//   *
//   * @param module the parameter "" is the file name, and timers will be stored
//   *        here
//   * @param default_point_inout a point object, in which everything EXCEPT
//   *        vec() is initialized
//   * @param cache_out this will be initialized to store all the loaded points
//   * @param n_block_vectors the number of vectors in the block
//   */
//  template<typename Point>
//  success_t nbr_utils::LoadVectors(fx_submodule *module,
//      Point* default_point_inout,
//      TempCacheArray<SpVectorPoint> *cache_out,
//      index_t n_block_vectors) {
//    Matrix matrix;
//    SpVectorPoint first_row;
//    success_t success;
//
//    fx_timer_start(module, "read_matrix");
//    success = data::Load(fx_param_str_req(module, ""), &matrix);
//    fx_timer_stop(module, "read_matrix");
//
//    default_point_inout->vec().Init(matrix.n_rows());
//    cache_out->Init(*default_point_inout, matrix.n_cols(), n_block_vectors);
//
//    fx_timer_start(module, "copy_into_cache");
//    for (index_t i = 0; i < matrix.n_cols(); i++) {
//      CacheWrite<SpVectorPoint> dest_vector(&cache_out, i);
//      dest_vector->vec().CopyValues(matrix.GetColumnPtr(i));
//    }
//    fx_timer_stop(module, "copy_into_cache");
//
//    return success;
//  }

//  template<typename Point, typename Node, typename Param>
//  success_t LoadKdTree(struct datanode *module,
//      Param* param,
//      TempCacheArray<Vector> *points_out,
//      TempCacheArray<Node> *nodes_out) {
//    index_t vectors_per_block = fx_param_int(
//        module, "vectors_per_block", 4096);
//    index_t nodes_per_block = fx_param_int(
//        module, "nodes_per_block", 2048);
//    success_t success;
//
//    success = nbr_utils::Load(module, points_out,
//        vectors_per_block);
//
//    if (success != SUCCESS_PASS) {
//      // WALDO: Do something better?
//      abort();
//    }
//
//    const Vector* first_point = points_out->StartRead(0);
//    param->AnalyzePoint(*first_point);
//    Node *example_node = new Node();
//    example_node->Init(first_point->length(), *param);
//    nodes_out->Init(*example_node, 0, nodes_per_block);
//    delete example_node;
//    points_out->StopRead(0);
//
//    KdTreeMidpointBuilder<Point, Node, Param>::Build(
//        module, param, points_out, nodes_out);
//
//    return SUCCESS_PASS;
//  }

/*
 TODO:

  - Array iterators (for when you know the code fits)
  - ArrayForall(visitor, range)
  - ArrayForall2(visitor, range)

  - TreeForall(preop, postop)
  - TreeForall2(preop, postop)

  - ParallelTree class

NECESSITIES
  - Mutable information can be easily keyed on the tree
    - 1: Index
    - 2: Explicit structure induction

XXX Assume no index structure:
XXX    BAD BAD BAD
XXX    Tree<NodeType> tree;
XXX
XXX    TreeNode<NodeType> handle(tree.Root());
XXX    TreeNode<NodeType> handle(handle.child(i));
XXX    handle.child(j);
XXX    tree.AllocChild(existing_node);

Assume index structure:
   - Looks like what we have now
   - Index structure is good

*/

  /*
  template<typename GNP, typename Solver>
  void MpiDualTreeMain(datanode *module, const char *gnp_name) {
    RemoteObjectServer server;

    server.Init();

    int q_points_channel = server.NewTag();
    int q_point_infos_channel = server.NewTag();
    int q_nodes_channel = server.NewTag();
    int r_points_channel = server.NewTag();
    int r_point_infos_channel = server.NewTag();
    int r_nodes_channel = server.NewTag();
    int q_results_channel = server.NewTag();

    remove point-info as a separate array

    if (server) {
      TempCacheArray<typename GNP::Point> q_points;
      TempCacheArray<typename GNP::QNode> q_nodes;
      TempCacheArray<typename GNP::Point> r_points;
      TempCacheArray<typename GNP::RNode> r_nodes;
      TempCacheArray<typename GNP::QResult> q_results_;

      export all of these arrays to the network

      typename GNP::Param param;

      param.Init(fx_submodule(module, gnp_name, gnp_name));

      nbr_utils::LoadKdTree(fx_submodule(module, "q", "q"),
          &param, &q_point_infos, &q_points, &q_nodes);
      nbr_utils::LoadKdTree(fx_submodule(module, "r", "r"),
          &param, &r_point_infos, &r_points, &r_nodes);

      typename GNP::QResult default_result;
      default_result.Init(param);
      q_results_.Init(default_result, q_points.end_index(),
          q_points.n_block_elems());

      server.Loop();
    } else if (worker) {
      MPI_Barrier(MPI_COMM_WORLD);

      initialize them all with default elements

      NetCacheArray<typename GNP::Point> q_points;
      q_points.Init(q_points_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::QNode> q_nodes;
      q_nodes.Init(q_nodes_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::Point> r_points;
      r_points.Init(r_points_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::RNode> r_nodes;
      r_nodes.Init(r_nodes_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::QResult> q_results_;
      q_results_.Init(q_results_channel, BlockDevice::CREATE);

      while (work) {
        Solver solver;
        solver.Init(fx_submodule(module, "solver", "solver"), param,
            &q_points, &q_point_infos, &q_nodes,
            &r_points, &r_point_infos, &r_nodes, &q_results_);
        solver.Begin();
      }
    }
  }
  */

};

#endif
