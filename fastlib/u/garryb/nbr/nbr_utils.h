#ifndef NBR_UTILS_H
#define NBR_UTILS_H

#include "kdtree.h"
#include "work.h"
#include "rpc.h"

#include "par/thread.h"
#include "par/task.h"

namespace nbr_utils {

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

success_t Load(const char *fname, TempCacheArray<Vector> *cache_out,
    index_t vectors_per_block);

template<typename Node, typename Param>
success_t LoadKdTree(struct datanode *module,
    Param* param,
    TempCacheArray<Vector> *points_out,
    TempCacheArray<Node> *nodes_out) {
  index_t vectors_per_block = fx_param_int(
      module, "vectors_per_block", 4096);
  index_t nodes_per_block = fx_param_int(
      module, "nodes_per_block", 2048);
  success_t success;

  fx_timer_start(module, "read");
  success = nbr_utils::Load(fx_param_str_req(module, ""), points_out,
      vectors_per_block);
  fx_timer_stop(module, "read");

  if (success != SUCCESS_PASS) {
    // WALDO: Do something better?
    abort();
  }

  const Vector* first_point = points_out->StartRead(0);
  param->AnalyzePoint(*first_point);
  Node *example_node = new Node();
  example_node->Init(first_point->length(), *param);
  nodes_out->Init(*example_node, 0, nodes_per_block);
  delete example_node;
  points_out->StopRead(0);

  fx_timer_start(module, "tree");
  KdTreeMidpointBuilder<Node, Param> builder;
  builder.InitBuild(module, param, points_out, nodes_out);
  fx_timer_stop(module, "tree");

  return SUCCESS_PASS;
}

template<typename GNP, typename Solver>
void SerialDualTreeMain(datanode *module, const char *gnp_name) {
  typename GNP::Param param;

  param.Init(fx_submodule(module, gnp_name, gnp_name));

  TempCacheArray<typename GNP::Point> q_points;
  TempCacheArray<typename GNP::QNode> q_nodes;
  TempCacheArray<typename GNP::Point> r_points;
  TempCacheArray<typename GNP::RNode> r_nodes;
  TempCacheArray<typename GNP::QResult> q_results;

  nbr_utils::LoadKdTree(fx_submodule(module, "q", "q"),
      &param, &q_points, &q_nodes);
  nbr_utils::LoadKdTree(fx_submodule(module, "r", "r"),
      &param, &r_points, &r_nodes);

  typename GNP::QResult default_result;
  default_result.Init(param);
  q_results.Init(default_result, q_points.end_index(),
      q_points.n_block_elems());

  Solver solver;
  solver.InitSolve(fx_submodule(module, "solver", "solver"), param, 0,
      q_points.cache(), q_nodes.cache(),
      r_points.cache(), r_nodes.cache(), q_results.cache());
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

          String name;
          name.InitSprintf("grain_%d", work[i]);
          base_->mutex_.Lock();
          struct datanode *submodule = fx_submodule(base_->module_,
              name.c_str(), name.c_str());
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
  void InitSolve(
      datanode *module,
      const typename GNP::Param& param,
      int n_threads,
      WorkQueueInterface *work_queue,
      SmallCache *q_points_cache_in,
      SmallCache *q_nodes_cache_in,
      SmallCache *r_points_cache_in,
      SmallCache *r_nodes_cache_in,
      SmallCache *q_results_cache_in
      ) {
    module_ = module;
    param_ = &param;
    work_queue_ = work_queue;

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
  }
};

template<typename GNP, typename Solver>
void ThreadedDualTreeMain(datanode *module, const char *gnp_name) {
  typename GNP::Param param;

  param.Init(fx_submodule(module, gnp_name, gnp_name));

  TempCacheArray<typename GNP::Point> q_points;
  TempCacheArray<typename GNP::QNode> q_nodes;
  TempCacheArray<typename GNP::Point> r_points;
  TempCacheArray<typename GNP::RNode> r_nodes;
  TempCacheArray<typename GNP::QResult> q_results;

  nbr_utils::LoadKdTree(fx_submodule(module, "q", "q"),
      &param, &q_points, &q_nodes);
  nbr_utils::LoadKdTree(fx_submodule(module, "r", "r"),
      &param, &r_points, &r_nodes);

  typename GNP::QResult default_result;
  default_result.Init(param);
  q_results.Init(default_result, q_points.end_index(),
      q_points.n_block_elems());

  index_t n_threads = fx_param_int(module, "n_threads", 1);
  index_t n_grains = fx_param_int(module, "n_grains",
      n_threads == 1 ? 1 : (n_threads * 3));
  SimpleWorkQueue<typename GNP::QNode> work_queue;
  work_queue.Init(&q_nodes, n_grains);
  fx_format_result(module, "n_grains_actual", "%"LI"d", work_queue.n_grains());

  ThreadedDualTreeSolver<GNP, Solver> solver;
  solver.InitSolve(
      fx_submodule(module, "solver", "solver"), param,
      n_threads, &work_queue,
      q_points.cache(), q_nodes.cache(),
      r_points.cache(), r_nodes.cache(),
      q_results.cache());
}

/*
- add the code for loading the tree
- standardize channel numbers
- write code that detects and runs the server
*/

#ifdef USE_MPI

struct MpiDualTreeConfig {
  int n_threads;
  bool monochromatic;
  
  void Copy(const MpiDualTreeConfig& other) {
    *this = other;
  }

  OT_DEF(MpiDualTreeConfig) {
    OT_MY_OBJECT(n_threads);
  }
};

template<typename GNP, typename Solver>
void MpiDualTreeMain(datanode *module, const char *gnp_name) {
  const int MASTER_RANK = 0;
  const int PARAM_CHANNEL = 2;
  const int WORK_CHANNEL = 3;
  const int Q_POINTS_CHANNEL = 4;
  const int Q_NODES_CHANNEL = 5;
  const int R_POINTS_CHANNEL = 6;
  const int R_NODES_CHANNEL = 7;
  const int Q_RESULTS_CHANNEL = 8;
  const int CONFIG_CHANNEL = 9;
  typename GNP::Param param;
  int my_rank;
  int n_machines;
  int n_workers_total;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_machines);

  n_workers_total = n_machines - 1;

  if (my_rank == MASTER_RANK) {
    param.Init(fx_submodule(module, gnp_name, gnp_name));

    TempCacheArray<typename GNP::Point> q_points;
    TempCacheArray<typename GNP::QNode> q_nodes;
    TempCacheArray<typename GNP::Point> r_points;
    TempCacheArray<typename GNP::RNode> r_nodes;
    TempCacheArray<typename GNP::QResult> q_results;

    nbr_utils::LoadKdTree(fx_submodule(module, "q", "q"),
        &param, &q_points, &q_nodes);
    nbr_utils::LoadKdTree(fx_submodule(module, "r", "r"),
        &param, &r_points, &r_nodes);

    MpiDualTreeConfig config;
    config.n_threads = fx_param_int(module, "n_threads", 1);
    config.monochromatic = fx_param_bool(module, "monochromatic", 1);
    
    typename GNP::QResult default_result;
    default_result.Init(param);
    q_results.Init(default_result, q_points.end_index(),
        q_points.n_block_elems());

    SimpleWorkQueue<typename GNP::QNode> work_queue;
    int n_grains = fx_param_int(module, "n_grains",
        config.n_threads * n_workers_total * 3);
    work_queue.Init(&q_nodes, n_grains);
    fx_format_result(module, "n_grains_actual", "%d", work_queue.n_grains());
    

    RemoteWorkQueueBackend work_queue_backend;
    work_queue_backend.Init(&work_queue);

    RemoteBlockDeviceBackend q_points_backend;
    q_points_backend.Init(q_points.cache());
    RemoteBlockDeviceBackend q_nodes_backend;
    q_nodes_backend.Init(q_nodes.cache());
    RemoteBlockDeviceBackend r_points_backend;
    r_points_backend.Init(r_points.cache());
    RemoteBlockDeviceBackend r_nodes_backend;
    r_nodes_backend.Init(r_nodes.cache());
    RemoteBlockDeviceBackend q_results_backend;
    q_results_backend.Init(q_results.cache());

    DataGetterBackend<typename GNP::Param> param_backend;
    param_backend.Init(&param);

    DataGetterBackend<MpiDualTreeConfig> config_backend;
    config_backend.Init(&config);

    RemoteObjectServer server;
    server.Init();

    server.Register(PARAM_CHANNEL, &param_backend);
    server.Register(WORK_CHANNEL, &work_queue_backend);
    server.Register(Q_POINTS_CHANNEL, &q_points_backend);;
    server.Register(Q_NODES_CHANNEL, &q_nodes_backend);;
    server.Register(R_POINTS_CHANNEL, &r_points_backend);;
    server.Register(R_NODES_CHANNEL, &r_nodes_backend);;
    server.Register(Q_RESULTS_CHANNEL, &q_results_backend);;
    server.Register(CONFIG_CHANNEL, &config_backend);

    fx_timer_start(module, "server");
    server.Loop(n_workers_total);
    fx_timer_stop(module, "server");
    
    q_points_backend.Report(fx_submodule(module, NULL, "backends/q_points"));
    q_nodes_backend.Report(fx_submodule(module, NULL, "backends/q_nodes"));
    r_points_backend.Report(fx_submodule(module, NULL, "backends/r_points"));
    r_nodes_backend.Report(fx_submodule(module, NULL, "backends/r_nodes"));
    q_results_backend.Report(fx_submodule(module, NULL, "backends/q_results"));
  } else {
    String my_fx_scope;

    my_fx_scope.InitSprintf("rank%d", my_rank);
    fx_scope(my_fx_scope.c_str());
    
    RemoteObjectServer::Connect(MASTER_RANK);

    RemoteDataGetter<MpiDualTreeConfig> config_getter;
    MpiDualTreeConfig config;
    config_getter.Init(CONFIG_CHANNEL, MASTER_RANK);
    config_getter.GetData(&config);
    
    RemoteDataGetter<typename GNP::Param> param_getter;
    param_getter.Init(PARAM_CHANNEL, MASTER_RANK);
    param_getter.GetData(&param);

    RemoteWorkQueue work_queue;
    work_queue.Init(WORK_CHANNEL, MASTER_RANK);

    RemoteBlockDevice q_points_device;
    RemoteBlockDevice q_nodes_device;
    RemoteBlockDevice r_points_device;
    RemoteBlockDevice r_nodes_device;
    RemoteBlockDevice q_results_device;

    q_points_device.Init(Q_POINTS_CHANNEL, MASTER_RANK);
    q_nodes_device.Init(Q_NODES_CHANNEL, MASTER_RANK);
    r_points_device.Init(R_POINTS_CHANNEL, MASTER_RANK);
    r_nodes_device.Init(R_NODES_CHANNEL, MASTER_RANK);
    q_results_device.Init(Q_RESULTS_CHANNEL, MASTER_RANK);

    SmallCache q_points_cache;
    SmallCache q_nodes_cache;
    SmallCache *r_points_cache;
    SmallCache r_nodes_cache;
    SmallCache q_results_cache;

    q_points_cache.Init(&q_points_device,
        new CacheArrayBlockActionHandler<typename GNP::Point>,
        BlockDevice::READ);
    q_nodes_cache.Init(&q_nodes_device,
        new CacheArrayBlockActionHandler<typename GNP::QNode>,
        BlockDevice::READ);
    if (!config.monochromatic) {
      r_points_cache = new SmallCache();
      r_points_cache->Init(&r_points_device,
          new CacheArrayBlockActionHandler<typename GNP::Point>,
          BlockDevice::READ);
    } else {
      r_points_cache = &q_points_cache;
    }
    r_nodes_cache.Init(&r_nodes_device,
        new CacheArrayBlockActionHandler<typename GNP::RNode>,
        BlockDevice::READ);
    q_results_cache.Init(&q_results_device,
        new CacheArrayBlockActionHandler<typename GNP::QResult>,
        BlockDevice::MODIFY);

    ThreadedDualTreeSolver<GNP, Solver> solver;
    solver.InitSolve(
        fx_submodule(module, "solver", "solver"), param,
        config.n_threads, &work_queue,
        &q_points_cache, &q_nodes_cache,
        r_points_cache, &r_nodes_cache,
        &q_results_cache);

    RemoteObjectServer::Disconnect(MASTER_RANK);
    
    if (!config.monochromatic) {
      delete r_points_cache;
    }
  }
}
#endif

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
      TempCacheArray<typename GNP::QResult> q_results;

      export all of these arrays to the network

      typename GNP::Param param;

      param.Init(fx_submodule(module, gnp_name, gnp_name));

      nbr_utils::LoadKdTree(fx_submodule(module, "q", "q"),
          &param, &q_point_infos, &q_points, &q_nodes);
      nbr_utils::LoadKdTree(fx_submodule(module, "r", "r"),
          &param, &r_point_infos, &r_points, &r_nodes);

      typename GNP::QResult default_result;
      default_result.Init(param);
      q_results.Init(default_result, q_points.end_index(),
          q_points.n_block_elems());

      server.Loop();
    } else if (worker) {
      MPI_Barrier();

      initialize them all with default elements

      NetCacheArray<typename GNP::Point> q_points;
      q_points.Init(q_points_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::QNode> q_nodes;
      q_nodes.Init(q_nodes_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::Point> r_points;
      r_points.Init(r_points_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::RNode> r_nodes;
      r_nodes.Init(r_nodes_channel, BlockDevice::READ);
      NetCacheArray<typename GNP::QResult> q_results;
      q_results.Init(q_results_channel, BlockDevice::CREATE);

      while (work) {
        Solver solver;
        solver.Init(fx_submodule(module, "solver", "solver"), param,
            &q_points, &q_point_infos, &q_nodes,
            &r_points, &r_point_infos, &r_nodes, &q_results);
        solver.Begin();
      }
    }
  }
  */

};

#endif
