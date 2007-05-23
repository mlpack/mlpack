#ifndef NBR_UTILS_H
#define NBR_UTILS_H

#include "kdtree.h"
#include "work.h"

#include "par/thread.h"
#include "par/task.h"

namespace nbr_utils {

success_t Load(const char *fname, TempCacheArray<Vector> *cache_out,
    index_t vectors_per_block);

template<typename Node, typename Param>
success_t LoadKdTree(struct datanode *module,
    Param* param,
    TempCacheArray<Vector> *points_out,
    TempCacheArray<Node> *nodes_out) {
  index_t vectors_per_block = fx_param_int(
      module, "vectors_per_block", 256);
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
  nodes_out->Init(*example_node, 0, 256);
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

  ThreadedDualTreeSolver<GNP, Solver> solver;
  solver.InitSolve(
      fx_submodule(module, "solver", "solver"), param,
      n_threads, &work_queue,
      q_points.cache(), q_nodes.cache(),
      r_points.cache(), r_nodes.cache(),
      q_results.cache());
}

    /*
    RemoteWorkQueue queue;
    queue.Init(WORK_CHANNEL, MASTER_RANK);

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

    q_points_cache_.Init(&q_points_device,
        new CacheArrayBlockActionHandler<typename GNP::Point>,
        BlockDevice::READ);
    q_nodes_cache_.Init(&q_nodes_device,
        new CacheArrayBlockActionHandler<typename GNP::QNode>,
        BlockDevice::READ);
    r_points_cache_.Init(&r_points_device,
        new CacheArrayBlockActionHandler<typename GNP::Point>,
        BlockDevice::READ);
    r_nodes_cache_.Init(&r_nodes_device,
        new CacheArrayBlockActionHandler<typename GNP::RPoint>,
        BlockDevice::READ);
    q_results_cache_.Init(&q_results_device,
        new BlockActionHandler<typename GNP::QResult>, BlockDevice::MODIFY);
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
