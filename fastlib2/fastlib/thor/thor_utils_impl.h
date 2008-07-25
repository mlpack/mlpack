/* Template implementations for thor_utils.h */ 

template<typename GNP, typename Solver>
void thor::ThreadedDualTreeSolver<GNP, Solver>::Doit(
    index_t n_threads, int rank, SchedulerInterface *work_queue_in,
    const typename GNP::Param& param,
    DistributedCache *q_points_cache_in, DistributedCache *q_nodes_cache_in,
    DistributedCache *r_points_cache_in, DistributedCache *r_nodes_cache_in,
    DistributedCache *q_results_cache_in,
    DistributedCache *q_mutables_cache_in) { //
  param_ = &param;
  work_queue_ = work_queue_in;
  rank_ = rank;

  q_points_cache_ = q_points_cache_in;
  q_nodes_cache_ = q_nodes_cache_in;
  r_points_cache_ = r_points_cache_in;
  r_nodes_cache_ = r_nodes_cache_in;
  q_results_cache_ = q_results_cache_in;
  q_mutables_cache_ = q_mutables_cache_in; //

  global_result_.Init(*param_);
  stats_.Init();

  if (n_threads > 1) {
    ArrayList<Thread> threads;
    threads.Init(n_threads);

    for (index_t i = 0; i < n_threads; i++) {
      threads[i].Init(new WorkerTask(this));
      // Set these threads to low priority to make sure the network thread
      // gets full priority.
      threads[i].Start(Thread::LOW_PRIORITY);
    }

    for (index_t i = 0; i < n_threads; i++) {
      threads[i].WaitStop();
    }
  } else {
    (new WorkerTask(this))->Run();
  }
}

template<typename GNP, typename Solver>
void thor::ThreadedDualTreeSolver<GNP, Solver>::ThreadBody_() {
  while (1) {
    ArrayList<SchedulerInterface::Grain> work;

    mutex_.Lock();
    work_queue_->GetWork(rank_, &work);
    mutex_.Unlock();

    if (work.size() == 0) {
      break;
    }

    for (index_t i = 0; i < work.size(); i++) {
      Solver solver;

      solver.Doit(*param_,
          work[i].node_index, work[i].node_end_index,
          q_points_cache_, q_nodes_cache_,
          r_points_cache_, r_nodes_cache_,
	  q_results_cache_, q_mutables_cache_); //

      mutex_.Lock();
      stats_.Add(solver.stats());
      global_result_.Accumulate(*param_, solver.global_result());
      mutex_.Unlock();
    }
  }
}

//--------------------------------------------------------------------------

template<typename Point, typename Param>
index_t thor::ReadPointsMaster(
    const Param& param, int points_channel,
    const char *filename, int block_size_kb, double megs,
    DistributedCache *points_cache) {
  Point default_point;
  TextLineReader reader;
  DatasetInfo schema;
  int dimension;
  Vector vector;
  index_t n_points;

  if (FAILED(reader.Open(filename))) {
    FATAL("Could not open data file '%s'", filename);
  }

  schema.InitFromFile(&reader, "data");
  dimension = schema.n_features();

  default_point.Init(param, schema);

  CacheArray<Point> points_array;
  points_array.InitCreate(points_channel,
      CacheArray<Point>::ConvertBlockSize(default_point, block_size_kb),
      default_point, megs, points_cache);

  vector.Init(schema.n_features());

  n_points = 0;

  for (;;) {
    bool is_done;
    success_t rv = schema.ReadPoint(&reader, vector.ptr(), &is_done);

    if (unlikely(FAILED(rv))) {
      FATAL("Data file has problems");
    } else if (is_done) {
      break;
    } else {
      int owner = rpc::rank();
      // Distributing the points seems to make it a ton slower, disk seems
      // to be better for some odd reason.
      //int owner = (n_points / points_array.n_block_elems()) % rpc::n_peers();
      index_t i = points_array.AllocD(owner, 1);
      CacheWrite<Point> point(&points_array, i);
      point->Set(param, i, vector);
      n_points++;
      if (n_points % 1000000 == 0) {
        fprintf(stderr, "... read %"LI"d points...\n", n_points);
      }
    }
  }
  fprintf(stderr, "... done reading, %"LI"d points.\n", n_points);

  return n_points;
}

template<typename Point, typename Param>
index_t thor::ReadPoints(const Param& param, int points_channel,
    int extra_channel, datanode *module,
    DistributedCache *points_cache) {
  double megs = fx_param_int(module, "megs", 2000);
  Broadcaster<index_t> broadcaster;

  fx_timer_start(module, "read");

  if (rpc::is_root()) {
    const char *filename = fx_param_str_req(module, "");
    int block_size_kb = fx_param_int(module, "block_size_kb", 256);
    index_t n_points = ReadPointsMaster<Point>(param, points_channel,
        filename, block_size_kb, megs, points_cache);
    broadcaster.SetData(n_points);
  } else {
    CacheArray<Point>::CreateCacheWorker(points_channel, megs, points_cache);
  }

  broadcaster.Doit(extra_channel);
  points_cache->StartSync();
  points_cache->WaitSync();

  fx_timer_stop(module, "read");

  return broadcaster.get();
}

template<typename GNP, typename SerialSolver, typename QTree, typename RTree>
void thor::RpcDualTree(datanode *module, int base_channel,
    const typename GNP::Param& param, QTree *q, RTree *r,
    DistributedCache *q_results, DistributedCache *q_mutables, //
    typename GNP::GlobalResult *global_result_out) {
  int n_threads = fx_param_int(module, "n_threads", 2);
  RemoteSchedulerBackend *work_backend = NULL;
  SchedulerInterface *work_queue;
  datanode *io_module = fx_submodule(module, "io");
  datanode *work_module = fx_submodule(module, "scheduler");

  if (rpc::is_root()) {
    // Make a static work queue
    CentroidScheduler<typename GNP::QNode> *actual_work_queue =
        new CentroidScheduler<typename GNP::QNode>;
    CacheArray<typename GNP::QNode> q_nodes_array;
    q_nodes_array.Init(&q->nodes(), BlockDevice::M_READ);
    actual_work_queue->Init(&q_nodes_array,
        q->decomp().root(), n_threads, module);
    work_queue = new LockedScheduler(actual_work_queue);

    work_backend = new RemoteSchedulerBackend();
    work_backend->Init(work_queue);
    rpc::Register(base_channel + 0, work_backend);
  } else {
    RemoteScheduler *remote_work_queue = new RemoteScheduler();
    remote_work_queue->Init(base_channel + 0, 0);
    work_queue = remote_work_queue;
  }

  rpc::Barrier(base_channel + 1);

  fx_timer_start(module, "gnp");
  ThreadedDualTreeSolver<GNP, SerialSolver> solver;
  solver.Doit(n_threads, rpc::rank(), work_queue, param,
      &q->points(), &q->nodes(), &r->points(), &r->nodes(),
      q_results, q_mutables); //
  rpc::Barrier(base_channel + 2);
  fx_timer_stop(module, "gnp");

  fx_timer_start(module, "write_results");
  q->points().StartSync();
  q->nodes().StartSync();
  if (!mem::PtrsEqual(q, r)) {
    r->points().StartSync();
    r->nodes().StartSync();
  }
  q_results->StartSync();
  q_mutables->StartSync(); //
  q->points().WaitSync(fx_submodule(io_module, "q_points"));
  q->nodes().WaitSync(fx_submodule(io_module, "q_nodes"));
  if (!mem::PtrsEqual(q, r)) {
    r->points().WaitSync(fx_submodule(io_module, "r_points"));
    r->nodes().WaitSync(fx_submodule(io_module, "r_nodes"));
  }
  q_results->WaitSync(fx_submodule(io_module, "q_results"));
  q_mutables->WaitSync(fx_submodule(io_module, "q_mutables")); //
  fx_timer_stop(module, "write_results");

#ifdef DEBUG
  DualTreeRecursionStats stats = solver.stats();
  rpc::Reduce(base_channel + 5, DualTreeRecursionStats::Reductor(), &stats);
  stats.Report(fx_submodule(module, "recursion"));
#endif

  GlobalResultReductor<GNP> global_result_reductor;
  typename GNP::GlobalResult my_global_result = solver.global_result();
  global_result_reductor.Init(&param);
  rpc::Reduce(base_channel + 6, global_result_reductor, &my_global_result);

  work_queue->Report(work_module);
  my_global_result.Report(param,
      fx_submodule(module, "global_result"));

  if (rpc::is_root()) {
    rpc::Unregister(base_channel + 0);
    delete work_backend;
  }

  if (global_result_out) {
    global_result_out->Copy(my_global_result);
  }

  delete work_queue;
}

template<typename GNP, typename Solver>
void thor::MonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  const int DATA_CHANNEL = 110;
  const int Q_RESULTS_CHANNEL = 120;
  const int GNP_CHANNEL = 200;
  double results_megs = fx_param_double(module, "results/megs", 1000);
  double mutables_megs = fx_param_double(module, "mutables/megs", 1000); //
  DistributedCache *points_cache;
  index_t n_points;
  ThorTree<typename GNP::Param, typename GNP::QPoint, typename GNP::QNode> tree;
  DistributedCache q_results;
  DistributedCache q_mutables; //
  typename GNP::Param param;

  rpc::Init();

  //fx_submodule(module, "io"); // influnce output order

  param.Init(fx_submodule(module, gnp_name));

  fx_timer_start(module, "read");
  points_cache = new DistributedCache();
  n_points = ReadPoints<typename GNP::QPoint>(
      param, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
      fx_submodule(module, "data"), points_cache);
  fx_timer_stop(module, "read");

  typename GNP::QPoint default_point;
  CacheArray<typename GNP::QPoint>::GetDefaultElement(
      points_cache, &default_point);
  param.SetDimensions(default_point.vec().length(), n_points);

  fx_timer_start(module, "tree");
  CreateKdTree<typename GNP::QPoint, typename GNP::QNode>(
      param, DATA_CHANNEL + 2, DATA_CHANNEL + 3, // DATA_CHANNEL + 4,
      fx_submodule(module, "tree"), n_points, points_cache, &tree);
  fx_timer_stop(module, "tree");

  typename GNP::QResult default_result;
  default_result.Init(param);
  tree.CreateResultCache(Q_RESULTS_CHANNEL, default_result,
        results_megs, &q_results);

  // Alert!  Temporary hack; does not support rbfs
  typename Solver::QMutables default_mutable; // 
  default_mutable.summary_result.Init(param); //
  default_mutable.postponed.Init(param); //
  tree.CreateMutableCache(Q_RESULTS_CHANNEL + 1, default_mutable, //
        mutables_megs, &q_mutables); //

  typename GNP::GlobalResult global_result;
  RpcDualTree<GNP, Solver>(
      fx_submodule(module, "gnp"), GNP_CHANNEL, param,
      &tree, &tree, &q_results, &q_mutables, &global_result);

  rpc::Done();
}
