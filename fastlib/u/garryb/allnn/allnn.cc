#include "fastlib/fastlib_int.h"
#ifdef USE_MPI
#include "par/mpigrain.h"
#endif

class AllNN {
 public:
  struct Result {
    double dist;
    index_t index;
  };
 
 protected:
  datanode *module_;
  Matrix q_matrix_;
  Matrix r_matrix_;
  ArrayList<Result> results_; // TODO: Two separate arrays?  Good for smaller datasets
  bool bichromatic_;
  
 public:
  AllNN() {}
  ~AllNN() {}
  
 public:
  void Init(datanode *module_in,
      const Matrix& q_matrix_in, const Matrix& r_matrix_in) {
    module_ = module_in;
    q_matrix_.Alias(q_matrix_in);
    r_matrix_.Alias(r_matrix_in);
    bichromatic_ = (q_matrix_.ptr() != r_matrix_.ptr());
    results_.Init(q_matrix_.n_cols());
    for (index_t i = 0; i < results_.size(); i++) {
      results_[i].dist = DBL_MAX;
      	results_[i].index = BIG_BAD_NUMBER;
    }
  }
  
  const ArrayList<Result>& results() const {
    return results_;
  }
  
  ArrayList<Result>& results() {
    return results_;
  }
  
  void ReportDistanceSq(const char *name) {
    double sumsdistance = 0;
    
    for (index_t i = 0; i < results_.size(); i++) {
      sumsdistance += results_[i].dist;
    }
    
    fx_format_result(module_, "msdistance", "%e", sumsdistance / results_.size());
  }
};

class AllNNNaive : public AllNN {
 public:
  AllNNNaive() {}
  ~AllNNNaive() {}
  
  void Init(datanode *module_in,
      const Matrix& q_matrix_in, const Matrix& r_matrix_in) {
    AllNN::Init(module_in, q_matrix_in, r_matrix_in);
  }
  
  void Compute();
};

void AllNNNaive::Compute() {
  fx_timer_start(module_, "allnn");
  
  for (index_t q_i = 0; q_i < q_matrix_.n_cols(); q_i++) {
    const double *q_col = q_matrix_.GetColumnPtr(q_i);
    double best_dist = DBL_MAX;
    index_t best_index = 0;
    
    for (index_t r_i = 0; r_i < r_matrix_.n_cols(); r_i++) {
      const double *r_col = r_matrix_.GetColumnPtr(r_i);
      
      double dist = la::DistanceSqEuclidean(
          r_matrix_.n_rows(), q_col, r_col);
      
      if (unlikely(dist < best_dist) && (q_col != r_col)) {
        best_dist = dist;
        best_index = r_i;
      }
    }
    
    results_[q_i].dist = best_dist;
    results_[q_i].index = best_index;
    
  }

  fx_timer_stop(module_, "allnn");
}

class AllNNDualTree : public AllNN {
 public:
  class Stat {
   public:
    double dist_upper;
    
    void Init() {
      dist_upper = DBL_MAX;
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      Init();
    }
    
    void Init(const Matrix& dataset, index_t start, index_t count,
        const Stat& left_stat, const Stat& right_stat) {
      Init();
    }
  };

  typedef BinarySpaceTree<DHrectBound, Matrix, Stat> Tree;

 private:
  /**
   * A serializable work grain.
   */
  struct Grain {
   public:
    int num;
    index_t q_begin;
    index_t q_count;
    index_t r_begin;
    index_t r_count;
    double closest_offer;
   
   public:
    Grain(int num_in,
        index_t q_begin_in, index_t q_count_in,
        index_t r_begin_in, index_t r_count_in,
        double closest_offer_in) {
      num = num_in;
      q_begin = q_begin_in;
      q_count = q_count_in;
      r_begin = r_begin_in;
      r_count = r_count_in;
      closest_offer = closest_offer_in;
    }
    /**
     * For MPI construction
     * TODO: Change this if we end up using serialization
     */
    Grain() {}
    
    void Run(AllNNDualTree *problem) {
#if defined(DEBUG) || defined(USE_MPI)
      fprintf(stderr, "[*] Grain %d starting\n", num);
#endif
      Tree *q = problem->q_root_->FindByBeginCount(q_begin, q_count);
      const Tree *r = problem->r_root_->FindByBeginCount(r_begin, r_count);
      DEBUG_ASSERT(problem->q_root_ != NULL);
      DEBUG_ERR_MSG_IF(q == NULL, "%u, %u", q_begin, q_count);
      DEBUG_ASSERT(r != NULL);
      problem->DualAllNN(q, r, closest_offer);
    }
  };
  
  void BaseCase(Tree *q, const Tree *r, double closest_offer);
  
  void DualAllNN(Tree *q, const Tree *r, double closest_offer);
  
  void SplitReference(Tree *q, const Tree *r1, const Tree *r2) {
    double m1 = q->bound().MidDistanceSqToBound(r1->bound());
    double m2 = q->bound().MidDistanceSqToBound(r2->bound());
    double d1 = q->bound().MinDistanceSqToBound(r1->bound());
    double d2 = q->bound().MinDistanceSqToBound(r2->bound());
    
    if (m1 < m2) {
      DualAllNN(q, r1, d1);
      DualAllNN(q, r2, d2);
    } else {
      DualAllNN(q, r2, d2);
      DualAllNN(q, r1, d1);
    }
  }
  
  void CreateGrains(Tree *q, const Tree *r, index_t grain_size_max,
      GrainQueue<Grain> *queue);
  
  void Granulate(int num_threads, index_t num_grains);
  
 private:
  Tree *q_root_;
  const Tree *r_root_;
  uint64 n_naive_;
  uint64 n_pre_naive_;
  uint64 n_recurse_;
  
 public:
  AllNNDualTree() {}
  
  ~AllNNDualTree() {}
  
  void Init(datanode *module_in,
      const Matrix& q_matrix_in, const Matrix& r_matrix_in,
      Tree *q, const Tree *r) {
    AllNN::Init(module_in, q_matrix_in, r_matrix_in);
    q_root_ = q;
    r_root_ = r;
    n_naive_ = 0;
    n_recurse_ = 0;
    n_pre_naive_ = 0;
  }
  
  void Compute(int num_threads, index_t num_grains) {
    fx_timer_start(module_, "allnn");
    Granulate(num_threads, num_grains);
    fx_timer_stop(module_, "allnn");
    
#ifdef DEBUG
    fx_format_result(module_, "naive_ratio", "%f",
        (1.0 * n_naive_ / q_root_->count() / r_root_->count()));
    fx_format_result(module_, "naive_per_query", "%f",
        (1.0 * n_naive_ / q_root_->count()));
    fx_format_result(module_, "pre_naive_ratio", "%f",
        (1.0 * n_pre_naive_ / q_root_->count() / r_root_->count()));
    fx_format_result(module_, "pre_naive_per_query", "%f",
        (1.0 * n_pre_naive_ / q_root_->count()));
    fx_format_result(module_, "recurse_ratio", "%f",
        (1.0 * n_recurse_ / q_root_->count() / r_root_->count()));
    fx_format_result(module_, "recurse_per_query", "%f",
        (1.0 * n_recurse_ / q_root_->count()));
#endif

    ReportDistanceSq("allnn_dual");
  }
};

void AllNNDualTree::CreateGrains(Tree *q, const Tree *r,
    index_t grain_size_max, GrainQueue<Grain> *queue) {
  // We divide the query tree up into pieces, where no piece is larger than
  // the expected grain size.
  
  // We will likely create more grains than we intended to, but that is okay.
  
  if (q->is_leaf() || q->count() <= grain_size_max) {
    int num = queue->size();
    queue->Put(q->count(),
        new Grain(num, q->begin(), q->count(), r->begin(), r->count(), 0));
  } else {
    CreateGrains(q->left(), r, grain_size_max, queue);
    CreateGrains(q->right(), r, grain_size_max, queue);
  }
}

void AllNNDualTree::Granulate(int num_threads, index_t num_grains) {
#ifndef USE_MPI
  ThreadedGrainRunner<Grain, AllNNDualTree*> runner;
  GrainQueue<Grain> queue;
  queue.Init();
  CreateGrains(q_root_, r_root_,
      (q_root_->count() + num_grains - 1) / num_grains, &queue);
  fx_format_result(module_, "n_grains_actual", "%d", int(queue.size()));
  runner.Init(&queue, this);
  runner.RunThreads(num_threads);
#else
  MPIGrainRunner<Grain, AllNNDualTree*> runner;
  GrainQueue<Grain> *queue = NULL;
  runner.Init("dual", 102, this, 0);
  if (runner.dispatcher()) {
    queue = new GrainQueue<Grain>;
    queue->Init();
    CreateGrains(q_root_, r_root_,
        (q_root_->count() + num_grains - 1) / num_grains, queue);
    runner.dispatcher()->set_queue(queue);
    fx_format_result(module_, "n_grains_actual", "%d", int(queue->size()));
  }
  runner.RunThreads(num_threads);
  if (queue) {
    delete queue;
  }
#endif
}

void AllNNDualTree::BaseCase(Tree *q, const Tree *r, double closest_offer) {
  double dist_upper = 0;
  index_t r_end = r->end();
  
  #ifdef DEBUG
  index_t n_naive_l = 0;
  #endif
  
  // TODO: Does it help to prune individual q or r points?
  for (index_t q_i = q->begin(); q_i < q->end(); q_i++) {
    const double *q_col = q_matrix_.GetColumnPtr(q_i);
    double q_best_dist = results_[q_i].dist;
    
    if (closest_offer <= q_best_dist
        && r->bound().MinDistanceSqToPoint(q_col) <= q_best_dist) {
      for (index_t r_i = r->begin(); r_i < r_end; r_i++) {
        const double *r_col = r_matrix_.GetColumnPtr(r_i);
        double dist = la::DistanceSqEuclidean(
            r_matrix_.n_rows(), q_col, r_col);
        
        if (unlikely(dist < q_best_dist) && (q_col != r_col)) {
          results_[q_i].dist = q_best_dist = dist;
          results_[q_i].index = r_i;
        }
      }
      
      DEBUG_ONLY(n_naive_l += r->count());
    }
    
    if (unlikely(q_best_dist > dist_upper)) {
      dist_upper = q_best_dist;
    }
  }
  
  DEBUG_ONLY(n_naive_ += n_naive_l);
  DEBUG_ONLY(n_pre_naive_ += q->count() * r->count());
  q->stat().dist_upper = dist_upper;
}

void AllNNDualTree::DualAllNN(Tree *q, const Tree *r, double closest_offer) {
  DEBUG_ONLY(n_recurse_++);
  if (closest_offer > q->stat().dist_upper) {
    /* pruned */
  } else if (q->is_leaf() && r->is_leaf()) {
    BaseCase(q, r, closest_offer);
  } else if ((q->count() >= r->count() && !q->is_leaf()) || r->is_leaf()) {
    DualAllNN(q->left(), r,
        q->left()->bound().MinDistanceSqToBound(r->bound()));
    DualAllNN(q->right(), r,
        q->right()->bound().MinDistanceSqToBound(r->bound()));
    q->stat().dist_upper = max(
        q->left()->stat().dist_upper, q->right()->stat().dist_upper);
  } else {
    SplitReference(q, r->left(), r->right());
  }
}

int main(int argc, char *argv[]) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
  fx_init(argc, argv);
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    fx_scope(tsprintf("rank%d", rank));
  }
#endif
  
  bool do_naive = fx_param_bool(NULL, "do_naive", 0);
  bool do_dual = fx_param_bool(NULL, "do_dual", 1);
  int num_threads = fx_param_int(NULL, "num_threads", 1);
  index_t num_grains = fx_param_int(NULL, "num_grains", 1);
  
  Matrix q_matrix;
  Matrix r_matrix;
  
  AllNNDualTree::Tree *q_tree;
  AllNNDualTree::Tree *r_tree;
  
  ArrayList<index_t> q_old_from_new; // permutation
  
  tree::LoadKdTree(fx_submodule(NULL, "q", "read_q"), &q_matrix, &q_tree, &q_old_from_new);
  
  if (fx_param_exists(NULL, "r")) {
    tree::LoadKdTree(fx_submodule(NULL, "r", "read_r"), &r_matrix, &r_tree, NULL);
  } else {
    r_matrix.Alias(q_matrix);
    r_tree = q_tree;
  }
  
  if (do_dual) {
    AllNNDualTree dual;
    dual.Init(fx_submodule(NULL, "dual", "dual"),
        q_matrix, r_matrix, q_tree, r_tree);
    dual.Compute(num_threads, num_grains);
  }
  
  if (do_naive) {
    AllNNNaive naive;
    naive.Init(fx_submodule(NULL, "naive", "naive"), q_matrix, r_matrix);
    naive.Compute();
  }
  
  delete q_tree;
  if (q_tree != r_tree) {
    delete r_tree;
  }
  
  fx_done();
#ifdef USE_MPI
  MPI_Finalize();
#endif
}
