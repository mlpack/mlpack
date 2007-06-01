
#include "fastlib/fastlib_int.h"

typedef double Timestamp;

struct WorkItem {
  Vector center;
  index_t n_points;
  Timestamp actual_duration;
  bool assigned;
  
  OT_DEF(WorkItem) {
    OT_MY_OBJECT(center);
    OT_MY_OBJECT(n_points);
    OT_MY_OBJECT(actual_duration);
    OT_MY_OBJECT(assigned);
  }
};

struct Machine {
  ArrayList<index_t> assigned;
  
  OT_DEF(Machine) {
    OT_MY_OBJECT(assigned);
  }
};

class Selector {
 public:
  virtual ~Selector() {}
  virtual int Select(const ArrayList<index_t>& assigned,
      const ArrayList<WorkItem>& pool) = 0;
};

class CentroidSelector : public Selector {
 public:
  virtual ~CentroidSelector() {}
  
  int Select(const ArrayList<index_t>& assigned,
      const ArrayList<WorkItem>& pool) {
    // First find the centroid of work completed so far
    Vector centroid;
    index_t n_points_assigned;

    centroid.Init(pool[0].center.length());
    centroid.SetZero();
    n_points_assigned = 0;

    for (index_t i = 0; i < assigned.size(); i++) {
      index_t n_points = pool[assigned[i]].n_points;
      la::AddExpert(n_points, pool[assigned[i]].center, &centroid);
      n_points_assigned += n_points;
    }

    la::Scale(1.0 / n_points_assigned, &centroid);

    // Now find the closest non-completed work item.
    index_t choice_i = -1;
    double choice_dsq = DBL_MAX;

    for (index_t i = 0; i < pool.size(); i++) {
      if (!pool[i].assigned) {
        double dsq = la::DistanceSqEuclidean(
            pool[i].center, centroid);
        if (dsq < choice_dsq) {
          choice_i = i;
          choice_dsq = dsq;
        }
      }
    }

    return choice_i;
  }
};

class Simulator {
 private:
  Selector *selector_;
  ArrayList<Machine> machines_;
  MinHeap<Timestamp, int> next_completion_;
  ArrayList<WorkItem> work_items_;
  Timestamp cur_time_;
  double linear_time_;

 public:
  void Init(Selector *selector_in,
      struct datanode *params) {
    index_t n_machines = fx_param_int(params, "n_machines", 16);
    index_t n_threads = fx_param_int(params, "n_threads", 4);
    index_t n_work_items = fx_param_int(params, "n_work_items",
        n_threads * n_machines * 4);
    index_t dim = fx_param_int(params, "dim", 3);
    double d_min = fx_param_double(params, "d_min", 0.5);
    double d_max = fx_param_double(params, "d_max", 1.5);

    linear_time_ = 0;
    selector_ = selector_in;
    work_items_.Init();
    
    for (index_t i = 0; i < n_work_items; i++) {
      WorkItem *item = work_items_.AddBack();
      item->center.Init(dim);
      for (index_t d = 0; d < dim; d++) {
        item->center[d] = math::Random(0, 1);
      }
      item->n_points = 1000; // perfectly balanced
      item->actual_duration = item->n_points * math::Random(d_min, d_max);
      linear_time_ += item->actual_duration;
      item->assigned = false;
    }
    linear_time_ /= (n_machines * n_threads);
    
    cur_time_ = 0;

    next_completion_.Init();
    machines_.Init(n_machines);
    for (index_t i = 0; i < n_machines; i++) {
      machines_[i].assigned.Init();

      index_t work_i;
      
      while (1) {
        work_i = math::RandInt(work_items_.size());
        if (!work_items_[work_i].assigned) {
          break;
        }
      }

      Assign(i, work_i);

      for (index_t thread = 1; thread < n_threads; thread++) {
        Assign(i);
      }
    }
  }

  void Assign(index_t machine_i, index_t work_i) {
    work_items_[work_i].assigned = 1;
    *machines_[machine_i].assigned.AddBack() = work_i;
    Timestamp done_time = work_items_[work_i].actual_duration + cur_time_;
    next_completion_.Put(done_time, machine_i);
  }

  void Assign(index_t machine_i) {
    index_t work_i = selector_->Select(
        machines_[machine_i].assigned, work_items_);
    if (work_i >= 0) {
      Assign(machine_i, work_i);
    }
  }

  void SimuStep() {
    int completion_machine;

    cur_time_ = next_completion_.top_key();
    completion_machine = next_completion_.Pop();

    Assign(completion_machine);
  }

  double Simulate() {
    while (!next_completion_.is_empty()) {
      SimuStep();
    }
    return cur_time_;
  }
  
  double ratio() const {
    return cur_time_ / linear_time_;
  }
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  CentroidSelector selector;
  int n_iter = 20;
  double total_ratios = 0;
  
  srand(time(NULL));

  for (int i = 0; i < n_iter; i++) {
    String name;
    name.InitSprintf("run_%d", i);
    fx_timer_start(fx_root, name.c_str());
    Simulator simulator;
    simulator.Init(&selector, fx_root);
    simulator.Simulate();
    total_ratios += simulator.ratio();
    DEBUG_ASSERT(simulator.ratio() >= 1);
    fx_timer_stop(fx_root, name.c_str());
  }

  fx_format_result(fx_root, "average_ratio", "%f", total_ratios / n_iter);

  fx_done();
}

