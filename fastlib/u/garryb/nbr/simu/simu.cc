
class Simulator {
 private:
  typedef int Timestamp;
  
  struct WorkItem {
    uint n_points;
    Timestamp expected_time;
  };
  
  struct Machine {
    ArrayList<WorkItem> local_queue;
    MinHeap<Timestamp, int> next_completion_;
  };
  
  ArrayList<SimMachine> machines_;
  MinHeap<Timestamp, int> next_completion_;
  
 public:
  
  void SimuStep() {
    Timestamp cur_time;
    Timestamp next_time;
    int completion_machine;
    int completion_thread;
    Machine *machine;
    
    cur_time = next_completion_.top_key();
    completion_machine = next_completion_.Pop();
    machine = &machines_[completion_machine];
    completion_thread = machine->next_completion_.Pop();
    
    select a new work item
    
    next_time = cur_time + new_work_time;
    machine->next_completion_.Put(next_time, thread_id);
    next_completion_.Put(next_time, completion_machine);
  }
};

