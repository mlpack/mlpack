class MultiTreeQueryPostponed {
    
 public:
  
  DRange negative_potential_bound;
  
  double negative_potential_e;
  
  DRange positive_potential_bound;
  
  double positive_potential_e;
  
  double n_pruned;
  
  double used_error;
  
  void ApplyDelta(const MultiTreeDelta &delta_in, index_t node_index) {
    
    negative_potential_bound +=
      delta_in.negative_potential_bound[node_index];
    negative_potential_e += delta_in.negative_potential_e[node_index];
    positive_potential_bound +=
      delta_in.positive_potential_bound[node_index];
    positive_potential_e += delta_in.positive_potential_e[node_index];
    n_pruned += delta_in.n_pruned[node_index];
    used_error += delta_in.used_error[node_index];
  }
  
  void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in) {
    negative_potential_bound += postponed_in.negative_potential_bound;
    negative_potential_e += postponed_in.negative_potential_e;
    positive_potential_bound += postponed_in.positive_potential_bound;
    positive_potential_e += postponed_in.positive_potential_e;
    n_pruned += postponed_in.n_pruned;
    used_error += postponed_in.used_error;
  }
  
  void SetZero() {
    negative_potential_bound.Init(0, 0);
    negative_potential_e = 0;
    positive_potential_bound.Init(0, 0);
    positive_potential_e = 0;
    n_pruned = 0;
    used_error = 0;
  }
  
  void Init() {
    
    // Initializes to zeros...
    SetZero();
  }
};
