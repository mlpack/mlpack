class MultiTreeDelta {
  
 public:
  
  /** @brief Stores the negative lower and the negative upper
   *         contribution of the $i$-th node in consideration among
   *         the $n$ tuples.
   */
  ArrayList<DRange> negative_potential_bound;
  
  /** @brief The estimated negative component.
   */
  Vector negative_potential_e;
  
  /** @brief Stores the positive lower and the positive upper
   *         contribution of the $i$-th node in consideration among
   *         the $n$ tuples.
   */    
  ArrayList<DRange> positive_potential_bound;
  
  /** @brief The estimated positive component.
   */
  Vector positive_potential_e;
  
  Vector n_pruned;
  
  Vector used_error;

  OT_DEF_BASIC(MultiTreeDelta) {
    OT_MY_OBJECT(negative_potential_bound);
    OT_MY_OBJECT(negative_potential_e);
    OT_MY_OBJECT(positive_potential_bound);
    OT_MY_OBJECT(positive_potential_e);
    OT_MY_OBJECT(n_pruned);
    OT_MY_OBJECT(used_error);
  }
  
 public:
  
  template<typename TGlobal, typename Tree>
  bool ComputeFiniteDifference(TGlobal &globals,
			       ArrayList<Tree *> &nodes,
			       const Vector &total_n_minus_one_tuples) {
    
    // Compute the pairwise distances among the nodes.
    globals.kernel_aux.ComputePairwiseDistances(globals, nodes);

    
    return true;
  }
  
  void SetZero() {
    for(index_t i = 0; i < TKernel::order; i++) {
      negative_potential_bound[i].Init(0, 0);
      positive_potential_bound[i].Init(0, 0);
    }
    negative_potential_e.SetZero();
    positive_potential_e.SetZero();
    n_pruned.SetZero();
    used_error.SetZero();
  }
  
  void Init(const Vector &total_n_minus_one_tuples) {
    
    negative_potential_bound.Init(TKernel::order);
    negative_potential_e.Init(TKernel::order);
    positive_potential_bound.Init(TKernel::order);
    positive_potential_e.Init(TKernel::order);
    n_pruned.Init(TKernel::order);
    used_error.Init(TKernel::order);
    
    // Copy the number of pruned tuples...
    n_pruned.CopyValues(total_n_minus_one_tuples);
    
    // Initializes to zeros...
    SetZero();
  }    
};
