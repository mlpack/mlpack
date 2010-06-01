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

  bool initialized;

  OT_DEF_BASIC(MultiTreeDelta) {
    OT_MY_OBJECT(negative_potential_bound);
    OT_MY_OBJECT(negative_potential_e);
    OT_MY_OBJECT(positive_potential_bound);
    OT_MY_OBJECT(positive_potential_e);
    OT_MY_OBJECT(n_pruned);
    OT_MY_OBJECT(used_error);
    OT_MY_OBJECT(initialized);
  }
  
 public:
  
  template<typename TGlobal>
  void RefineBounds(TGlobal &globals, double negative_potential_avg, 
		    double negative_standard_deviation,
		    double positive_potential_avg,
		    double positive_standard_deviation) {

    double positive_error, negative_error;

    for(index_t i = 0; i < positive_potential_e.length(); i++) {
      if(initialized) {	
	positive_potential_e[i] = positive_potential_avg * n_pruned[i];
	negative_potential_e[i] = negative_potential_avg * n_pruned[i];
	positive_error = globals.z_score * n_pruned[i] *
	  positive_standard_deviation;
	negative_error = globals.z_score * n_pruned[i] *
	  negative_standard_deviation;
	
	used_error[i] = sqrt(math::Sqr(positive_error) +
			     math::Sqr(negative_error));

	positive_potential_bound[i].lo = 
	  std::max(positive_potential_bound[i].lo,
		   positive_potential_e[i] - positive_error);
	positive_potential_bound[i].hi = 
	  std::min(positive_potential_bound[i].hi,
		   positive_potential_e[i] + positive_error);
	negative_potential_bound[i].lo = 
	  std::max(negative_potential_bound[i].lo,
		   negative_potential_e[i] - negative_error);
	negative_potential_bound[i].hi = 
	  std::min(negative_potential_bound[i].hi,
		   negative_potential_e[i] + negative_error);
      }
      else {
	positive_potential_e[i] = positive_potential_avg * n_pruned[i];
	negative_potential_e[i] = negative_potential_avg * n_pruned[i];
	positive_error = globals.z_score * n_pruned[i] *
	  positive_standard_deviation;
	negative_error = globals.z_score * n_pruned[i] *
	  negative_standard_deviation;

	used_error[i] = sqrt(math::Sqr(positive_error) +
			     math::Sqr(negative_error));

	positive_potential_bound[i].lo = 
	  std::max(0.0, positive_potential_e[i] - positive_error);
	positive_potential_bound[i].hi = positive_potential_e[i] + 
	  positive_error;
	negative_potential_bound[i].lo = negative_potential_e[i] - 
	  negative_error;
	negative_potential_bound[i].hi = 
	  std::min(0.0, negative_potential_e[i] + negative_error);
      }
    } // end of looping over each node entity...
  }

  template<typename TGlobal, typename Tree>
  bool ComputeFiniteDifference(TGlobal &globals,
			       ArrayList<Tree *> &nodes,
			       const Vector &total_n_minus_one_tuples) {
    
    // Compute the pairwise distances among the nodes, and min and max
    // contributions...
    bool flag = 
      globals.kernel_aux.ComputeFiniteDifference(globals, nodes, *this);

    if(flag) {
      initialized = true;
    }
    return flag;
  }
  
  void SetZero() {
    for(index_t i = 0; i < TKernel::order; i++) {
      negative_potential_bound[i].Init(0, 0);
      positive_potential_bound[i].Init(0, 0);
    }
    negative_potential_e.SetZero();
    positive_potential_e.SetZero();
    used_error.SetZero();

    // WARNING: I don't set n_pruned to zero because I assume it will
    // be re-used after the initialization Init function.
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

    initialized = false;
  }    
};
