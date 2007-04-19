
 - delta can take up lots of room
   - this can be very useful (for summing moments)
   - this can be very bad (for summing multi-bandwidth stuff)

 - what does delta still do
   - allows undoes
   - allows reconstruction of parts
   - most importantly: can be passed along and applied to root nodes
     - can be queued up and applied and undone?  how does this trickle down?
   - concept: "approximate pi"
     - actually this can be mu
     - allow mus to be composed horizontally (and thus downwards)
     - concept: mu is:
       - (a) a sum of pis, and thus, a sum of horizontally
         separate/vertically complete mus
       - (b) a product of horizontally complete/vertically partitioned mus
     - propagate downward changes in mu, can undo other mus
       - mus can be vertically merged even if they are horizontally incomplete
       - mus have a circle-plus AND circle-times operator!!!
       - composing mus horizontally fundamentally requires loss of info,
       but that is okay -- mu is "best-effort"
     - allow rhos to be composed with mu
       - unfortunately this creates a class interdependency :-(
       - solve using templates
       - but another problem
         - does rho represent incomplete or complete results?
         - rho might want mu in order to do per-point termination pruning
         - SOLUTION: rho.SetFakeMuStuff(mu)

 - step 1 (intrinsic prune check)
   - ConsiderPair(param, q_node, r_node, &pi, &gamma, &mu)
   - if (intrinsic prune)
     - apply to pi
       - do NOT apply to mu, queue up pi first
     - apply to gamma
     - record null delta?
     else
     - apply to mu
     - apply estimate to gamma
     - record change to gamma and mu (delta)
       - recording change to gamma must be completely accurate
       - does change to mu have to even exist?
         - yes, for undo step
 - step 2 (extrinsic prune check)
   - my mu must be valid incoming!!
   - question: what happens when extrinsic prunes happen
     - series moment expansion: apply to pi an expansion
       - since i applied a pi, my delta is removed from next round (or undone)
     - lots of examples
       - nothing to apply to pi
       - keep my delta in gamma forever
   - hypothesis
     - never remove my delta from gamma
       - tweaks to gamma not allowed?
       - two implementations
         - recreate gamma: i could add a more refined delta into the new gamma
         - change gamma: i could refine the gamma
         - either: i can create a new delta with better values
         - or if we stretch it: delta can be a "refinement"
     - mu is short-lived, what happens with mu is up to traversal pattern
     - changes to mu and rho must be stored in pi
   - verdict
     - tweaks to gamma NOT allowed
     - changes to mu and rho must be stored in pi

objects
 - postponed pruning information (pi)
   - apply to mu given qnode
   - apply to rho given qpoint
 - exploration
   - preprocessing
     - undo relevant changes to gamma
     - in some recursive patterns this may not be necessary
   - stage 1: check for intrinsic prunes:
     - if intrinsic prune
       - apply to gamma
       - apply to pi
     - if no intrinsic prune
       - apply to mu
       - record change to gamma
   - intermediate
     - make sure we have a clean, crisp mu and gamma
   - stage 2: check for termination prunes
     - this might be folded into step 3
   - stage 3: check for non-intrinsic prunes
     - if extrinsic prune
       - update pi
       - calling code will assume to keep changes to gamma etc???
     - else
       - 
     - use our new mu and gamma
     - update pi, gamma
     - if i decided to prune, 
 - new rules
   - mu q-join function must be idempotent
     - mu r-join function might have to be non-idempotent if we want to
     allow undoing on mu
   - gamma qr-join function may not be idempotent
   - mu join (pi1 join pi1) != (mu join pi1) join pi2

class Queue {
 public:
  struct Entry {
    Entry() {
      DEBUG_POISON_PTR(r_node);
    }
    
    RNode *r_node;
    Delta delta;
  };
 
 private: 
  ArrayList<Entry> list_;
  Delta delta_forward_;
  Delta delta_pruned_;
  const Param *param;
  QNode *q_node_in;
  
 public:
  void Init(QNode* q_node_in, const Param& param) {
    param_ = &param;
    q_node_ = q_node_in;
    
    list_.Init();
    delta_pruned_.Init(*param_);
  }
  
  void Init(QNode* q_node_in, const Queue& parent) {
    param_ = parent.param;
    q_node_ = q_node_in;
    
    list_.Init();
    delta_pruned_.Init(*param_);
    delta_pruned_.Apply(*param_, parent.delta_pruned_);
    
    delta_forward_.Init();
  }
  
  void Add(RNode *r_node) {
    Entry *entry = list_.AddBack();
    bool try_explore = entry->delta.InitCompute(param, *q_node, *r_node);
    
    delta_forward_.Apply(entry->delta);
    
    if (try_explore) {
      entry->r_node = r_node;
    } else {
      delta_pruned_.Apply(*param_, entry->delta);
      list_.PopBack();
    }
  }
  
  /**
   * Do the postprocessing reverse step.
   */
  void TransformDeltas() {
    const Delta* cur_delta = &delta_pruned_;
    
    for (index_t i = list_.size() - 2; i != 0; i--) {
      Entry *entry = &list_[i];
      entry->delta.Apply(*param_, *cur_delta);
      cur_delta = &entry->delta;
    }
  }
  
  index_t size() const {
    return list_.size();
  }
  
  RNode* rnode(index_t i) const {
    return ;
  }
  
  /** returns the sum of deltas including the specified up to the end */
  const Delta& delta_backward(int i) const {
    return list_[i].delta;
  }
  
  const Delta& delta_forward() const {
    return delta_forward_;
  }
  
  const Delta& delta_pruned() const {
    return delta_pruned_;
  }
  
  const ArrayList<Entry>& list() const {
    return list_;
  }
};


void SplitQ(QNode *q_node, const Queue& list_old) {
  if (q_node->is_leaf()) {
    SplitR(q_node, list_old);
    return;
  }

  Queue list_new[cardinality];
  
  /* TODO: termination prunes can be checked here */
  
  for (int c = 0; c < cardinality; c++) {
    list_new[c].Init(q_node->child(i), list_old);
  }
  
  QMassResult my_mass_result;
  GlobalResult my_global_result;
  
  // We haven't done any exhaustive comparisons, we start with an empty.
  my_mass_result.Init(param_);
  my_mass_result.Apply(param_, list_old.delta_forward(), *q_node_);
  my_global_result.Init(param_);
  my_global_result.Accumulate(param_, global_result_siblings);
  my_global_result.Apply(param_, list_old.delta_forward());
  
  for (index_t i = 0; i < list_old.size(); i++) {
    RNode *r_node = list_old.rnode(i);
    Delta left_delta;
    
    if (likely(Algorithm::MustExplore(param_, *q_node, *r_node,
         list_old.delta(i), my_mass_result, my_global_result))) {
      for (int c = 0; c < cardinality; c++) {
        list_new[c].Add(r_node);
      }
    }
  }
  
  /* recurse over query children */
  
  for (int c = 0; c < cardinality; c++) {
    list_new[c].Finish(param_);
    SplitR(q_node->child(c), list_new[c]);
  }
}

void SplitR(QNode *q_node, const ArrayList<Entry>& list_old) {
  Queue list_new;
  
  list_new.Init(q_node, list_old);
  
  for (index_t i = 0; i < list_old.size(); i++) {
    const Entry *entry_old = &list_old[i];
    RNode *r_node = entry_old->r_node;
    
    if (entry_old->r_node->is_leaf()) {
      for (int c = 0; c < cardinality; c++) {
        queue.Add(node->child(i));
      }
    } else {
      queue.Add(r_node);
    }
  }
}

