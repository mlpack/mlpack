
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

template<
    class TAlgorithm,
    class TParam,
    class TPoint,
    class TDataset,
    class TQInfo,
    class TRInfo,
    class TBound,
    class TQStat,
    class TRStat,
    class TQResult,
    class TPairVisitor,
    class TQMassResult,
    class TQPostponedResult,
    class TDelta,
    class TGlobalResult
    >
class DualTreeGNP {
 public:
  typedef TAlgorithm Algorithm;
  typedef TParam Param;
  typedef TPoint Point;
  typedef TDataset Dataset;
  typedef TQInfo QInfo;
  typedef TRInfo RInfo;
  typedef TBound Bound;
  typedef TQStat QStat;
  typedef TRStat RStat;
  typedef TQResult QResult;
  typedef TPairVisitor PairVisitor;
  typedef TQMassResult QMassResult;
  typedef TQPostponedResult QPostponedResult;
  typedef TDelta Delta;
  typedef TGlobalResult GlobalResult;
  
  // our use of tree-nodes is kind of broken
  typedef BinarySpaceNode<Bound, Dataset, QStat> QNode;
  typedef BinarySpaceNode<Bound, Dataset, RStat> RNode;
};

template<class GNP>
class RecursiveBreadthFirstDualTreeRunner {
 public:
  typedef typename GNP::Algorithm Algorithm;
  typedef typename GNP::Param Param;
  // TODO: only Vector is supported for point
  typedef typename GNP::Point Point;
  typedef typename GNP::Dataset Dataset;
  typedef typename GNP::QInfo QInfo;
  typedef typename GNP::RInfo RInfo;
  typedef typename GNP::Bound Bound;
  typedef typename GNP::QStat QStat;
  typedef typename GNP::RStat RStat;
  typedef typename GNP::QResult QResult;
  typedef typename GNP::PairVisitor PairVisitor;
  typedef typename GNP::QMassResult QMassResult;
  typedef typename GNP::QPostponedResult QPostponedResult;
  typedef typename GNP::Delta Delta;
  typedef typename GNP::GlobalResult GlobalResult;
  
  typedef typename GNP::QNode QNode;
  typedef typename GNP::RNode RNode;
  
 private:
  Dataset q_matrix_;
  Dataset r_matrix_;
  Param param_;
  ArrayList<QInfo> q_info_;
  ArrayList<RInfo> r_info_;
  QNode *q_root_;
  RNode *r_root_;
  
 public:
  void Init(struct datanode *module,
      const Dataset& q_matrix_in, const Dataset& r_matrix_in) {
    q_matrix_.Init(q_matrix_in);
    r_matrix_.Init(r_matrix_in);
    
    param_.Init(fx_submodule(module, "algorithm", "algorithm"));
    
    q_info_.Init(q_matrix_in.n_cols());
    r_info_.Init(q_matrix_in.n_cols());
    
    q_root_ = MakeTree(q_matrix);
    r_root_ = MakeTree(r_matrix);
  }
  
  ArrayList<QInfo>& q_info() {
    return q_info;
  }
  
  ArrayList<RInfo>& r_info() {
    return r_info;
  }

 private:
 
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
    const Param *param_;
    QNode *q_node_;
    ArrayList<Entry> list_;
    MassResult q_mass_result_;
    PostponedResult q_postponed_;
    GlobalResult *global_result_;
    
   public:
    void Init(QNode* q_node_in, const Param* param,
        GlobalResult* global_result_in) {
      param_ = param;
      q_node_ = q_node_in;
      
      list_.Init();
      
      q_mass_result_.Init(*param_);
      
      q_postponed_.Init(*param_);
      
      global_result_ = global_result_in;
    }
    
    void Init(QNode* q_node_in, const Queue& parent) {
      param_ = parent.param;
      q_node_ = q_node_in;
      
      list_.Init();
      
      q_mass_result_.Init(*param_);
      
      q_postponed_.Init(*param_);
      q_postponed_.ApplyPostponed(*param_, *parent.q_postponed_);
      
      global_result_ = parent.global_result_;
    }
    
    void Add(RNode *r_node) {
      Entry *entry = list_.AddBack();
      bool try_explore = Algorithm::ConsiderPairIntrinsic(
          *param_, *q_node_, *r_node,
          &entry->delta, &q_mass_result_, global_result_, &q_postponed_);
      
      if (try_explore) {
        entry->r_node = r_node;
      } else {
        list_.PopBack();
      }
    }
    
    void Finish() {
      q_mass_result_.ApplyPostponed(*param_, q_postponed_, *q_node_);
    }
    
    
    index_t size() const {
      return list_.size();
    }
    
    RNode* rnode(index_t i) const {
      return ;
    }
    
    /** returns the sum of deltas including the specified up to the end */
    const Delta& delta(int i) const {
      return list_[i].delta;
    }

    const MassResult& q_mass_result() const { return q_mass_result_; }
    
    const PostponedResult& q_postponed() const { return q_postponed_; }
    
    PostponedResult& q_postponed() { return q_postponed_; }
  };


  void SplitQ(QNode *q_node, const Queue& list_old) {
    if (q_node->is_leaf()) {
      DoQLeafStuffWALDO_TODO(q_node, list_old);
      return;
    }
    
    if (Algorithm::ConsiderQueryTermination(&param_, *q_node,
        list_old.q_mass_result(), *global_result_, &list_old.q_postponed())) {
      RecursivelyApplyPostponed(q_node, list_old.q_postponed());
      return;
    }

    Queue list_new[cardinality];
    
    /* TODO: termination prunes can be checked here */
    
    for (int c = 0; c < cardinality; c++) {
      list_new[c].Init(q_node->child(i), list_old);
    }
    
    QPostponedResults postponed;
    
    postponed.Init(*param_);
    // We haven't done any exhaustive comparisons, we start with an empty.
    
    for (index_t i = 0; i < list_old.size(); i++) {
      RNode *r_node = list_old.rnode(i);
      const Delta* delta = &list_old.delta(i);
      
      if (likely(Algorithm::ConsiderPairExtrinsic(param_, *q_node, *r_node,
          *delta, list_old.q_mass_result(), *global_result_, &postponed))) {
        global_result_.UndoDelta(param_, *delta);
        for (int c_q = 0; c_q < cardinality; c_q++) {
          global_result_.ApplyDelta(param_, *delta);
          list_new[c_q].Add(r_node);
        }
      }
    }
    
    /* recurse over query children */
    
    for (int c = 0; c < cardinality; c++) {
      list_new[c].q_postponed().ApplyPostponed(*param_, postponed);
      list_new[c].Finish(param_);
      SplitR(q_node->child(c), list_new[c]);
    }
  }

  void SplitR(QNode *q_node, const ArrayList<Entry>& list_old) {
    Queue list_new;
    
    if (Algorithm::ConsiderQueryTermination(&param_, *q_node,
        list_old.q_mass_result(), *global_result_, &list_old.q_postponed())) {
      RecursivelyApplyPostponed(q_node, list_old.q_postponed());
      return;
    }
    
    list_new.Init(q_node, list_old);
    
    for (index_t i = 0; i < list_old.size(); i++) {
      RNode *r_node = list_old.rnode(i);
      const Delta* delta = &list_old.delta(i);
      
      if (entry_old->r_node->is_leaf()) {
        // TODO: We can collapse the mu's for these together
        list_new.Add(r_node);
      } else {
        if (likely(Algorithm::ConsiderPairExtrinsic(param_, *q_node, *r_node,
            *delta, list_old.q_mass_result(), *global_result_,
            &list_new.q_postponed()))) {
          global_result_.UndoDelta(*delta);
          for (int c = 0; c < cardinality; c++) {
            global_result_.ApplyDelta(*delta);
            list_new.Add(node->child(i));
          }
        }
      }
    }
  }
};

