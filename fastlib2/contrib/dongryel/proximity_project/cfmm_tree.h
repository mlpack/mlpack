#ifndef CFMM_TREE_H
#define CFMM_TREE_H

namespace proximity {

  // Forward declaration...
  template<typename TStatistics> class CFmmTree;

  template<typename TStatistics>
  class CFmmWellSeparatedTree {

   public:

    /** @brief The beginning index of the point for each dataset.
     */
    ArrayList<index_t> begin_;

    /** @brief The number of points contained in this node for each
     *         dataset.
     */
    ArrayList<index_t> count_;

    /** @brief The well-separated index for each dataset, i.e. the
     *         maximum WS index for the points for each group.
     */
    Vector well_separated_indices_;
    
    /** @brief The total number of points for all the datasets.
     */
    index_t total_count_;

    /** @brief The pointers to the children nodes.
     */
    ArrayList<CFmmTree<TStatistics> *> children_;
    
   public:

    CFmmTreeWellSeparatedTree() {
    }

    ~CFmmTreeWellSeparatedTree() {
      if(children_.size() > 0) {
	for(index_t i = 0; i < children_.size(); i++) {
	  delete children_[i];
	}
      }
    }

    /** @brief Gets the index of the begin point of this subset.
     */
    index_t begin(index_t particle_set_number) const {
      return begin_[particle_set_number];
    }
    
    /** @brief Gets the index one beyond the last index in the series.
     */
    index_t end(index_t particle_set_number) const {
      return begin_[particle_set_number] + count_[particle_set_number];
    }

    /** @brief Tests whether it is a leaf node or not.
     */
    bool is_leaf() const {
      return children_.size() == 0;      
    }

    void Init(index_t number_of_particle_sets, index_t dimension) {
      begin_.Init(number_of_particle_sets);
      count_.Init(number_of_particle_sets);
      total_count_ = 0;
      node_index_ = 0;
      children_.Init();
    }

    void Init(index_t particle_set_number, index_t begin_in, 
	      index_t count_in) {

      begin_[particle_set_number] = begin_in;
      count_[particle_set_number] = count_in;
      total_count_ += count_in;
    }
    
    CFmmTree *AllocateNewChild(index_t number_of_particle_sets,
			       index_t dimension, unsigned int node_index_in) {
      
      CFmmTree *new_node = new CFmmTree();
      *(children_.PushBackRaw()) = new_node;

      new_node->Init(number_of_particle_sets, dimension);
      new_node->node_index_ = node_index_in;

      return new_node;
    }

    void Print() const {
      if (!is_leaf()) {
	printf("internal node: %d points total on level %d\n", total_count_,
	       level_);
	for(index_t i = 0; i < begin_.size(); i++) {
	  printf("   set %d: %d to %d: %d points total\n", i, 
		 begin_[i], begin_[i] + count_[i] - 1, count_[i]);	  
	}
	for(index_t c = 0; c < children_.size(); c++) {
	  children_[c]->Print();
	}
      }
      else {
	printf("leaf node: %d points total on level %d\n", total_count_,
	       level_);
	for(index_t i = 0; i < begin_.size(); i++) {
	  printf("   set %d: %d to %d: %d points total\n", i, 
		 begin_[i], begin_[i] + count_[i] - 1, count_[i]);	  
	}
      }
    }    
  };

  template<typename TStatistics>
  class CFmmTree {
    
   public:

    /** @brief The bounding box type is the rectangle bound.
     */
    typedef DHrectBound<2> Bound;

    /** @brief The type of the dataset.
     */
    typedef Matrix Dataset;

    /** @brief The type of statistics stored in each node.
     */
    typedef TStatistic Statistic;
    
    /** @brief The bounding box.
     */
    Bound bound_;

    /** @brief The beginning index of the point for each dataset.
     */
    ArrayList<index_t> begin_;

    /** @brief The number of points contained in this node for each
     *         dataset.
     */
    ArrayList<index_t> count_;

    /** @brief The well-separated index for each dataset, i.e. the
     *         maximum WS index for the points for each group.
     */
    Vector well_separated_indices_;
    
    /** @brief The total number of points for all the datasets.
     */
    index_t total_count_;

    /** @brief The current level of the tree.
     */
    index_t level_;

    /** @brief The global index of the node.
     */
    unsigned int node_index_;

    /** @brief The stored statistics for this node.
     */
    Statistic stat_;
    
    /** @brief The divided group based on the well-separated
     *         indices. This generalizes the 2-way branchings used in
     *         the CFMM paper.
     */
    ArrayList<CFmmWellSeparatedTree *> partitions_based_on_ws_indices_;

   public:

    CFmmTree() {
    }

    ~CFmmTree() {
      if(partitions_based_on_ws_indices_.size() > 0) {

	for(index_t i = 0; i < partitions_based_on_ws_indices_.size(); i++) {
	  delete partitions_based_on_ws_indices_[i];
	}
      }
    }

    const Statistics& stat() const {
      return stat_;
    }

    Statistics& stat() {
      return stat_;
    }

    /** @brief Tests whether the current node is a leaf node
     *         (childless).
     */
    bool is_leaf() const {
      if(partitions_based_on_ws_indices_.size() > 0) {
	bool flag = true;
	for(index_t i = 0; i < partitions_based_on_ws_indices_.size() && flag;
	    i++) {	  
	  flag = flag && (partitions_based_on_ws_indices_[i]->is_leaf());
	}
	return flag;
      }
      else {
	return true;
      }
    }

    void Init(index_t number_of_particle_sets, index_t dimension) {
      begin_.Init(number_of_particle_sets);
      count_.Init(number_of_particle_sets);
      total_count_ = 0;
      node_index_ = 0;
      children_.Init();
    }

    void Init(index_t particle_set_number, index_t begin_in, 
	      index_t count_in) {

      begin_[particle_set_number] = begin_in;
      count_[particle_set_number] = count_in;
      total_count_ += count_in;
    }

    double side_length() const {
      const DRange &range = bound_.get(0);
      return range.hi - range.lo;
    }

    const Bound& bound() const {
      return bound_;
    }
    
    Bound& bound() {
      return bound_;
    }

    GenHypercubeTree *get_child(int index) const {
      return children_[index];
    }

    void set_level(index_t level) {
      level_ = level;
    }

    CFmmWellSeparatedTree *AllocateNewPartition() {

      CFmmWellSeparatedTree *new_partition = new CFmmWellSeparatedTree();
      *(partitions_based_on_ws_indices_.PushBackRaw()) = new_partition;
      
      new_partition->Init();
      return new_partition;
    }

    void Print() const {
      if (!is_leaf()) {
	printf("internal node: %d points total on level %d\n", total_count_,
	       level_);
	printf("  bound:\n");
	for(index_t i = 0; i < bound_.dim(); i++) {
	  printf("%g %g\n", bound_.get(i).lo, bound_.get(i).hi);
	}
	for(index_t i = 0; i < begin_.size(); i++) {
	  printf("   set %d: %d to %d: %d points total\n", i, 
		 begin_[i], begin_[i] + count_[i] - 1, count_[i]);	  
	}
	for(index_t c = 0; c < partitions_based_on_ws_indices_.size(); c++) {
	  partitions_based_on_ws_indices_[c]->Print();
	}
      }
      else {
	printf("leaf node: %d points total on level %d\n", total_count_,
	       level_);
	printf("  bound:\n");
	for(index_t i = 0; i < bound_.dim(); i++) {
	  printf("%g %g\n", bound_.get(i).lo, bound_.get(i).hi);
	}
	for(index_t i = 0; i < begin_.size(); i++) {
	  printf("   set %d: %d to %d: %d points total\n", i, 
		 begin_[i], begin_[i] + count_[i] - 1, count_[i]);	  
	}
      }
    }

  };
};

#endif
