#ifndef ALLKNN_H
#define ALLKNN_H

class AllKNN {

  class RefStat {
  private:
    ArrayList <double> distance_to_qnode_;

  public:

    RefStat() {
      distance_to_qnode_.Init(0);
    }

    ~RefStat() {
    }

    double distance_to_qnode() {
      return distance_to_qnode_.back();
    }

    void set_distance_to_qnode(double new_dist) {
      distance_to_qnode_.PushBackCopy(new_dist);
      return;
    }

    void pop_last_distance() {
      DEBUG_ASSERT(distance_to_qnode_.size() > 0);
      distance_to_qnode_.PopBack();
      return;
    }

    //    void Init() {
    //  distance_to_qnode_ = DBL_MAX;
    //  return;
    //}
  };


  typedef CoverTreeNode<RefStat> TreeType;

 private:

  Matrix queries_, references_;

  TreeType *query_tree_;

  TreeType *reference_tree_;

  //ArrayList<double> neighbor_distances_;

  //ArrayList<index_t> neighbor_indices_;

  index_t knns_;

  struct datanode *module_;

 public:

  AllKNN() {
    query_tree_ = NULL;
    reference_tree_ = NULL;
  }

  ~AllKNN() {
     if (query_tree_ != NULL) {
      delete query_tree_;
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
  }

  inline void set_neighbor_distances(ArrayList<double> *neighbor_distances, 
				     index_t query_index, double dist) {

    for (index_t i = 0; i < knns_; i++) {
      (*neighbor_distances)[query_index + i] = dist;
    }
    return;
  }

  inline void update_neighbor_distances(ArrayList<double> *neighbor_distances, 
					index_t query_index, double dist) {

    for (index_t i = 0; i < knns_ - 1; i++) {

      if (dist < (*neighbor_distances)[i + query_index + 1]) {
	(*neighbor_distances)[i + query_index] = 
	  (*neighbor_distances)[i + query_index + 1];
      }
      else {
	(*neighbor_distances)[i + query_index] = dist;
	break;
      }
    }

    if (i == knns_ - 1) {
      (*neighbor_distances)[i + query_index] = dist;
    }

    return;
  }

  inline void reset_new_leaf_nodes(ArrayList<TreeType*> *new_leaf_nodes) {

    index_t size = new_leaf_nodes->size();
 
    for (index_t i = 0; i < size; i++) {
      (*new_leaf_nodes)[i]->stat().pop_last_distance();
    }

    return;
  }

  inline void reset_new_cover_sets(ArrayList<ArrayList<TreeType*> > *new_cover_sets, 
				   index_t current_scale, index_t max_scale) {

    for (index_t i = current_scale; i < max_scale; i++) {

      index_t size = (*new_cover_sets)[i].size();
      for (index_t j = 0; j < size; j++) {
	(*new_cover_sets)[i][j]->stat().pop_last_distance();
      }
    }
    return;
  }

  inline void CopyLeafNodes(TreeType *query, 
			    ArrayList<double> *neighbor_distances,
			    ArrayList<TreeType*> *leaf_nodes,
			    ArrayList<TreeType*> *new_leaf_nodes) {

    index_t size = leaf_nodes->size();
    Vector q_point;

    new_leaf_nodes->Resize(0);
    queries_.MakeColumnVector(query->point(), &q_point);

    for (index_t i = 0; i < size; i++) {
      double upper_bound = (*neighbor_distances)[query->point()] 
	+ query->max_dist_to_grandchild();

      if ((*leaf_nodes)[i]->stat().distance_to_qnode() 
	  - query->dist_to_parent() <= upper_bound) {

	Vector r_point;
	references_.MakeColumnVector(*(leaf_nodes)[i]->point(), &r_point);

	double dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));

	if (dist <= upper_bound) {
	  if (dist <= (*neighbor_distances)[query->point()]) {
	    update_neighbor_distances(neighbor_distances, 
				      query->point(), dist);
	  }

	  (*leaf_nodes)[i]->stat().set_distance_to_qnode(dist);
	  new_leaf_nodes->PushBackCopy((*leaf_nodes)[i]);
	}
      }
    }

    return;
  }

  inline void CopyNodes(TreeType *query, 
			ArrayList<double> *neighbor_distances, 
			ArrayList<ArrayList<TreeType*> > *cover_sets, 
			ArrayList<ArrayList<TreeType*> > *new_cover_sets, 
			index_t current_scale, index_t max_scale) {
    Vector q_point;

    new_cover_sets->Init(101);
    for (index_t i = 0; i < 101; i++) {
      (*new_cover_sets)[i].Init(0);
    }
    queries_.MakeColumnVector(query->point(), &q_node);


    for (index_t i = current_scale; i < max_scale; i++) {

      index_t size = (*cover_sets)[i].size();
      for (index_t j = 0; j < size; j++) {

	double upper_bound = (*neighbor_distances)[query->point()]
	  + query->max_distance_to_grandchild() 
	  + (*cover_set)[i][j]->max_distance_to_grandchild();

	if ((*cover_set)[i][j]->stat().distance_to_qnode() 
	    - query->dist_to_parent() <= upper_bound) {

	  Vector r_point;
	  references_.MakeColumnVector((*cover_sets)[i][j]->point(), 
				       &r_point);

	  double dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));

	  if (dist <= upper_bound) {
	    if (dist <= (*neighbor_distances)[query->point()]) {
	      update_neighbor_distances(neighbor_distances,
					query->point(), dist);
	    }

	    (*cover_sets)[i][j]->stat().set_distance_to_qnode(dist);
	    (*new_cover_sets)[i].PushBackCopy((*cover_sets)[i][j]);
	  }
	}
      }
    }
    return;
  }

  inline void DescendTheRefTree(TreeType *query, 
				ArrayList<double> *neighbor_distance, 
				ArrayList<ArrayList<TreeType*> > *cover_sets, 
				ArrayList<TreeType*> *leaf_nodes,
				index_t current_scale, index_t *max_scale) {

    Vector q_point;
    index_t size = (*cover_sets)[current_scale].size();

    queries_.MakeColumnVector(query->point(), &q_point);

    for (index_t i = 0; i < size; i++) {

      double upper_bound = (*neighbor_distance)[query->point()] 
	+ 2 * query->max_distance_to_grandchild();

      if ((*cover_sets)[current_scale][i]->stat().distance_to_qnode() 
	  <= upper_bound + 
	  (*cover_sets)[current_scale][i]->max_distance_to_grandchild()) {

	TreeType *self_child = (*cover_sets)[current_scale][i]->child(0);

	if ((*cover_sets)[current_scale][i]->stat().distance_to_qnode()
	    <= upper_bound + self_child->max_distance_to_grandchild()) {

	  if (self_child->num_of_children() > 0) {

	    if (*max_scale < self_child->scale_depth()) {
	      *max_scale = self_child->scale_depth();
	    }

	    self_child->stat().set_distance_to_qnode((*cover_sets)[current_scale][i]->stat().distance_to_qnode());
	    (*cover_sets)[self_child->scale_depth()].PushBackCopy(self_child);
	  }
	  else {
	    if ((*cover_sets)[current_scale][i]->stat().distance_to_qnode() 
		<= upper_bound) {
	      self_child->stat().set_distance_to_qnode((*cover_sets)[current_scale][i]->stat().distance_to_qnode());
	      leaf_nodes->PushBackCopy(self_child);
	    }
	  }
	}

	index_t num_of_children = (*cover_sets)[current_scale][i]->num_of_children();
	for (index_t in = 1; in < num_of_children; in++) {

	  TreeType *child = (*cover_sets)[current_scale}[i]->child(in);

	  double new_upper_bound = (*neighbor_distances)[query->point()] 
	    + child->max_distance_to_grandchild() 
	    + 2 * query->max_distance_to_grandchild();

	  if ((*cover_sets)[current_scale][i]->stat().distance_to_qnode()
	      - child->dist_to_parent() <= new_upper_bound) {

	    Vector r_point;
	    references_.MakeColumnVector(child->point(), &r_point);

	    double dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));

	    if (dist <= new_upper_bound) {

	      if (dist < (*neighbor_distances)[query->point()]) {
		update_neighbor_distances(neighbor_distances,
					  query->point(), dist);
	      }
	     
	      if (child->num_of_children() > 0) {
		if (*max_scale < child->scale_depth()) {
		  *max_scale = child->scale_depth();
		}
		child->stat().set_distance_to_qnode(dist);
		(*cover_sets)[child->scale_depth()].PushBackCopy(child);
	      }
	      else {
		if (dist <= new_upper_bound - child->max_distance_to_grandchild()) {
		  child->stat().set_distance_to_qnode(dist);
		  leaf_nodes->PushBackCopy(child);
		}
	      }
	    }
	  }
	}
      }
    }
    return;
  }

  void ComputeBaseCase(TreeType *query, 
		       ArrayList<double> *neighbor_distances,
		       ArrayList<TreeType*> *leaf_nodes, 
		       ArrayList<index_t> *neighbor_indices) {
    if (query->num_of_children()) {

      TreeType *self_child = query->child(0);
      ComputeBaseCase(self_child, neighbor_distances, leaf_nodes, neighbor_indices);

      index_t num_of_children = query->num_of_children();
      for (index_t i = 1; i < num_of_children; i++) {

	TreeType *child = query->child(i);
	ArrayList<TreeType*> new_leaf_nodes;

	new_leaf_nodes.Init(0);

	set_neighbor_distances(neighbor_distances, child->point(), 
			       (*neighbor_distances)[query->point()] 
			       + child->dist_to_parent());

	CopyLeafNodes(child, neighbor_distances, leaf_nodes, &new_leaf_nodes);
	ComputeBaseCase(child, neighbor_distances, &new_leaf_nodes, neighbor_indices);
	reset_new_leaf_nodes(&new_leaf_nodes);
      }
    }
    else {

      index_t size = leaf_nodes->size();
      for (index_t i = 0; i < size; i++) {
	if ((*leaf_nodes)[i]->stat().distance_to_qnode() 
	    <= (*neighbor_distances)[query->point()]) {
	  for (index_t j = 0; j < knns_; j++) {
	    if ((*leaf_nodes)[i]->stat().distance_to_qnode() == 
		(*neighbor_distances)[query->point()+j]) { //need to modify this later
	      (*neighbor_indices)[query->point() + j] = (*leaf_nodes)[i]->point();
	      break;
	    }
	  }
	}
      }
    }
    return;
  }

  void ComputeNeighborRecursion(TreeType* query, 
				ArrayList<double> *neighbor_distances, 
				ArrayList<ArrayList<TreeType*> > *cover_sets,
				ArrayList<TreeType*> leaf_nodes,
				index_t current_scale, index_t max_scale,
				ArrayList<index_t> *neighbor_indices) {

    if (current_scale > max_scale) {
      ComputeBaseCase(query, neighbor_distances, leaf_nodes, neighbor_indices);
    }
    else {
      if ((query->scale_depth() <= current_scale) && (query->scale_depth() != 100)) {

	for (index_t in = 1; in < query()->num_of_children(); in++) {
	  TreeType *child = query->child(in);
	  ArrayList<TreeType*> new_leaf_nodes;
	  ArrayList<ArrayList<TreeType*> > new_cover_sets;

	  new_leaf_nodes.Init(0);

	  set_neighbor_distances(neighbor_distances, child->point(), 
				 (*neighbor_distances)[query->point()] 
				 + child->dist_to_parent());

	  CopyLeafNodes(child, neighbor_distances, leaf_nodes, &new_leaf_nodes);
	  CopyCoverSets(child, neighbor_distances, cover_sets, &new_cover_sets,
			current_scale, max_scale);
	  ComputeNeighborRecursion(child, neighbor_distances, &new_cover_sets,
				   &new_leaf_nodes, current_scale, max_scale, 
				   neighbor_indices);
	  reset_new_leaf_nodes(&new_leaf_nodes);
	  reset_new_cover_sets(&new_cover_sets);
	}

	ComputeNeighborRecursion(query->child(0), neighbor_distances, cover_sets, 
				 leaf_nodes, current_scale, max_scale, 
				 neighbor_indices);
      }
      else {
	halfsort(&((*cover_sets)[current_scale]));
	DescendRefTree(query, neighbor_distances, cover_sets, 
		       leaf_nodes, current_scale, &max_scale);
	++current_scale;
	ComputeNeighborRecursion(query, neighbor_distances, cover_sets, 
				 leaf_nodes, current_scale, max_scale, 
				 neighbor_indices);
      }
    }
    return;
  }

  void ComputeNeighbors(ArrayList<index_t> *neighbor_indices, 
			ArrayList<double> *neighbor_distances) {

  }

};

#endif
