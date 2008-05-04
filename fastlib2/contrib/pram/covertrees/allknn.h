#ifndef ALLKNN_H
#define ALLKNN_H

#include <fastlib/fastlib.h>
#include "cover_tree.h"
#include "ctree.h"
#include "distances.h"

class AllKNN {

  class RefStat {
  private:
    ArrayList <double> distance_to_qnode_;

  public:

    RefStat() {
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

    void Init() {
      distance_to_qnode_.Init(0);
      return;
    }
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

  // getters

  TreeType *query_tree() {
    return query_tree_;
  }

  TreeType *reference_tree() {
    return reference_tree_;
  }

  inline double compare(TreeType **p, TreeType **q) {
    return (*p)->stat().distance_to_qnode() - (*q)->stat().distance_to_qnode();
  }

  inline void swap(TreeType **p, TreeType **q) {
    TreeType *temp = NULL;
    temp = *p;
    *p = *q;
    *q = temp;
    return;
  }

  void halfsort(ArrayList<TreeType*> *cover_set) {

    if (cover_set->size() <= 1) {
      return;
    }
    
    register TreeType **begin = cover_set->begin();
    TreeType **end = &(cover_set->back());
    TreeType **right = end;
    TreeType **left;

    while (right > begin) {

      TreeType **mid = begin + ((end - begin) >> 1);

      if (compare(mid, begin) < 0.0) {
	swap(mid, begin);
      }
      if (compare(end, mid) < 0.0) {
	swap(mid, end);
	if (compare(mid, begin) < 0.0) {
	  swap(mid, begin);
	}
      }

      left = begin + 1;
      right = end - 1;

      do {

	while (compare(left, mid) < 0.0) {
	  left++;
	}

	while (compare(mid, right) < 0.0) {
	  right--;
	}

	if (left < right) {
	  swap(left, right);
	  if (mid == left) {
	    mid = right;
	  }
	  else if (mid == right) {
	    mid = left;
	  }
	  left++;
	  right--;
	}
	else if (left == right) {
	  left ++;
	  right--;
	  break;
	}
      } while (left <= right);

      end = right;
    }

    return;    
  }

  inline void set_neighbor_distances(ArrayList<double> *neighbor_distances, 
				     index_t query_index, double dist) {

    for (index_t i = 0; i < knns_; i++) {
      (*neighbor_distances)[query_index * knns_ + i] = dist;
    }
    return;
  }

  inline void update_neighbor_distances(ArrayList<double> *neighbor_distances, 
					index_t query_index, double dist) {

    index_t i, start = query_index * knns_;
    for (i = 0; i < knns_ - 1; i++) {

      if (dist < (*neighbor_distances)[i + start + 1]) {
	(*neighbor_distances)[i + start] = 
	  (*neighbor_distances)[i + start + 1];
      }
      else {
	(*neighbor_distances)[i + start] = dist;
	break;
      }
    }

    if (i == knns_ - 1) {
      (*neighbor_distances)[i + start] = dist;
    }

    return;
  }

  inline void reset_new_leaf_nodes(ArrayList<TreeType*> *new_leaf_nodes) {

    TreeType **begin = new_leaf_nodes->begin();
    TreeType **end = new_leaf_nodes->end();
 
    for (; begin < end; begin++) {
      (*begin)->stat().pop_last_distance();
    }

    return;
  }

  inline void reset_new_cover_sets(ArrayList<ArrayList<TreeType*> > *new_cover_sets, 
				   index_t current_scale, index_t max_scale) {

    for (index_t i = current_scale; i <= max_scale; i++) {

      TreeType **begin = (*new_cover_sets)[i].begin();
      TreeType **end = (*new_cover_sets)[i].end();
      for (; begin < end; begin++) {
	(*begin)->stat().pop_last_distance();
      }
    }
    return;
  }

  inline void CopyLeafNodes(TreeType *query, 
			    ArrayList<double> *neighbor_distances,
			    ArrayList<TreeType*> *leaf_nodes,
			    ArrayList<TreeType*> *new_leaf_nodes) {

    TreeType **begin = leaf_nodes->begin();
    TreeType **end = leaf_nodes->end();
    Vector q_point;

    new_leaf_nodes->Resize(0);
    queries_.MakeColumnVector(query->point(), &q_point);

    for (; begin < end; begin++) {
      double upper_bound = (*neighbor_distances)[query->point() * knns_] 
	+ query->max_dist_to_grandchild();
    
      if ((*begin)->stat().distance_to_qnode() 
	  - query->dist_to_parent() <= upper_bound) {

	Vector r_point;
	references_.MakeColumnVector((*begin)->point(), &r_point);

	//double dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	double dist = pdc::DistanceEuclidean(q_point, r_point, upper_bound);

	if (dist <= upper_bound) {
	  if (dist <= (*neighbor_distances)[query->point()*knns_]) {
	    update_neighbor_distances(neighbor_distances, 
				      query->point(), dist);
	  }

	  (*begin)->stat().set_distance_to_qnode(dist);
	  new_leaf_nodes->PushBackCopy(*begin);
	}
      }
    }

    return;
  }

  inline void CopyCoverSets(TreeType *query, 
			    ArrayList<double> *neighbor_distances, 
			    ArrayList<ArrayList<TreeType*> > *cover_sets, 
			    ArrayList<ArrayList<TreeType*> > *new_cover_sets, 
			    index_t current_scale, index_t max_scale) {
    Vector q_point;

    new_cover_sets->Init(101);
    for (index_t i = 0; i < 101; i++) {
      (*new_cover_sets)[i].Init(0);
    }
    queries_.MakeColumnVector(query->point(), &q_point);


    for (index_t i = current_scale; i <= max_scale; i++) {

      TreeType **begin = (*cover_sets)[i].begin();
      TreeType **end = (*cover_sets)[i].end();

      for (; begin < end; begin++) {

	double upper_bound = (*neighbor_distances)[query->point() * knns_]
	  + query->max_dist_to_grandchild() 
	  + (*begin)->max_dist_to_grandchild();

	if ((*begin)->stat().distance_to_qnode() 
	    - query->dist_to_parent() <= upper_bound) {

	  Vector r_point;
	  references_.MakeColumnVector((*begin)->point(), 
				       &r_point);

	  //double dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	  double dist = pdc::DistanceEuclidean(q_point, r_point, upper_bound);

	  if (dist <= upper_bound) {
	    if (dist <= (*neighbor_distances)[query->point()* knns_]) {
	      update_neighbor_distances(neighbor_distances,
					query->point(), dist);
	    }

	    (*begin)->stat().set_distance_to_qnode(dist);
	    (*new_cover_sets)[i].PushBackCopy(*begin);
	  }
	}
      }
    }
    return;
  }

  inline void DescendTheRefTree(TreeType *query, 
				ArrayList<double> *neighbor_distances, 
				ArrayList<ArrayList<TreeType*> > *cover_sets, 
				ArrayList<TreeType*> *leaf_nodes,
				index_t current_scale, index_t *max_scale) {

    Vector q_point;
    TreeType **begin = (*cover_sets)[current_scale].begin();
    TreeType **end = (*cover_sets)[current_scale].end();
    
    queries_.MakeColumnVector(query->point(), &q_point);

    for (; begin < end; begin++) {

      double upper_bound = (*neighbor_distances)[query->point() * knns_] 
	+ 2 * query->max_dist_to_grandchild();

      if ((*begin)->stat().distance_to_qnode() 
	  <= upper_bound + (*begin)->max_dist_to_grandchild()) {
	
	TreeType **self_child = (*begin)->children()->begin();

	if ((*begin)->stat().distance_to_qnode()
	    <= upper_bound + (*self_child)->max_dist_to_grandchild()) {
	  
	  if ((*self_child)->num_of_children() > 0) {
	  
	    if (*max_scale < (*self_child)->scale_depth()) {
	      *max_scale = (*self_child)->scale_depth();
	    }

	    (*self_child)->stat().set_distance_to_qnode((*begin)->stat().distance_to_qnode());
	    (*cover_sets)[(*self_child)->scale_depth()].PushBackCopy(*self_child);
	  }
	  else {
	    if ((*begin)->stat().distance_to_qnode() <= upper_bound) {
	      (*self_child)->stat().set_distance_to_qnode((*begin)->stat().distance_to_qnode());
	      leaf_nodes->PushBackCopy(*self_child);
	    }
	  }
	}

      TreeType **child_end = (*begin)->children()->end();
	for (++self_child; self_child < child_end; self_child++) {


	  double new_upper_bound = (*neighbor_distances)[query->point() * knns_] 
	    + (*self_child)->max_dist_to_grandchild() 
	    + 2 * query->max_dist_to_grandchild();

	  if ((*begin)->stat().distance_to_qnode()
	      - (*self_child)->dist_to_parent() <= new_upper_bound) {

	    Vector r_point;
	    references_.MakeColumnVector((*self_child)->point(), &r_point);

	    //double dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	    double dist = pdc::DistanceEuclidean(q_point, r_point, new_upper_bound);

	    if (dist <= new_upper_bound) {

	      if (dist < (*neighbor_distances)[query->point() * knns_]) {
		update_neighbor_distances(neighbor_distances,
					  query->point(), dist);
	      }
	     
	      if ((*self_child)->num_of_children() > 0) {
		
		if (*max_scale < (*self_child)->scale_depth()) {
		  *max_scale = (*self_child)->scale_depth();
		}
		(*self_child)->stat().set_distance_to_qnode(dist);
		(*cover_sets)[(*self_child)->scale_depth()].PushBackCopy(*self_child);
	      }
	      else {
		if (dist <= new_upper_bound - (*self_child)->max_dist_to_grandchild()) {
		  (*self_child)->stat().set_distance_to_qnode(dist);
		  leaf_nodes->PushBackCopy(*self_child);
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
    if (query->num_of_children() > 0) {

      TreeType **self_child = query->children()->begin();
      ComputeBaseCase(*self_child, neighbor_distances, leaf_nodes, neighbor_indices);

      TreeType **child_end = query->children()->end();
      for (++self_child; self_child < child_end; self_child++) {

	ArrayList<TreeType*> new_leaf_nodes;

	new_leaf_nodes.Init(0);

	set_neighbor_distances(neighbor_distances, (*self_child)->point(), 
			       (*neighbor_distances)[query->point() * knns_] 
			       + (*self_child)->dist_to_parent());

	CopyLeafNodes(*self_child, neighbor_distances, leaf_nodes, &new_leaf_nodes);
	ComputeBaseCase(*self_child, neighbor_distances, &new_leaf_nodes, neighbor_indices);
	reset_new_leaf_nodes(&new_leaf_nodes);
      }
    }
    else {

      // option of optimization here
      index_t size = leaf_nodes->size();
      for (index_t i = 0, j = 0; i < size && j < knns_; i++) {
	if ((*leaf_nodes)[i]->stat().distance_to_qnode() 
	    <= (*neighbor_distances)[query->point() * knns_]) {

	  index_t k = 0;
	  while ((*leaf_nodes)[i]->stat().distance_to_qnode() > 
		 (*neighbor_distances)[query->point() * knns_ + knns_ - k -1]) { 
	    k++;
	  }
	  
	  (*neighbor_indices)[query->point() * knns_ + k] = (*leaf_nodes)[i]->point();
	  j++;
	  
	}
      }
    }
    return;
  }

  void ComputeNeighborRecursion(TreeType *query, 
				ArrayList<double> *neighbor_distances, 
				ArrayList<ArrayList<TreeType*> > *cover_sets,
				ArrayList<TreeType*> *leaf_nodes,
				index_t current_scale, index_t max_scale,
				ArrayList<index_t> *neighbor_indices) {

    if (current_scale > max_scale) {
      ComputeBaseCase(query, neighbor_distances, leaf_nodes, neighbor_indices);
    }
    else {
      if ((query->scale_depth() <= current_scale) && (query->scale_depth() != 100)) {

	TreeType **child_begin = query->children()->begin();
	TreeType **child_end = query->children()->end();

	for (++child_begin; child_begin < child_end; child_begin++) {

	  ArrayList<TreeType*> new_leaf_nodes;
	  ArrayList<ArrayList<TreeType*> > new_cover_sets;

	  new_leaf_nodes.Init(0);

	  set_neighbor_distances(neighbor_distances, (*child_begin)->point(), 
				 (*neighbor_distances)[query->point() * knns_] 
				 + (*child_begin)->dist_to_parent());

	  CopyLeafNodes(*child_begin, neighbor_distances, leaf_nodes, &new_leaf_nodes);
	  CopyCoverSets(*child_begin, neighbor_distances, cover_sets, &new_cover_sets,
			current_scale, max_scale);
	  ComputeNeighborRecursion(*child_begin, neighbor_distances, &new_cover_sets,
				   &new_leaf_nodes, current_scale, max_scale, 
				   neighbor_indices);
	  reset_new_leaf_nodes(&new_leaf_nodes);
	  reset_new_cover_sets(&new_cover_sets, current_scale, max_scale);
	}

	ComputeNeighborRecursion(query->child(0), neighbor_distances, cover_sets, 
				 leaf_nodes, current_scale, max_scale, 
				 neighbor_indices);
      }
      else {
	halfsort(&((*cover_sets)[current_scale]));
	
	DescendTheRefTree(query, neighbor_distances, cover_sets, 
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

    Vector q_root, r_root;
    ArrayList<ArrayList<TreeType*> > cover_sets;
    ArrayList<TreeType*> leaf_nodes;
    index_t current_scale = 0, max_scale = 0;

    queries_.MakeColumnVector(query_tree_->point(), &q_root);
    references_.MakeColumnVector(reference_tree_->point(), &r_root);
    neighbor_indices->Init(knns_* queries_.n_cols());
    neighbor_distances->Init(knns_ * queries_.n_cols());
    cover_sets.Init(101);
    for (index_t i = 0; i < 101; i++) {
      cover_sets[i].Init(0);
    }
    leaf_nodes.Init(0);

    set_neighbor_distances(neighbor_distances, query_tree_->point(), DBL_MAX);
  
    //double dist = sqrt(la::DistanceSqEuclidean(q_root, r_root));
    double dist = pdc::DistanceEuclidean(q_root, r_root, sqrt(DBL_MAX));
    
    update_neighbor_distances(neighbor_distances, query_tree_->point(), dist);

    reference_tree_->stat().set_distance_to_qnode(dist);
    cover_sets[0].PushBackCopy(reference_tree_);

    ComputeNeighborRecursion(query_tree_, neighbor_distances, &cover_sets, 
			     &leaf_nodes, current_scale, max_scale, 
			     neighbor_indices);

    return;
  }

  void Init(const Matrix& queries, const Matrix& references, struct datanode *module) {

    module_ = module;
    queries_.Copy(queries);
    references_.Copy(references);

    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

    knns_ = fx_param_int(module_, "knns", 1);

    fx_timer_start(module_, "tree_building");

    query_tree_ = ctree::MakeCoverTree<TreeType>(queries_);
    reference_tree_ = ctree::MakeCoverTree<TreeType>(references_);

    fx_timer_stop(module_, "tree_building");

    //NOTIFY("Query Tree:");
    //ctree::PrintTree<TreeType>(query_tree_);
    //NOTIFY("Reference Tree:");
    //ctree::PrintTree<TreeType>(reference_tree_);

    return;
  }

};

#endif
