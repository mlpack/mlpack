#ifndef ALLKNN_H
#define ALLKNN_H

#include <fastlib/fastlib.h>
#include "cover_tree.h"
#include "ctree.h"
#include "distances.h"

template<typename T>
class AllKNN {

  class RefStat {
  private:
    ArrayList <T> distance_to_qnode_;
    
  public:

    RefStat() {
    }

    ~RefStat() {
    }

    inline T distance_to_qnode() {
      return distance_to_qnode_.back();
    }

    void set_distance_to_qnode(T new_dist) {
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


  typedef CoverTreeNode<RefStat, T> TreeType;

 private:

  GenMatrix<T> queries_, references_;

  TreeType *query_tree_;

  TreeType *reference_tree_;

  TreeType *cquery_tree_;

  TreeType *creference_tree_;

  index_t ref_root_node_index_, query_root_node_index_;

  //ArrayList<T> neighbor_distances_;

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

  
  inline T compare(TreeType **p, TreeType **q) {
    return (*p)->stat().distance_to_qnode() - (*q)->stat().distance_to_qnode();
  }

  inline void swap(TreeType **p, TreeType **q) {
    TreeType *temp = NULL;
    temp = *p;
    *p = *q;
    *q = temp;
  
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

  }

  
  inline void set_neighbor_distances(ArrayList<T> *neighbor_distances, 
				     index_t query_index, T dist) {

    index_t start = query_index * knns_;
    T *begin = neighbor_distances->begin() + start;
    T *end = begin + knns_;

    for (; begin != end; begin++) {
      *begin = dist;
    }

  }

  inline void update_neighbor_distances(ArrayList<T> *neighbor_distances, 
					index_t query_index, T dist) {

    index_t start = query_index * knns_;
    T *begin = neighbor_distances->begin() + start;
    T *end = begin + knns_ - 1;
 
    for (; end != begin; begin++) {

      if (dist < *(begin + 1)) {
	*begin = *(begin + 1);
      }
      else {
	*begin = dist;
	break;
      }
    }

    if (end == begin) {
      *begin = dist;
    }

  }

  inline void reset_new_leaf_nodes(ArrayList<TreeType*> *new_leaf_nodes) {

    TreeType **begin = new_leaf_nodes->begin();
    TreeType **end = new_leaf_nodes->end();
 
    for (; begin != end; begin++) {
      (*begin)->stat().pop_last_distance();
    }

  }

  inline void reset_new_cover_sets(ArrayList<ArrayList<TreeType*> > *new_cover_sets, 
				   index_t current_scale, index_t max_scale) {

    for (index_t i = current_scale; i <= max_scale; i++) {

      TreeType **begin = (*new_cover_sets)[i].begin();
      TreeType **end = (*new_cover_sets)[i].end();
      for (; begin != end; begin++) {
	(*begin)->stat().pop_last_distance();
      }
    }

  }

  
  inline void CopyLeafNodes(TreeType *query, 
			    ArrayList<T> *neighbor_distances,
			    ArrayList<TreeType*> *leaf_nodes,
			    ArrayList<TreeType*> *new_leaf_nodes) {

    TreeType **begin = leaf_nodes->begin();
    TreeType **end = leaf_nodes->end();
    GenVector<T> q_point;
    T *query_upper_bound = &((*neighbor_distances)[query->point() * knns_]);
    T query_max_dist = query->max_dist_to_grandchild();
    T query_parent_dist = query->dist_to_parent();

    new_leaf_nodes->Resize(0);
    queries_.MakeColumnVector(query->point(), &q_point);

    for (; begin != end; begin++) {
      T upper_bound = *query_upper_bound + query_max_dist;
    
      if ((*begin)->stat().distance_to_qnode() 
	  - query_parent_dist <= upper_bound) {

	GenVector<T> r_point;
	references_.MakeColumnVector((*begin)->point(), &r_point);

	//T dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	T dist = pdc::DistanceEuclidean(q_point, r_point, upper_bound);

	if (dist <= upper_bound) {
	  if (dist <= *query_upper_bound) {
	    update_neighbor_distances(neighbor_distances, 
				      query->point(), dist);
	  }

	  (*begin)->stat().set_distance_to_qnode(dist);
	  new_leaf_nodes->PushBackCopy(*begin);
	}
      }
    }

  }

  
  inline void CopyCoverSets(TreeType *query, 
			    ArrayList<T> *neighbor_distances, 
			    ArrayList<ArrayList<TreeType*> > *cover_sets, 
			    ArrayList<ArrayList<TreeType*> > *new_cover_sets, 
			    index_t current_scale, index_t max_scale) {
    GenVector<T> q_point;
    T *query_upper_bound = &((*neighbor_distances)[query->point() * knns_]);
    T query_max_dist = query->max_dist_to_grandchild();
    T query_parent_dist = query->dist_to_parent();

    new_cover_sets->Init(101);
    for (index_t i = 0; i < 101; i++) {
      (*new_cover_sets)[i].Init(0);
    }
    queries_.MakeColumnVector(query->point(), &q_point);


    for (index_t i = current_scale; i <= max_scale; i++) {

      TreeType **begin = (*cover_sets)[i].begin();
      TreeType **end = (*cover_sets)[i].end();

      for (; begin != end; begin++) {

	T upper_bound = *query_upper_bound
	  + query_max_dist 
	  + (*begin)->max_dist_to_grandchild();

	if ((*begin)->stat().distance_to_qnode() 
	    - query_parent_dist <= upper_bound) {

	  GenVector<T> r_point;
	  references_.MakeColumnVector((*begin)->point(), 
				       &r_point);

	  //T dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	  T dist = pdc::DistanceEuclidean<T>(q_point, r_point, upper_bound);

	  if (dist <= upper_bound) {
	    if (dist <= *query_upper_bound) {
	      update_neighbor_distances(neighbor_distances,
					query->point(), dist);
	    }

	    (*begin)->stat().set_distance_to_qnode(dist);
	    (*new_cover_sets)[i].PushBackCopy(*begin);
	  }
	}
      }
    }

  }

  inline void DescendTheRefTree(TreeType *query, 
				ArrayList<T> *neighbor_distances, 
				ArrayList<ArrayList<TreeType*> > *cover_sets, 
				ArrayList<TreeType*> *leaf_nodes,
				index_t current_scale, index_t *max_scale) {

    TreeType **begin = (*cover_sets)[current_scale].begin();
    TreeType **end = (*cover_sets)[current_scale].end();

    T query_max_dist = query->max_dist_to_grandchild();
    T *query_upper_bound = &((*neighbor_distances)[query->point() * knns_]);    

    GenVector<T> q_point;
    queries_.MakeColumnVector(query->point(), &q_point);

    for (; begin != end; begin++) {

      T upper_bound = *query_upper_bound + query_max_dist + query_max_dist;

      T present_upper_dist = (*begin)->stat().distance_to_qnode();

      if (present_upper_dist <= upper_bound + (*begin)->max_dist_to_grandchild()) {
	
	TreeType **self_child = (*begin)->children()->begin();

	if (present_upper_dist <= upper_bound 
	    + (*self_child)->max_dist_to_grandchild()) {
	  
	  if ((*self_child)->num_of_children() > 0) {
	  
	    if (*max_scale < (*self_child)->scale_depth()) {
	      *max_scale = (*self_child)->scale_depth();
	    }

	    (*self_child)->stat().set_distance_to_qnode(present_upper_dist);
	    (*cover_sets)[(*self_child)->scale_depth()].PushBackCopy(*self_child);
	  }
	  else {
	    if (present_upper_dist <= upper_bound) {
	      (*self_child)->stat().set_distance_to_qnode(present_upper_dist);
	      leaf_nodes->PushBackCopy(*self_child);
	    }
	  }
	}

	TreeType **child_end = (*begin)->children()->end();
	for (++self_child; self_child != child_end; self_child++) {


	  T new_upper_bound = *query_upper_bound 
	    + (*self_child)->max_dist_to_grandchild() 
	    + query_max_dist + query_max_dist;

	  if (present_upper_dist - (*self_child)->dist_to_parent()
	      <= new_upper_bound) {

	    GenVector<T> r_point;
	    references_.MakeColumnVector((*self_child)->point(), &r_point);

	    //T dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	    T dist = pdc::DistanceEuclidean<T>(q_point, r_point, new_upper_bound);

	    if (dist <= new_upper_bound) {

	      if (dist < *query_upper_bound) {
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

  }

  void ComputeBaseCase(TreeType *query, 
		       ArrayList<T> *neighbor_distances,
		       ArrayList<TreeType*> *leaf_nodes, 
		       ArrayList<index_t> *neighbor_indices) {
    if (query->num_of_children() > 0) {

      TreeType **self_child = query->children()->begin();
      ComputeBaseCase(*self_child, neighbor_distances, leaf_nodes, neighbor_indices);
      T *query_upper_bound = &((*neighbor_distances)[query->point() * knns_]);
      TreeType **child_end = query->children()->end();

      for (++self_child; self_child != child_end; self_child++) {

	ArrayList<TreeType*> new_leaf_nodes;

	new_leaf_nodes.Init(0);
	set_neighbor_distances(neighbor_distances, (*self_child)->point(), 
			       *query_upper_bound + (*self_child)->dist_to_parent());

	CopyLeafNodes(*self_child, neighbor_distances, leaf_nodes, &new_leaf_nodes);
	ComputeBaseCase(*self_child, neighbor_distances, &new_leaf_nodes, neighbor_indices);
	reset_new_leaf_nodes(&new_leaf_nodes);
      }
    }
    else {


      TreeType **begin_nn = leaf_nodes->begin();
      TreeType **end_nn = leaf_nodes->end();
      T *begin = neighbor_distances->begin() + query->point() * knns_;
      T *end = neighbor_distances->begin() + query->point() * knns_ + knns_ ;
      index_t *indices = neighbor_indices->begin() + query->point() * knns_;
      ArrayList<bool> flags;

      flags.Init(knns_);
      for (index_t i = 0; i < knns_; i++) {
	flags[i] = 0;
      }

      begin = neighbor_distances->begin() + query->point() * knns_;
      for (index_t j = 0; j < knns_ && begin_nn != end_nn; begin_nn++) {
	if ((*begin_nn)->stat().distance_to_qnode() <= *begin) {
	  
	  index_t k = 0;
	  while ((*begin_nn)->stat().distance_to_qnode() > *(end - k- 1)) { 
	    k++;
	  }
	  while (flags[k] == 1 && k < knns_ - 1) {
	    k++;
	  }
	  if (k < knns_ && flags[k] == 0) {
	    *(indices + k) = (*begin_nn)->point();
	    flags[k] = 1;
	    j++;	  
	  }
	}
      }
    }

  }

  void ComputeNeighborRecursion(TreeType *query, 
				ArrayList<T> *neighbor_distances, 
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
	T *query_upper_bound = &((*neighbor_distances)[query->point() * knns_]);

	for (++child_begin; child_begin != child_end; child_begin++) {

	  ArrayList<TreeType*> new_leaf_nodes;
	  ArrayList<ArrayList<TreeType*> > new_cover_sets;

	  new_leaf_nodes.Init(0);

	  set_neighbor_distances(neighbor_distances, (*child_begin)->point(), 
				 *query_upper_bound + (*child_begin)->dist_to_parent());

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

  }

  void ComputeNeighbors(ArrayList<index_t> *neighbor_indices, 
			ArrayList<T> *neighbor_distances) {

    GenVector<T> q_root, r_root;
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
  
    //T dist = sqrt(la::DistanceSqEuclidean(q_root, r_root));
    T dist = pdc::DistanceEuclidean<T>(q_root, r_root, sqrt(DBL_MAX));

    update_neighbor_distances(neighbor_distances, query_tree_->point(), dist);

    reference_tree_->stat().set_distance_to_qnode(dist);
    cover_sets[0].PushBackCopy(reference_tree_);

    ComputeNeighborRecursion(query_tree_, neighbor_distances, &cover_sets, 
			     &leaf_nodes, current_scale, max_scale, 
			     neighbor_indices);


  }

  void Init(const GenMatrix<T>& queries, 
	    const GenMatrix<T>& references, 
	    struct datanode *module) {

    module_ = module;
    queries_.Copy(queries);
    references_.Copy(references);

    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

    knns_ = fx_param_int(module_, "knns", 1);

    fx_timer_start(module_, "tree_building");

    query_tree_ = ctree::MakeCoverTree<TreeType, T>(queries_);
    reference_tree_ = ctree::MakeCoverTree<TreeType, T>(references_);

    fx_timer_stop(module_, "tree_building");

    //NOTIFY("Query Tree:");
    //ctree::PrintTree<TreeType>(query_tree_);
    //NOTIFY("Reference Tree:");
    //ctree::PrintTree<TreeType>(reference_tree_);

    return;
  }

  void MakeCentroidRootTree(GenMatrix<T> data, 
			    TreeType *data_tree, 
			    index_t *root_node_index, 
			    TreeType **cdata_tree) {

    GenMatrix<T> single_point;
    GenVector<T> centroid;
    ArrayList<index_t> neighbor_index;
    ArrayList<T> neighbor_distance;

    centroid.Init(data.n_rows());

    GetCentroid_(data, &centroid);
    //centroid.PrintDebug("centroid");
    single_point.AliasColVector(centroid);
    TreeType *temp_query = ctree::MakeCoverTree<TreeType, T>(single_point);

    GenVector<T> q_root, r_root;
    ArrayList<ArrayList<TreeType*> > cover_sets;
    ArrayList<TreeType*> leaf_nodes;
    index_t current_scale = 0, max_scale = 0;

    single_point.MakeColumnVector(temp_query->point(), &q_root);
    data.MakeColumnVector(data_tree->point(), &r_root);
    neighbor_index.Init(1);
    neighbor_distance.Init(1);
    cover_sets.Init(101);
    for (index_t i = 0; i < 101; i++) {
      cover_sets[i].Init(0);
    }
    leaf_nodes.Init(0);

    set_neighbor_distances(&neighbor_distance, temp_query->point(), DBL_MAX);
  
    //T dist = sqrt(la::DistanceSqEuclidean(q_root, r_root));
    T dist = pdc::DistanceEuclidean<T>(q_root, r_root, sqrt(DBL_MAX));
    //NOTIFY("%lf",dist);
    
    update_neighbor_distances(&neighbor_distance, temp_query->point(), dist);

    data_tree->stat().set_distance_to_qnode(dist);
    cover_sets[0].PushBackCopy(data_tree);

    ComputeNeighborRecursion(temp_query, &neighbor_distance, &cover_sets, 
			     &leaf_nodes, current_scale, max_scale, 
			     &neighbor_index);

    *root_node_index = neighbor_index[0];
    //NOTIFY("%"LI"d,%lf", neighbor_index[0], neighbor_distance[0]);
    GenVector<T> root_node, first_point, temp_point;
    data.MakeColumnVector(neighbor_index[0], &root_node);
    //root_node.PrintDebug("root");
    data.MakeColumnVector(0, &first_point);
    temp_point.Copy(first_point);
    first_point.CopyValues(root_node);
    root_node.CopyValues(temp_point);

    (*cdata_tree) = ctree::MakeCoverTree<TreeType, T>(data);
    //ctree::PrintTree<TreeType>(*cdata_tree);
    return;
  }

  void GetCentroid_(GenMatrix<T> data, GenVector<T> *centroid) {

    DEBUG_ASSERT(centroid->length() == data.n_rows());
    index_t size = data.n_cols();

    centroid->SetZero();
    for (index_t i = 0; i < data.n_cols(); i++) {
      GenVector<T> point;
      data.MakeColumnVector(i, &point);
      index_t length =  point.length();
      T *vx = centroid->ptr();
      T *vy = point.ptr();

      do {
	*vx = *vx + (*vy)/size;
	vx++;
	vy++;
      } while (--length);
    }

    return;
  }

  void MakeCCoverTrees() {

    fx_timer_start(module_, "centroid_tree_building");
    MakeCentroidRootTree(references_, reference_tree_, &ref_root_node_index_, &creference_tree_);
    //NOTIFY("Reference Tree:");
    //ctree::PrintTree<TreeType>(creference_tree_);

    MakeCentroidRootTree(queries_, query_tree_, &query_root_node_index_, &cquery_tree_);
    fx_timer_stop(module_, "centroid_tree_building");

    //NOTIFY("Query Tree:");
    //ctree::PrintTree<TreeType>(cquery_tree_);

    return;
  }

  void ComputeNeighborsNew(ArrayList<index_t> *neighbor_indices, 
			   ArrayList<T> *neighbor_distances) {

    GenVector<T> q_root, r_root;
    ArrayList<ArrayList<TreeType*> > cover_sets;
    ArrayList<TreeType*> leaf_nodes;
    index_t current_scale = 0, max_scale = 0;

    queries_.MakeColumnVector(cquery_tree_->point(), &q_root);
    references_.MakeColumnVector(creference_tree_->point(), &r_root);
    neighbor_indices->Init(knns_* queries_.n_cols());
    neighbor_distances->Init(knns_ * queries_.n_cols());
    cover_sets.Init(101);
    for (index_t i = 0; i < 101; i++) {
      cover_sets[i].Init(0);
    }
    leaf_nodes.Init(0);

    set_neighbor_distances(neighbor_distances, cquery_tree_->point(), DBL_MAX);
  
    //T dist = sqrt(la::DistanceSqEuclidean(q_root, r_root));
    T dist = pdc::DistanceEuclidean<T>(q_root, r_root, sqrt(DBL_MAX));
    
    update_neighbor_distances(neighbor_distances, cquery_tree_->point(), dist);

    creference_tree_->stat().set_distance_to_qnode(dist);
    cover_sets[0].PushBackCopy(creference_tree_);

    fx_timer_start(module_, "jhinchak_neighbors");
    ComputeNeighborRecursion(cquery_tree_, neighbor_distances, &cover_sets, 
			     &leaf_nodes, current_scale, max_scale, 
			     neighbor_indices);
    fx_timer_stop(module_, "jhinchak_neighbors");

    T distance;
    index_t index;
    for (index_t i = 0; i < knns_; i++) {
      index = (*neighbor_indices)[0 + i];
      distance = (*neighbor_distances)[0 + i];
      (*neighbor_indices)[0 + i] = (*neighbor_indices)[knns_ * query_root_node_index_ + i];
      (*neighbor_distances)[0 + i] = (*neighbor_distances)[knns_ * query_root_node_index_ + i];
      (*neighbor_indices)[knns_ * query_root_node_index_ + i] = index;
      (*neighbor_distances)[knns_ * query_root_node_index_ + i] = distance;
    }

    for (index_t i = 0; i < neighbor_indices->size(); i++) {
      if ((*neighbor_indices)[i] == 0) {
	(*neighbor_indices)[i] = ref_root_node_index_;
      }
      else {
	if ((*neighbor_indices)[i] == ref_root_node_index_) {
	  (*neighbor_indices)[i] = 0;
	}
      }
    }
  
    return;
  }


};

#endif
