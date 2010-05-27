/**
 * @file allknn_impl.h
 * 
 * Implementation of all the 
 * functions prototyped in 
 * allknn.h for the class AllKNN
 */
#ifndef ALLKNN_IMPL_GC_H
#define ALLKNN_IMPL_GC_H
#include "allknn_gc.h"

template<typename T>
void AllKNN<T>::halfsort_(ArrayList<TreeType*> *cover_set) {

  if (cover_set->size() <= 1) {
    return;
  }
    
  register TreeType **begin = cover_set->begin();
  TreeType **end = &(cover_set->back());
  TreeType **right = end;
  TreeType **left;

  while (right > begin) {

    TreeType **mid = begin + ((end - begin) >> 1);

    if (compare_(mid, begin) < 0.0) {
      swap_(mid, begin);
    }
    if (compare_(end, mid) < 0.0) {
      swap_(mid, end);
      if (compare_(mid, begin) < 0.0) {
	swap_(mid, begin);
      }
    }

    left = begin + 1;
    right = end - 1;

    do {

      while (compare_(left, mid) < 0.0) {
	left++;
      }

      while (compare_(mid, right) < 0.0) {
	right--;
      }

      if (left < right) {
	swap_(left, right);
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

template<typename T>
void AllKNN<T>::reset_leaf_nodes_(ArrayList<TreeType*> *leaf_nodes) {

  TreeType **begin = leaf_nodes->begin();
  TreeType **end = leaf_nodes->end();
 
  for (; begin != end; begin++) {
    (*begin)->stat().pop_last_distance();
  }

}

template<typename T>
void AllKNN<T>::reset_cover_sets_(ArrayList<ArrayList<TreeType*> > *cover_sets,
				  index_t current_scale, index_t max_scale) {
  
  for (index_t i = current_scale; i <= max_scale; i++) {
    
    TreeType **begin = (*cover_sets)[i].begin();
    TreeType **end = (*cover_sets)[i].end();
    for (; begin != end; begin++) {
      (*begin)->stat().pop_last_distance();
    }
  }
}

template<typename T>
void AllKNN<T>::CopyLeafNodes_(TreeType *query, 
			       ArrayList<T> *upper_bounds,
			       ArrayList<TreeType*> *leaf_nodes,
			       ArrayList<TreeType*> *new_leaf_nodes) {
  
  TreeType **begin = leaf_nodes->begin();
  TreeType **end = leaf_nodes->end();
  GenVector<T> q_point;
  T *query_upper_bound = &((*upper_bounds)[query->point() * knns_]);
  T query_max_dist = query->max_dist_to_grandchild();
  T query_parent_dist = query->dist_to_parent();

  new_leaf_nodes->Resize(0);
  queries_.MakeColumnVector(query->point(), &q_point);

  for (; begin != end; begin++) {
    T upper_bound = *query_upper_bound + query_max_dist;
    
    // first pruning on the basis of distance to its parent
    if ((*begin)->stat().distance_to_qnode() 
	- query_parent_dist <= upper_bound) {

      GenVector<T> r_point;
      references_.MakeColumnVector((*begin)->point(), &r_point);

      //T dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
      T d = pdc::DistanceEuclidean(q_point, r_point, upper_bound);

      // now pruning with its own upper bound
      if (d <= upper_bound) {

	// distance less than present upper bound, hence update
	if (d < *query_upper_bound) {
	  update_upper_bounds_(upper_bounds, query->point(), d);
	}

	(*begin)->stat().set_distance_to_qnode(d);
	new_leaf_nodes->PushBackCopy(*begin);
      }
    }
  }
}

template<typename T>
void AllKNN<T>::CopyCoverSets_(TreeType *query, 
			       ArrayList<T> *upper_bounds, 
			       ArrayList<ArrayList<TreeType*> > *cover_sets, 
			       ArrayList<ArrayList<TreeType*> > *new_cover_sets, 
			       index_t current_scale, index_t max_scale) {
  GenVector<T> q_point;
  T *query_upper_bound = &((*upper_bounds)[query->point() * knns_]);
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

      // first pruning on the distance to the parent
      if ((*begin)->stat().distance_to_qnode() 
	  - query_parent_dist <= upper_bound) {

	GenVector<T> r_point;
	references_.MakeColumnVector((*begin)->point(), 
				     &r_point);

	//T dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	T d = pdc::DistanceEuclidean<T>(q_point, r_point, upper_bound);

	// pruning on its own distance
	if (d <= upper_bound) {

	  // distance computed is less than present upper bound
	  // hence update
	  if (d < *query_upper_bound) {
	    update_upper_bounds_(upper_bounds, query->point(), d);
	  }

	  (*begin)->stat().set_distance_to_qnode(d);
	  (*new_cover_sets)[i].PushBackCopy(*begin);
	}
      }
    }
  }
}

template<typename T>
void AllKNN<T>::DescendTheRefTree_(TreeType *query, 
				   ArrayList<T> *upper_bounds, 
				   ArrayList<ArrayList<TreeType*> > *cover_sets,
				   ArrayList<TreeType*> *leaf_nodes,
				   index_t current_scale, index_t *max_scale) {

  TreeType **begin = (*cover_sets)[current_scale].begin();
  TreeType **end = (*cover_sets)[current_scale].end();

  T query_max_dist = query->max_dist_to_grandchild();
  T *query_upper_bound = &((*upper_bounds)[query->point() * knns_]);    

  GenVector<T> q_point;
  queries_.MakeColumnVector(query->point(), &q_point);
  //  NOTIFY("descend start at %"LI"d, max_scale now = %"LI"d, size: %"LI"d",
  // current_scale, *max_scale, (*cover_sets)[current_scale].size());
  index_t i = 0;
  for (; begin != end; begin++, i++) {
    //NOTIFY("Iteration %"LI"d", i);
    DEBUG_WARN_MSG_IF(current_scale != (*begin)->scale_depth(), 
		      "scales don't mathc, no good");

    // forming the upper bound for a particular reference node
    T upper_bound = *query_upper_bound 
      + query_max_dist + query_max_dist;
    // NOTIFY("LO");
    // NOTIFY("%"LI"d", (*begin)->point());
    T present_upper_dist = (*begin)->stat().distance_to_qnode();
    // NOTIFY("here");
    // pruning a reference node if it is no longer a 
    // source of potential NN for the query node and/or its 
    // descendants
    if (present_upper_dist <= upper_bound 
	+ (*begin)->max_dist_to_grandchild()) {
	
      TreeType **child = (*begin)->children()->begin();
      //   NOTIFY("!");
      // pruning condition for the self child
      if (present_upper_dist <= upper_bound 
	  + (*child)->max_dist_to_grandchild()) {
	  
	// if it is a non-leaf node, we add it to the 
	// cover set
	if ((*child)->num_of_children() > 0) {
	  
	  if (*max_scale < (*child)->scale_depth()) {
	    *max_scale = (*child)->scale_depth();
	  }

	  (*child)->stat().set_distance_to_qnode(present_upper_dist);
	  (*cover_sets)[(*child)->scale_depth()].PushBackCopy(*child);
	  //  NOTIFY("%"LI"d chosen at scale %"LI"d",
	  // (*child)->point(), (*child)->scale_depth());
	}
	// otherwise add it to the leaf set
	else {
	  if (present_upper_dist <= upper_bound) {
	    (*child)->stat().set_distance_to_qnode(present_upper_dist);
	    leaf_nodes->PushBackCopy(*child);
	    DEBUG_ASSERT((*child)->scale_depth() == 100);
	  }
	}
      }
      //   NOTIFY("!!");
      index_t j = 0;
      TreeType **child_end = (*begin)->children()->end();
      for (++child; child != child_end; child++, j++) {
	//	NOTIFY("sub-iteration: %"LI"d", j);

	T new_upper_bound = *query_upper_bound 
	  + (*child)->max_dist_to_grandchild() 
	  + query_max_dist + query_max_dist;

	if (present_upper_dist - (*child)->dist_to_parent()
	    <= new_upper_bound) {

	  GenVector<T> r_point;
	  references_.MakeColumnVector((*child)->point(), &r_point);

	  //T dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	  T d = pdc::DistanceEuclidean<T>(q_point, r_point, new_upper_bound);

	  if (d <= new_upper_bound) {

	    if (d < *query_upper_bound) {
	      update_upper_bounds_(upper_bounds, query->point(), d);
	    }
	     
	    if ((*child)->num_of_children() > 0) {
	      DEBUG_ASSERT((*child)->children()->size() > 0);
	      if (*max_scale < (*child)->scale_depth()) {
		*max_scale = (*child)->scale_depth();
	      }
	      (*child)->stat().set_distance_to_qnode(d);
	      (*cover_sets)[(*child)->scale_depth()].PushBackCopy(*child);
	      //    NOTIFY("%"LI"d child chosen for cover_sets at scale : %"LI"d", 
	      //     (*child)->point(), (*child)->scale_depth());
	    }
	    else {
	      if (d <= new_upper_bound - 
		  (*child)->max_dist_to_grandchild()) {
		(*child)->stat().set_distance_to_qnode(d);
		leaf_nodes->PushBackCopy(*child);
		//NOTIFY("%"LI"d chosen for leaf sets",
		//     (*child)->point());
		DEBUG_ASSERT((*child)->scale_depth() == 100);
	      }
	    }
	  }
	}
	//	NOTIFY("HERE");
      }
      //NOTIFY("hh");
    }
    //  NOTIFY("lo");
  }
  //  NOTIFY("Descend end at current scale:%"LI"d, max scale now:%"LI"d",
  // current_scale, *max_scale);
  //  NOTIFY("bye");
}

template<typename T>
void AllKNN<T>::ComputeBaseCase_(TreeType *query, 
				 ArrayList<T> *upper_bounds,
				 ArrayList<TreeType*> *leaf_nodes, 
				 ArrayList<index_t> *neighbor_indices) {
  
  // If this is not a leaf level of the query nodes, 
  // we traverse down the query nodes
  if (query->num_of_children() > 0) {
      
    TreeType **child = query->children()->begin();
    // Descending down the self child
    ComputeBaseCase_(*child, upper_bounds, 
		     leaf_nodes, neighbor_indices);
    T *query_upper_bound = &((*upper_bounds)[query->point() * knns_]);
    TreeType **child_end = query->children()->end();

    for (++child; child != child_end; child++) {

      ArrayList<TreeType*> new_leaf_nodes;

      new_leaf_nodes.Init(0);
      set_upper_bounds_(upper_bounds, (*child)->point(), 
			*query_upper_bound 
			+ (*child)->dist_to_parent());

      // copying all the references leaves which are within
      // the upper_bound of the query child
      CopyLeafNodes_(*child, upper_bounds, 
		     leaf_nodes, &new_leaf_nodes);

      // descend down the child
      ComputeBaseCase_(*child, upper_bounds, 
		       &new_leaf_nodes, neighbor_indices);

      // This is done so that the new child at the same level 
      // can use these reference nodes
      reset_leaf_nodes_(&new_leaf_nodes);
    }
  }
  // if the query node is also a leaf, select all the 
  // points in the leaf set that are the kNN
  else {


    TreeType **begin_nn = leaf_nodes->begin();
    TreeType **end_nn = leaf_nodes->end();
    T *begin = upper_bounds->begin() + query->point() * knns_;
    T *end = upper_bounds->begin() + query->point() * knns_ + knns_ ;
    index_t *indices = neighbor_indices->begin() + query->point() * knns_;
    ArrayList<bool> flags;

    flags.Init(knns_);
    for (index_t i = 0; i < knns_; i++) {
      flags[i] = 0;
    }

    // Here we sort so as to store the NN in the ascending order
    // instead of descending
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

template<typename T>
void AllKNN<T>::ComputeNeighborRecursion_(TreeType *query, 
					  ArrayList<T> *upper_bounds, 
					  ArrayList<ArrayList<TreeType*> > *cover_sets,
					  ArrayList<TreeType*> *leaf_nodes,
					  index_t current_scale, index_t max_scale,
					  ArrayList<index_t> *neighbor_indices) {
    
  // if there is no more reference nodes to descend
  // we just do base case where we just descend down 
  // the query tree
  if (current_scale > max_scale) {
    ComputeBaseCase_(query, upper_bounds, leaf_nodes, neighbor_indices);
  }
  else {
    // if the query tree is at a higher scale than the reference tree
    // we descend the query tree
    if ((query->scale_depth() <= current_scale) 
	&& (query->scale_depth() != 100)) {

      // we can descend the self child first and get lower 
      // upper bounds for the other children, but we will
      // do that later
      TreeType **child = query->children()->begin();
      TreeType **child_end = query->children()->end();
      T *query_upper_bound = &((*upper_bounds)[query->point() * knns_]);

      for (++child; child != child_end; child++) {

	ArrayList<TreeType*> new_leaf_nodes;
	ArrayList<ArrayList<TreeType*> > new_cover_sets;
	new_leaf_nodes.Init(0);

	// setting the upper bound for a query child on the 
	// basis of the upper bounds of the parent
	set_upper_bounds_(upper_bounds, (*child)->point(), 
			  *query_upper_bound 
			  + (*child)->dist_to_parent());
	  
	// copying reference node sets for the query child
	// for the query descends
	CopyLeafNodes_(*child, upper_bounds, 
		       leaf_nodes, &new_leaf_nodes);
	//	NOTIFY("CCS");
	CopyCoverSets_(*child, upper_bounds, 
		       cover_sets, &new_cover_sets,
		       current_scale, max_scale);

	// descending down the query tree
	ComputeNeighborRecursion_(*child, upper_bounds, 
				  &new_cover_sets,
				  &new_leaf_nodes, 
				  current_scale, 
				  max_scale, 
				  neighbor_indices);

	// resetting the reference sets so that we can use 
	// them for the other children
	reset_leaf_nodes_(&new_leaf_nodes);
	reset_cover_sets_(&new_cover_sets, current_scale, max_scale);
      }

      // Descending the self child of the query node
      ComputeNeighborRecursion_(query->child(0), 
				upper_bounds, cover_sets, 
				leaf_nodes, current_scale,
				max_scale, neighbor_indices);
    }
    else {

      // halfsorting the cover set to descend the closer reference
      // nodes earlier and prune away more reference nodes later
      halfsort_(&((*cover_sets)[current_scale]));
	
      // Descend the reference nodes with respect to this query 
      // node
      //    NOTIFY("DRT");
      DescendTheRefTree_(query, upper_bounds, cover_sets, 
			 leaf_nodes, current_scale, &max_scale);

      // increasing the scale to go to the next level
      ++current_scale;
	
      // traversing down the query tree now
      ComputeNeighborRecursion_(query, upper_bounds,
				cover_sets, leaf_nodes, 
				current_scale, max_scale, 
				neighbor_indices);
    }
  }

}

template<typename T>
void AllKNN<T>::DepthFirst_(TreeType *query, TreeType *reference, 
			    ArrayList<ArrayList<LeafNodes> > *neighbors,
			    ArrayList<T> *upper_bounds) { 
  
  // one of the problems is that some nodes are completely pruned 
  // and you don't ever encounter them even though they maybe your NN

  // next you need to make sure the copy of the possible NN set 
  // is not repeated over and over again

  // first change set_neighbor_distances to update_new which replaces 
  // all > parent_ub + parent_dist with the same.

  // second thing is that whenever it comes for the first time 
  // from the reference descend, save the neighbors and use a 
  // new set for the next cycle

  DEBUG_ASSERT(query != NULL);
  DEBUG_ASSERT(reference != NULL);
  T *query_upper_bound = upper_bounds->begin() + query->point() * knns_;
  T reference_query_dist = reference->stat().distance_to_qnode();
  GenVector<T> q, r;
  queries_.MakeColumnVector(query->point(), &q);
  references_.MakeColumnVector(reference->point(), &r);
  // Descend the query tree here and add points which are not 
  // leaves in the cover set and the points that are leaves 
  // are added to the zero sets 

  // if the cover set size becomes zero, no longer descend the 
  // reference node, just descend query tree -> make reference==NULL
  // now if query is a leaf, copy zero set and takes elements which are 
  // within query_upper_bound
  if (query->is_leaf() && reference->is_leaf()) {

    // the base case or something close to that (should try 
    // the traditional base case
    // though later
 
    //NOTIFY("leaf node for Point %"LI"d %"LI"d",
    //	     query->point()+1, reference->point()+1);
    if (reference_query_dist <= *query_upper_bound){
      LeafNodes leaf; 
      leaf.Init(reference->point(), reference_query_dist);
      (*neighbors)[query->point()].PushBackCopy(leaf);
    }
  }
  else if (reference->is_leaf()) {

    // reached the leaf node of the reference node 
    // descending the query tree

    TreeType **qchild = query->children()->begin();
    //NOTIFY("Ref leaf: self child point %"LI"d %"LI"d",
    //     (*qchild)->point()+1, reference->point()+1);
    DepthFirst_(*qchild, reference, 
		neighbors, upper_bounds);

    TreeType **child_end = query->children()->end();
    for (++qchild; qchild != child_end; qchild++) {

      set_update_upper_bounds_(upper_bounds, (*qchild)->point(), 
			       *query_upper_bound 
			       + (*qchild)->dist_to_parent());

      GenVector<T> qrs;
      queries_.MakeColumnVector((*qchild)->point(), &qrs);
      T *qchild_ub =  upper_bounds->begin() + (*qchild)->point() * knns_;

      T new_upper_bound = *qchild_ub + 2 * (*qchild)->max_dist_to_grandchild();
	
      if (reference_query_dist 
	  - (*qchild)->dist_to_parent() 
	  <= new_upper_bound) {
	 
	T d = pdc::DistanceEuclidean(qrs, r, new_upper_bound);
	if (d <= new_upper_bound) {
	  if (d < *qchild_ub) {
	    //NOTIFY("UPDATE!!");
	    update_upper_bounds_(upper_bounds, 
				 (*qchild)->point(), d);
	  }
	  reference->stat().set_distance_to_qnode(d);
	  //NOTIFY("Ref leaf: child point %"LI"d %"LI"d",
	  //	   (*qchild)->point()+1, reference->point()+1);
	  DepthFirst_(*qchild, reference, 
		      neighbors, upper_bounds);
	  // &new_leaf_nodes);
	  //	reset_new_leaf_nodes(&new_leaf_nodes);
	  reference->stat().pop_last_distance();
	}
      }
    }
  }
  else {
    // descend both trees
    // first descend the reference tree choosing those nodes that 
    // are within the required distance
    // after that descend the query tree and do 
    // copy_cover_sets and copy_leaf_nodes

    ArrayList<TreeType*> ref_set;
    ref_set.Init(0);
      
    // after everything, we need to reset this cover set by removing all 
    // the last distance to qnode for every element in the cover set
    // don't know what to do with the leaf list 
    T query_max_dist = query->max_dist_to_grandchild();

    T upper_bound = *query_upper_bound 
      + query_max_dist + query_max_dist;

    T present_upper_dist = reference_query_dist;

    //NOTIFY("q=%"LI"d r=%"LI"d", 
    //     query->point()+1, reference->point()+1);

    if (present_upper_dist <= upper_bound 
	+ reference->max_dist_to_grandchild()) {
	
      //NOTIFY("Initial pruning criteria passed!");
      TreeType **rchild = reference->children()->begin();

      if (present_upper_dist <= upper_bound 
	  + (*rchild)->max_dist_to_grandchild()) {
	// NOTIFY("self child selected!!");
	(*rchild)->stat().set_distance_to_qnode(present_upper_dist);
	ref_set.PushBackCopy(*rchild);
	//printf("%"LI"d ", (*rchild)->point()+1);
      }

      TreeType **child_end = reference->children()->end();
      for (++rchild; rchild != child_end; rchild++) {

	T new_upper_bound = *query_upper_bound 
	  + (*rchild)->max_dist_to_grandchild() 
	  + query_max_dist + query_max_dist;

	if ((*rchild)->is_leaf()) {
	  DEBUG_ASSERT((*rchild)->max_dist_to_grandchild() == 0.0);
	}

	if (present_upper_dist - (*rchild)->dist_to_parent()
	    <= new_upper_bound) {
	  //NOTIFY("child selected for dist comp");
	  GenVector<T> refs;
	  references_.MakeColumnVector((*rchild)->point(), &refs);

	  //T dist = sqrt(la::DistanceSqEuclidean(q_point, r_point));
	  T d = pdc::DistanceEuclidean<T>(q, refs, new_upper_bound);

	  if (d <= new_upper_bound) {
	    // NOTIFY("child selected");
	    if (d < *query_upper_bound) {
	      //NOTIFY("UPDATED!!");
	      update_upper_bounds_(upper_bounds,
				   query->point(), d);
	    }
	     
	    (*rchild)->stat().set_distance_to_qnode(d);
	    ref_set.PushBackCopy(*rchild);
	    //   printf("%"LI"d ", (*rchild)->point()+1);
	  }
	}
      }
    }
    //      printf("\n");
    if (ref_set.size() == 0) {
      return;
    }
    // sorting the nodes here according to their closeness to the 
    // query point's self child
    halfsort_(&ref_set);
    TreeType **rbegin = ref_set.begin();
    TreeType **rend = ref_set.end();
      
    if (query->is_leaf()) {
      DEBUG_ASSERT(query->max_dist_to_grandchild() == 0.0);
      for (TreeType **begin = rbegin; begin != rend; begin++) {
	if ((*begin)->stat().distance_to_qnode() <= *query_upper_bound
	    + (*begin)->max_dist_to_grandchild()) {
	  //NOTIFY("Query leaf: point %"LI"d %"LI"d", 
	  // query->point()+1, (*begin)->point()+1);
	  DepthFirst_(query, *begin, 
		      neighbors, upper_bounds);
	}
      }
    }
    else {

      TreeType **qchild = query->children()->begin();
      TreeType **qend = query->children()->end();

      for (TreeType **begin = rbegin; begin != rend; begin++) {
// 	if ((*begin)->stat().distance_to_qnode() <= *query_upper_bound 
// 	    + (*begin)->max_dist_to_grandchild() 
// 	    + (*qchild)->max_dist_to_grandchild()){
	  //NOTIFY("Query: self child point %"LI"d %"LI"d",
	  //	   (*qchild)->point()+1, (*begin)->point()+1);
	DepthFirst_(*qchild, *begin, 
		    neighbors, upper_bounds);
// 	}
      }

      for (++qchild; qchild != qend; qchild++) {
	set_update_upper_bounds_(upper_bounds, (*qchild)->point(), 
				 *query_upper_bound 
				 + (*qchild)->dist_to_parent());

	GenVector<T> qrs;
	queries_.MakeColumnVector((*qchild)->point(), &qrs);
	T *qchild_ub =  upper_bounds->begin() + (*qchild)->point() * knns_;
	T qchild_max_dist = (*qchild)->max_dist_to_grandchild();
	T qchild_parent_dist = (*qchild)->dist_to_parent();

	// add equivalent here
	//NOTIFY("ref set %"LI"d", ref_set.size());
	for (TreeType **begin = rbegin; begin != rend; begin++) {
	  T new_upper_bound = *qchild_ub + qchild_max_dist 
	    + qchild_max_dist 
	    + (*begin)->max_dist_to_grandchild();

	  //NOTIFY("%"LI"d -> %"LI"d", 
	  //	   (*qchild)->point()+1, (*begin)->point()+1);
	  if ((*begin)->stat().distance_to_qnode() 
	      - qchild_parent_dist <= new_upper_bound) {
	    GenVector<T> refs;
	    references_.MakeColumnVector((*begin)->point(), &refs);
	    //NOTIFY("here");
	    T d = pdc::DistanceEuclidean(qrs, refs, new_upper_bound);
	    //NOTIFY("nub = %lf, dist = %lf", new_upper_bound, dist);
	    if (d <= new_upper_bound) {
	      if (d < *qchild_ub) {
		//NOTIFY("updated!!");
		update_upper_bounds_(upper_bounds, 
				     (*qchild)->point(), d);
	      }
	      (*begin)->stat().set_distance_to_qnode(d);
	      //NOTIFY("Query: child point %"LI"d %"LI"d",
	      //     (*qchild)->point()+1, (*begin)->point()+1);
	      DepthFirst_(*qchild, *begin, 
			  neighbors, upper_bounds);
	      // &new_leaf_nodes);
	      //	reset_new_leaf_nodes(&new_leaf_nodes);
	      (*begin)->stat().pop_last_distance();
	      //NOTIFY("return %"LI"d", (*qchild)->point()+1);
	    }
	  }
	  //NOTIFY("loop end");
	}
      }
    }
  }
}

#endif
