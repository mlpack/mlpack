#ifndef TREE_COVER_TREE_IMPL_H
#define TREE_COVER_TREE_IMPL_H
#include "ctree.h"

void ctree::print_space(index_t n) {
  for (index_t i = 0; i < n; i++) {
    printf("\t");
  }
  return;
}

template<typename TCoverTreeNode>
void ctree::print_tree(index_t depth, TCoverTreeNode *top_node) {
  print_space(depth);
  printf("Point %"LI"d:", top_node->point()+1);
  if (top_node->num_of_children() > 0) {
    printf("scale_depth = %"LI"d, max_dist = %lf, children = %"LI"d\n",
           top_node->scale_depth(), top_node->max_dist_to_grandchild(),
           top_node->num_of_children());
    for (index_t i = 0; i < top_node->num_of_children(); i++) {
      print_tree(depth+1, top_node->child(i));
    }
  }
  else {
    printf("\n");
  }
  return;
}

template<typename T>
T ctree::max_set(ArrayList<NodeDistances<T>*> *set, index_t *point) {
  
  T max = 0.0;
  for (index_t i = 0; i < set->size(); i++) {
    if(max < (*set)[i]->distances()->back()) {
      max = (*set)[i]->distances()->back();
      if (point != NULL) {
	*point = i;
      }
    }
  }
  return max;
}


template<typename T>
void ctree::split_far(ArrayList<NodeDistances<T>*> *point_set, 
		      ArrayList<NodeDistances<T>*> *far,
		      index_t scale) {
 
  T bound = scaled_distance<T>(scale);
  index_t initial_size = far->size();
  ArrayList<NodeDistances<T>*> near;
  NodeDistances<T> **begin = point_set->begin();
  NodeDistances<T> **end = point_set->end();

  near.Init(0);
  for (; begin < end; begin++) {
    if ((*begin)->distances()->back() > bound) {
      far->PushBackCopy(*begin);
    }
    else {
      near.PushBackCopy(*begin);
    }
  }

  DEBUG_ASSERT_MSG(point_set->size() == 
		   far->size() - initial_size + near.size(), 
		   "split_far: point set size doesn't add up\n");

  point_set->Renew();
  point_set->InitSteal(&near);

  return;
}

// here we assume that the point_set and the near set are 
// already initialized
template<typename T>
void ctree::split_near(index_t point, const GenMatrix<T>& data,
		       ArrayList<NodeDistances<T>*> *point_set,
		       ArrayList<NodeDistances<T>*> *near,
		       index_t scale) {
 
  T bound = scaled_distance<T>(scale);
  index_t initial_size = near->size();
  ArrayList<NodeDistances<T>*> far;
  NodeDistances<T> **begin = point_set->begin();
  NodeDistances<T> **end = point_set->end();
  GenVector<T> p;

  data.MakeColumnVector(point, &p);
  far.Init(0);
  for (; begin < end; begin++) {

    GenVector<T> q;

    data.MakeColumnVector((*begin)->point(), &q);
    //T dist = sqrt(la::DistanceSqEuclidean(p,q));
    T dist = pdc::DistanceEuclidean<T>(p, q, bound);
    if (dist > bound) {
      far.PushBackCopy(*begin);
    }
    else {
      (*begin)->add_distance(dist);
      near->PushBackCopy(*begin);
    }
  }

  DEBUG_ASSERT_MSG(point_set->size() == 
		   near->size() - initial_size + far.size(),
		   "split_near: point set doesn't add up\n");

  point_set->Renew();
  point_set->InitSteal(&far);

  return;
}

template<typename TCoverTreeNode, typename T>
TCoverTreeNode *ctree::private_make_tree(index_t point, 
					 const GenMatrix<T>& data,
					 index_t current_scale,
					 index_t max_scale, 
					 ArrayList<NodeDistances<T>*> *point_set,
					 ArrayList<NodeDistances<T>*> *consumed_set) {
  
  // no other point so leaf in explicit tree
  if (point_set->size() == 0) { 
    TCoverTreeNode *node = new TCoverTreeNode();
    node->MakeLeafNode(point);
    return node;
  }
  else {
    T max_dist = max_set(point_set);
    index_t next_scale = min(current_scale - 1, 
			     scale_of_distance(max_dist));
      
    // At the -INF level so all points are nodes
    // and we have point with zero distances
    if (next_scale == NEG_INF) { 
      ArrayList<TCoverTreeNode*> children;
      NodeDistances<T> **begin = point_set->begin();
      NodeDistances<T> **end = point_set->end();

      children.Init(0);
      TCoverTreeNode *self_node = new TCoverTreeNode();
      self_node->MakeLeafNode(point);
      children.PushBackCopy(self_node);

      for (; begin < end; begin++) {
	TCoverTreeNode *node = new TCoverTreeNode();
	node->MakeLeafNode((*begin)->point());
	children.PushBackCopy(node);
	consumed_set->PushBackCopy(*begin);
      }

      DEBUG_ASSERT(children.size() == point_set->size()+1);
      point_set->Resize(0);
      TCoverTreeNode *node = new TCoverTreeNode();
      node->MakeNode(point, 0.0, 100, &children);

      return node;
    }

    // otherwise you need to recurse
    else {

      ArrayList<NodeDistances<T>*> far;

      far.Init(0);
      split_far(point_set, &far, current_scale);
	
      TCoverTreeNode *child = 
	private_make_tree<TCoverTreeNode>(point, data, next_scale, 
					  max_scale, point_set, 
					  consumed_set);
	
      if (point_set->size() == 0) {
	point_set->Renew();
	point_set->InitSteal(&far);

	return child;
      }

      else {

	ArrayList<TCoverTreeNode*> children;
	ArrayList<NodeDistances<T>*> new_point_set, new_consumed_set;

	children.Init(0);
	new_point_set.Init(0);
	new_consumed_set.Init(0);
	children.PushBackCopy(child);
	//NOTIFY("%"LI"d:%"LI"d", child->point()+1, child->scale_depth());

	while (point_set->size() != 0) {

	  index_t new_point;
	  T new_dist = point_set->back()->distances()->back();

	  new_point = point_set->back()->point();
	  // remember to check here what to use, PushBackRaw() or AddBack()
	  // so that we can use PopBackInit(Element *dest)
	  point_set->PopBackInit(consumed_set->PushBackRaw()); 

	  split_near(new_point, data, point_set, 
		     &new_point_set, current_scale);
	  split_near(new_point, data, &far, 
		     &new_point_set, current_scale);
	    
	  TCoverTreeNode *child_node = 
	    private_make_tree<TCoverTreeNode>(new_point, data,
					      next_scale,max_scale,
					      &new_point_set,
					      &new_consumed_set);
	    
	  child_node->set_dist_to_parent(new_dist);
	  children.PushBackCopy(child_node);
	  //NOTIFY("%"LI"d:%"LI"d", child_node->point()+1, child_node->scale_depth());
	    
	  T bound = scaled_distance<T>(current_scale);
	  NodeDistances<T> **begin = new_point_set.begin();
	  NodeDistances<T> **end = new_point_set.end();

	  for (; begin < end; begin++) {

	    (*begin)->distances()->PopBack();
	    if ((*begin)->distances()->back() <= bound) {
	      point_set->PushBackCopy(*begin);
	    }
	    else {
	      far.PushBackCopy(*begin);
	    }
	  }
	  new_point_set.Resize(0);
	    
	  while (new_consumed_set.size() > 0) {
	    new_consumed_set.back()->distances()->PopBack();
	    new_consumed_set.PopBackInit(consumed_set->PushBackRaw());
	  }
	   
	}

	point_set->Renew();
	point_set->InitSteal(&far);

	TCoverTreeNode *node = new TCoverTreeNode();

	node->MakeNode(point, max_set(consumed_set), 
		       max_scale - current_scale, &children);
	  
	return node;
      }
    }
  }
}

#endif
