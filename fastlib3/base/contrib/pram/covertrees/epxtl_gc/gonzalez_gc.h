#ifndef GONZALEZ_GC_H
#define GONZALEZ_GC_H

#include <fastlib/fastlib.h>
#include "distances_gc.h"
#include "cover_tree_gc.h"

namespace gc {

  double EC = 1.3;

  template<typename T> 
  inline index_t scale_of_distance(T distance) {
    return (index_t) ceil(log(distance) / log((T)EC));
  }

  template<typename T>
  class Point {
  
  private:
    index_t point_;
    T alpha_k_;
    index_t center_present_;
    index_t center_1_;
    index_t center_2_;
    index_t center_at_last_level_;
    T dist_at_last_level_;
  
  public:
    Point() {
    }

    ~Point() {
    }

    void set_alpha_k(T alpha_k) {
      alpha_k_ = alpha_k;
    }

    void set_center(index_t center) {
      center_2_ = center_1_;
      center_1_ = center_present_;
      center_present_ = center;
    }

    void set_point(index_t point) {
      point_ = point;
    }

    void set_center_at_last_level(index_t center) {
      center_at_last_level_ = center;
    }

    void set_dist_at_last_level(T dist) {
      dist_at_last_level_ = dist;
    }

    index_t center() {
      return center_present_;
    }

    index_t center_2() {
      return center_2_;
    }

    T alpha_k() {
      return alpha_k_;
    }

    index_t point() {
      return point_;
    }

    index_t center_at_last_level() {
      return center_at_last_level_;
    }

    T dist_at_last_level() {
      return dist_at_last_level_;
    }

    void Init(index_t point, T alpha_k, index_t center) {
      point_ = point;
      alpha_k_ = alpha_k;
      center_present_ = center;
      center_1_ = -1;
      center_2_ = -1;
    }

  };

  template<typename T> 
  class Center {

  private:
    index_t point_;
    index_t index_;
    ArrayList<Center*> friends_list_;
    ArrayList<T> friends_list_distances_;
    ArrayList<Center *> children;
    T r_i_;
    T r_k_;
    T radius_;
    index_t farthest_;
    index_t num_points_;
    ArrayList<Point<T> *> points_served_;
    index_t parent_;
    T distance_to_parent_;

  public:

    Center() {
      friends_list_.Init(0);
      friends_list_distances_.Init(0);
      points_served_.Init(0);
      parent_ = -1;
    }

    ~Center() {
    }

    void set_r_i(T r_i) {
      r_i_ = r_i;
    }

    void set_r_k(T r_k) {
      r_k_ = r_k;
    }

    void set_farthest(index_t farthest) {
      farthest_ = farthest;
    }

    void add_friend(Center<T> *frnd, T friend_distance) {
      friends_list_.PushBackCopy(frnd);
      friends_list_distances_.PushBackCopy(friend_distance);
    }

    void set_num_points(index_t num_points) {
      num_points_ = num_points;
    }

    void set_point(index_t point) {
      point_ = point;
    }

    void set_index(index_t index) {
      index_ = index;
    }

    void set_radius(T radius) {
      radius_ = radius;
    }

    void set_parent(index_t parent) {
      parent_ = parent;
    }

    void set_distance_to_parent(T dist) {
      distance_to_parent_ = dist;
    }

    ArrayList<Center *> *friends_list() {
      return &friends_list_;
    }

    ArrayList<T> *friends_list_distances() {
      return &friends_list_distances_;
    }

    ArrayList<Point<T> *> *points_served() {
      return &points_served_;
    }

    Point<T> *points_served(index_t i) {
      return points_served_[i];
    }

    T r_i() {
      return r_i_;
    }

    T r_k() {
      return r_k_;
    }

    T radius() {
      return radius_;
    }

    index_t num_points() {
      return num_points_;
    }

    index_t farthest() {
      return farthest_;
    }

    index_t point() {
      return point_;
    }

    index_t index() {
      return index_;
    }

    index_t parent() {
      return parent_;
    }

    T distance_to_parent() {
      return distance_to_parent_;
    }

    void Init(index_t point, T radius, index_t index) {
      point_ = point;
      r_i_ = radius;
      r_k_ = radius;
      radius_ = radius;
      index_ = index;
      parent_ = -1;
      distance_to_parent_ = 0.0;
    }
  };

  // this splits the clusters for the new cluster center 
  // according to it being a member of the new cluster 
  // or the old one
  template<typename T> 
  void split(Point<T> *p, index_t center_index,
	     ArrayList<Point<T> *> *set, 
	     ArrayList<Point<T> *> *new_set, 
	     GenMatrix<T>& data, T upper_bound) {

    index_t initial_size = new_set->size();
    ArrayList<Point<T> *> far;
    far.Init(0);

    Point<T> **begin = set->begin();
    Point<T> **end = set->end();

    GenVector<T> pv; 
    data.MakeColumnVector(p->point(), &pv);

    for(; begin != end; begin++) {
      GenVector<T> q;
      data.MakeColumnVector((*begin)->point(), &q);

      T dist = pdc::DistanceEuclidean<T>(pv, q, upper_bound);
      if (dist < (*begin)->alpha_k()) {
	(*begin)->set_alpha_k(dist);
	(*begin)->set_center(center_index);
	new_set->PushBackCopy(*begin);
      }
      else {
	(*begin)->set_center((*begin)->center());
	far.PushBackCopy(*begin);
      }
    }
    DEBUG_ASSERT(initial_size + set->size() 
		 == new_set->size() + far.size());

    set->Renew();
    set->InitSteal(&far);
  }

  // update the farthest point for a center after splitting 
  // occurs and it looses some points :-)
  template<typename T>
  void update_farthest(Center<T> *c) {

    if (c->points_served()->size() == 0) {
      c->set_farthest(-1);
      c->set_radius(-1.0);
    }
    else {

      Point<T> **begin = c->points_served()->begin();
      Point<T> **end = c->points_served()->end();
      T max = 0;
      index_t max_index = -1;

      for(index_t i = 0; begin != end; begin++, i++) {
	if ((*begin)->alpha_k() >= max) {
	  max = (*begin)->alpha_k();
	  max_index = i;
	}
      }
      DEBUG_ASSERT(max_index != -1);
      c->set_farthest(max_index);
      c->set_radius(max);
    }
  }

  // finds the point which is the candidate for becoming 
  // the next cluster center
  template<typename T>
  void find_max(ArrayList<Point<T>*> *set, Point<T> **point) {

    Point<T> **begin = set->begin();
    Point<T> **end = set->end();
    T max = 0.0;
    bool flag = 0;

    for(; begin != end; begin++) {
      if ((*begin)->alpha_k() >= max) {
	max = (*begin)->alpha_k();
	*point = *begin;
	flag = 1;
      }
    }
    DEBUG_ASSERT(flag == 1);
  }

  // finds the maximum radius using the farthest set
  template<typename T>
  T find_max_radius(ArrayList<Point<T> *> *set) {

    Point<T> **begin = set->begin();
    Point<T> **end = set->end();
    T max = 0.0;

    for(; begin != end; begin++) {
      if ((*begin)->alpha_k() >= max) {
	max = (*begin)->alpha_k();
      }
    }
    return max;
  }

  // assign centers to points when we are at a level
  // of the cover tree
  template<typename T>
  void set_parents(Center<T> *center) {
    Point<T> **begin = center->points_served()->begin();
    Point<T> **end = center->points_served()->end();

    for (; begin != end; begin++) {
      (*begin)->set_center_at_last_level((*begin)->center());
      (*begin)->set_dist_at_last_level((*begin)->alpha_k());
    }
  }

  // the helper function
  template<typename T, typename TreeType>
  T FindMaxDistGrandChild(TreeType *root, 
			  ArrayList<index_t> *point,
			  GenMatrix<T>& data) {

    if (root->is_leaf()) {
      point->PushBackCopy(root->point());
      root->set_max_dist_to_grandchild(0.0);
      return 0;
    }
    else {
      index_t node_point = root->point();
      ArrayList<index_t> subtree;
      subtree.Init(0);
      T max = 0.0;
      TreeType **begin = root->children()->begin();
      TreeType **end = root->children()->end();
      GenVector<T> p;
      data.MakeColumnVector(node_point, &p);

      for (; begin != end; begin++) {
	T child_max = FindMaxDistGrandChild(*begin, &subtree, data);
	if (max < child_max + (*begin)->dist_to_parent()) {
	  for (index_t *b = subtree.begin(), *e = subtree.end();
	       b != e; b++) { 
	    GenVector<T> q;
	    data.MakeColumnVector(*b, &q);
	    T dist = pdc::DistanceEuclidean<T>(p, q, DBL_MAX);
	    if (dist > max) {
	      max = dist;
	    }
	  }
	}
	while(subtree.size() > 0) {
	  subtree.PopBackInit(point->PushBackRaw());
	}
      }
      root->set_max_dist_to_grandchild(max);
      return max;
    }
  } 
					     
  template<typename T, typename TreeType>
  void update_scale(TreeType *root, index_t max_scale, 
		    index_t current_scale) {

    if (root->is_leaf()) {
      root->set_scale_depth(100);
    }
    else {
      index_t next_scale = min(current_scale - 1, 
			       scale_of_distance<T>(root->max_dist_to_grandchild()));
      if (next_scale == ((index_t) log(0.0))) {
	next_scale = max_scale - 100;
      }

      TreeType **begin = root->children()->begin();
      TreeType **end = root->children()->end();

      for (; begin != end; begin++) {
	update_scale<T, TreeType>(*begin, max_scale, next_scale);
      }

      root->set_scale_depth(max_scale - current_scale);
    }
  }
 
  
  // updating the tree so as to obtain the max_dist_to_grandchild thing
  template<typename T, typename TreeType>
  void update_tree(TreeType *root, 
		   GenMatrix<T>& data) { 
    ArrayList<index_t> points_subtree;
    points_subtree.Init(0);
    FindMaxDistGrandChild(root, &points_subtree, data);
  }

  // Gonzalez algorithm
  template<typename T, typename TreeType>
  TreeType* Gonzalez(GenMatrix<T>& dataset, index_t k,
		     ArrayList<Center<T> *> *centers) {

    ArrayList<Point<T> *> points;
    ArrayList<Point<T> *> farthest_set;
    index_t n = dataset.n_cols();
    ArrayList<TreeType *> tree;

    centers->Init(0);
    points.Init(0);
    farthest_set.Init(0);
    tree.Init(0);

    Center<T> *p1 = new Center<T>();
    GenVector<T> p;
    dataset.MakeColumnVector(0, &p);

    T max = 0.0;
    index_t max_ind = -1;

    for (index_t i = 1; i < n; i++) {
      GenVector<T> q;
      dataset.MakeColumnVector(i, &q);

      T dist = pdc::DistanceEuclidean<T>(p, q, DBL_MAX);
      Point<T> *q1 = new Point<T>();
      q1->Init(i, dist, 0);
      points.PushBackCopy(q1);
      if (dist > max) {
	max = dist; 
	max_ind = i-1;
      }
    }

    DEBUG_ASSERT(points.size() == n - 1);

    index_t max_scale = scale_of_distance<T>(max);

    farthest_set.PushBackCopy(points[max_ind]);

    p1->Init(0, max, 0);
    p1->set_farthest(max_ind);
    p1->set_num_points(points.size());
    p1->points_served()->Renew();
    p1->points_served()->InitSteal(&points);

    centers->PushBackCopy(p1);

    T r_0 = max;
    T j = 1.0;
    TreeType *root;

    // The clustering loop
    while (centers->size() < k + 1) {

      T r_k_1 = find_max_radius(&farthest_set);

      if ((r_k_1 <= r_0/j) || (centers->size() == k)) {

	index_t initial_size = tree.size();
	j *= (T)EC;

	Center<T> **begin = centers->begin();
	Center<T> **end = centers->end();

	for (index_t i = 0; begin != end; begin++, i++) {
	  set_parents(*begin);
	  TreeType *node = new TreeType();
	  if (initial_size == 0) {
	    DEBUG_ASSERT((*begin)->parent() == -1);
	    node->set_point((*begin)->point());
	    node->set_max_dist_to_grandchild(max);
	    node->set_scale_depth(0);
	    DEBUG_ASSERT(tree.size() == 0);
	    tree.PushBackCopy(node);
	    root = node;
	    DEBUG_ASSERT(tree.size() - 1 
			 == (*begin)->index());
	    (*begin)->set_parent((*begin)->index());
	  }
	  else {
	    if ((*begin)->parent() != -1) {
	      if ((*begin)->num_points() == 0) {
		// this is a leaf node, no need to do 
		// anything with it anymore
		// so set parent to -1
		
		// the scale might be set somewhere else
		node->set_scale_depth(100);
		node->set_num_of_children(0);
		if ((*begin)->parent() == (*begin)->index()) {
		  DEBUG_ASSERT(tree[(*begin)->index()]->point() 
			       == (*begin)->point());
		  node->set_dist_to_parent(0.0);
		  node->set_point((*begin)->point());
		}
		else {
		  node->set_dist_to_parent((*begin)->distance_to_parent());
		  node->set_point((*begin)->point());
		  tree.PushBackCopy(node);
		  
		  DEBUG_ASSERT_MSG(tree.size() - 1 
				   == (*begin)->index(), 
				   "tree:%"LI"d center index:%"LI"d",
				   tree.size(), (*begin)->index());
		  
		}
		tree[(*begin)->parent()]->children()->PushBackCopy(node);
		(*begin)->set_parent(-1);
	      }
	      else {
		if ((*begin)->parent() == (*begin)->index()) {
		  // forming self child
		  DEBUG_ASSERT_MSG(tree[(*begin)->index()]->point() 
				   == (*begin)->point(), "%"LI"d -> %"LI"d",
				   tree[(*begin)->index()]->point(),
				   (*begin)->point());
		  node->set_dist_to_parent(0.0);
		  node->set_point((*begin)->point());
		  tree[(*begin)->parent()]->children()->PushBackCopy(node);
		}
		else {
		  // forming new child
		  node->set_dist_to_parent((*begin)->distance_to_parent());
		  node->set_point((*begin)->point());
		  tree.PushBackCopy(node);
		  
		  DEBUG_ASSERT_MSG(tree.size() - 1 
				   == (*begin)->index(), 
				   "tree:%"LI"d center index:%"LI"d",
				   tree.size(), (*begin)->index());
		  // ending it with making its new parent itself
		  tree[(*begin)->parent()]->children()->PushBackCopy(node);
		  (*begin)->set_parent((*begin)->index());
		}
	      }
	    }
	  }
	}


	for (index_t j = 0; j < initial_size; j++) {
	  index_t child_size = tree[j]->children()->size();
	  if (child_size == 0) {
	    DEBUG_ASSERT(tree[j]->is_leaf());
	  }
	  else {
	    if (child_size == 1) {
	      DEBUG_ASSERT(tree[j]->point() 
			   == tree[j]->child(0)->point());
	      tree[j]->children()->PopBack();
	      tree[j]->set_num_of_children(0);
	    }
	    else {
	      tree[j]->set_num_of_children(child_size);
	      tree[j] = tree[j]->child(0);
	    }
	  }
	}
	if (centers->size() == k) {
	  for (index_t j = 0; j < centers->size(); j++) {
	    DEBUG_ASSERT((*centers)[j]->num_points() == 0);
	  }
	  break;
	}
      }
      
      // choosing the new_center from the farthest_set
      Point<T> *new_center;
      find_max(&farthest_set, &new_center);
      
      // the new_point's cluster center at the end of the 
      // (k-1) round
      Center<T> *c_pk = (*centers)[new_center->center()];

      //removing the chosen point from the c_pk's cluster
      c_pk->points_served()->Remove(c_pk->farthest());

      // scan all the points in the c_pk's cluster and 
      // friends_list(c_pk)'s cluster and update stuff
      ArrayList<Point<T> *> new_points_served;
      new_points_served.Init(0);
 //      split(new_center, centers->size(),
// 	    c_pk->points_served(), 
// 	    &new_points_served, dataset, r_k_1);
//       c_pk->set_num_points(c_pk->points_served()->size());
      
//       // updating the farthest for the center
//       update_farthest(c_pk);
      
//       if (c_pk->points_served()->size() == 0) {
// 	// if the center becomes empty, we 
// 	// remove it from the farthest set
// 	Point<T> *temp = new Point<T>();
// 	temp->set_point(-1);
// 	temp->set_alpha_k(-1.0);
// 	farthest_set[c_pk->index()] = temp;

//       }
//       else {
// 	// else we update the farthest set accordingly
// 	farthest_set[c_pk->index()] 
// 	  = (*(c_pk->points_served()))[c_pk->farthest()];
//       }
      
      // splitting points for all the centers which are 
      // in the friends list of c_pk
      GenVector<T> p_k;
      dataset.MakeColumnVector(new_center->point(), &p_k);

      Center<T> **old_centers = centers->begin();
      Center<T> **cend = centers->end();

      for (; old_centers != cend; old_centers++) {
	
	GenVector<T> old_p_k;
	dataset.MakeColumnVector((*old_centers)->point(), &old_p_k);

	T ub = r_k_1 + (*old_centers)->radius();
	T distance = pdc::DistanceEuclidean(p_k, old_p_k, ub);

	if (distance <= ub) {
	  split(new_center, centers->size(),
		(*old_centers)->points_served(), 
		&new_points_served, 
		dataset, r_k_1);
	}
	(*old_centers)->set_num_points((*old_centers)->points_served()->size());
	update_farthest(*old_centers);
	if ((*old_centers)->num_points() == 0) {
	  Point<T> *temp = new Point<T>();
	  temp->set_point(-1);
	  temp->set_alpha_k(-1.0);
	  farthest_set[(*old_centers)->index()] = temp;
	}
	else {
	  farthest_set[(*old_centers)->index()] 
	    = (*((*old_centers)->points_served()))[(*old_centers)->farthest()];
	}
      }


//       Center **frnd_center = c_pk->friends_list()->begin();
//       Center **frnd_center_end = c_pk->friends_list()->end();
//       for (; frnd_center != frnd_center_end; frnd_center++) {
// 	split(new_center, centers->size(),
// 	      (*frnd_center)->points_served(), 
// 	      &new_points_served, dataset, r_k_1);
// 	(*frnd_center)->set_num_points((*frnd_center)->points_served()->size());
// 	update_farthest(*frnd_center);

// 	if ((*frnd_center)->points_served()->size() == 0) {
// 	  Point<T> *temp = new Point<T>();
// 	  temp->set_point(-1);
// 	  temp->set_alpha_k(-1.0);
// 	  farthest_set[(*frnd_center)->index()] = temp;	  
// 	}
// 	else {
// 	  farthest_set[(*frnd_center)->index()] = 
// 	    (*((*frnd_center)->points_served()))[(*frnd_center)->farthest()];
// 	}
//       }

      // Firstly this r_k value is wrong. The r_k value is computed
      // using the new p_k too asshole

//       T r_k = find_max_radius(&farthest_set);

      // The only hope is to find out if the friends list thing
      // is actually causing all the problem and maybe the 
      // cluster formed in actually not the Gonzalez clusters
      // need to check the friends list forming and maintaining again 

      // forming the friends list here using the friends list of the 
      // cluster center two phases ago

//       ArrayList<Center<T> *> friends_list;
//       ArrayList<T> friends_list_distances;
//       friends_list.Init(0);
//       friends_list_distances.Init(0);

//       if (new_center->center_2() != -1) {
// 	Center<T> *old_center = (*centers)[new_center->center_2()];
// 	GenVector<T> p_k;
// 	dataset.MakeColumnVector(new_center->point(), &p_k);

// 	Center<T> **frnd = old_center->friends_list()->begin();
// 	Center<T> **frnd_end = old_center->friends_list()->end();

// 	for (; frnd != frnd_end; frnd++) {
// 	  GenVector<T> old_p_k;
// 	  dataset.MakeColumnVector((*frnd)->point(), &old_p_k);

// 	  T dist = pdc::DistanceEuclidean<T>(p_k, old_p_k, 4*r_k);
// 	  if (dist < 4 * r_k) {
// 	    friends_list.PushBackCopy(*frnd);
// 	    friends_list_distances.PushBackCopy(dist);
// 	  }
// 	}
//       }
//       else {
// 	DEBUG_ASSERT(new_center->center_2() == -1);
// 	Center<T> **cbegin = centers->begin();
// 	Center<T> **cend = centers->end();

// 	GenVector<T> p;
// 	dataset.MakeColumnVector(new_center->point(), &p);
// 	for (; cbegin != cend; cbegin++) {
// 	  GenVector<T> q;
// 	  dataset.MakeColumnVector((*cbegin)->point(), &q);
// 	  T dist = pdc::DistanceEuclidean<T>(p, q, 4*r_k);
// 	  if (dist < 4 * r_k) {
// 	    friends_list.PushBackCopy(*cbegin);
// 	    friends_list_distances.PushBackCopy(dist);
// 	  }
// 	}
//       }

//       // updating the friends_list of all the centers
//       Center<T> **begin = centers->begin();
//       Center<T> **end = centers->end();
    

//       for (; begin != end; begin++) {
// 	T bound = min(4 * (*begin)->r_i(), 8 * r_k);
// 	(*begin)->set_r_k(r_k);
// 	for (index_t i = 0; i < (*begin)->friends_list()->size();) {
// 	  if ((*((*begin)->friends_list_distances()))[i] > bound) {
// 	    (*begin)->friends_list_distances()->Remove(i);
// 	    (*begin)->friends_list()->Remove(i);
// 	  }
// 	  else {
// 	    i++;
// 	  }
// 	}
//       }

      // forming the new center now
      Center<T> *new_node = new Center<T>();
      T r_i = 0.0;
      index_t r_i_max = -1;

      for (index_t i = 0; i < new_points_served.size(); i++) {
	if (new_points_served[i]->alpha_k() >= r_i) {
	  r_i = new_points_served[i]->alpha_k();
	  r_i_max = i;
	}
      }
      if (r_i_max == -1) {
	DEBUG_ASSERT(new_points_served.size() == 0);
      }
      new_node->Init(new_center->point(), r_i, centers->size());
      new_node->points_served()->Renew();
      new_node->set_num_points(new_points_served.size());
      new_node->points_served()->InitSteal(&new_points_served);
//       new_node->friends_list()->Renew();
//       new_node->friends_list()->InitSteal(&friends_list);
//       new_node->friends_list_distances()->Renew();
//       new_node->friends_list_distances()->InitSteal(&friends_list_distances);
      new_node->set_farthest(r_i_max);
      new_node->set_parent(new_center->center_at_last_level());
      new_node->set_distance_to_parent(new_center->dist_at_last_level());

      if (r_i_max == -1) {
	Point<T> *temp = new Point<T>();
	temp->set_point(-1);
	temp->set_alpha_k(-1.0);
	farthest_set.PushBackCopy(temp);
      }
      else {
	farthest_set.PushBackCopy((*(new_node->points_served()))[new_node->farthest()]);
      }
      centers->PushBackCopy(new_node);
    }
    
    for (index_t i = 0; i < root->num_of_children(); i++) {
      update_tree(root->child(i), dataset);
    }

    update_scale<T, TreeType>(root, max_scale, max_scale);
    return root;
  }

  template<typename T, typename TreeType>
  TreeType* Cluster(GenMatrix<T>& dataset, T base) {

    EC = (double)base;
    ArrayList<Center<T>*> centers;
    index_t k = dataset.n_cols();
    TreeType *root 
      = Gonzalez<T, TreeType>(dataset, k, &centers);
    
    return root;
  }
};
#endif
