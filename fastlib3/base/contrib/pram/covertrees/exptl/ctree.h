#ifndef TREE_COVER_TREE_H
#define TREE_COVER_TREE_H

#include <fastlib/fastlib.h>

#include "cover_tree.h"
#include "distances.h"

// need to document these soon

namespace ctree {

  template<typename T> 
    T BASE() {
    T BASE = 1.3;
    return BASE;
  }

  template<typename T> 
    T inverse_log_base() {
    T inverse_log_base  = 1.0 / log(BASE<T>());
    return inverse_log_base;
  }

  const index_t NEG_INF = (int) log(0);

  template<typename T>
    inline T scaled_distance(index_t scale) {
    return pow(BASE<T>(), scale);
  }

  template<typename T>
    inline index_t scale_of_distance(T distance) {
    return (index_t) ceil(log(distance)*inverse_log_base<T>());
  }

  template<typename T>
    class NodeDistances {
    
    private:
    // the point not needed, we just use the index when we 
    // make an arraylist for each of the point
    // Vector point_;
    index_t point_;
    ArrayList<T> distances_;
    ArrayList<index_t> index_at_scale_;
    
    public:
    
    NodeDistances() {
      distances_.Init(0);
      index_at_scale_.Init(101);
      for (index_t i = 0; i < 101; i++) {
	index_at_scale_[i] = -1;
      }
    }
    
    ~NodeDistances() {
    }

    index_t point() {
      return point_;
    }

    ArrayList<T> *distances() {
      return &distances_;
    }

    T distances(index_t in) {
      return distances_[in];
    }

    void add_distance(T dist) {
      distances_.PushBackCopy(dist);
      return;
    }

    void set_index_at_scale(index_t scale, index_t index) {
      index_at_scale_[scale] = index;
    }

    void reset_index_at_scale(index_t scale) {
      index_at_scale_[scale] = -1;
    }

    index_t index_at_scale(index_t scale) {
      return index_at_scale_[scale];
    }

    void Init(index_t point, T dist) {
      point_ = point;
      distances_.PushBackCopy(dist);
      return;
    }
  };
  
  template<typename T>
    class Points{

    private:
    index_t index_at_level_;
    T distance_;

    public:

    Points() {
      index_at_level_ = -1;
      distance_ = 0.0;
    }

    ~Points() {
    }

    void Init(index_t index_at_level, T distance) {
      index_at_level_ = index_at_level;
      distance_ = distance;
    }
 
    void set_index_at_level(index_t index_at_level) {
      index_at_level_ = index_at_level;
    }

    void set_distance(T dist) {
      distance_ = distance;
    }

    index_t index_at_level() {
      return index_at_level_;
    }

    T distance() {
      return distance_;
    }
  };

  template<typename T>
    T max_set(ArrayList<NodeDistances<T>*> *set) {

    T max = 0.0;
    for (index_t i = 0; i < set->size(); i++) {
      if(max < (*set)[i]->distances()->back()) {
	max = (*set)[i]->distances()->back();
      }
    }
    return max;
  }

  void print_space(index_t n) {
    for (index_t i = 0; i < n; i++) {
      printf("\t");
    }
    return;
  }

  // find out how to print templatized things
  template<typename TCoverTreeNode>
    void print_tree(index_t depth, TCoverTreeNode *top_node) {
    print_space(depth);
    printf("Point %"LI"d:", top_node->point()+1);
    //NOTIFY("%"LI"d:%"LI"d", top_node->point()+1, top_node->scale_depth());
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

  template<typename TCoverTreeNode>
    void PrintTree(TCoverTreeNode *top_node) {
    index_t depth = 0;
    print_tree(depth, top_node);
    return;
  }

  // here we assume that both the point_set and the far set are
  // already initialized
  template<typename T>
    void split_far(ArrayList<NodeDistances<T>*> *point_set, 
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
    void split_near(index_t point, const GenMatrix<T>& data,
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

  /**
   * make a matrix in which the i-j th entry implies that
   * point j is in the near set of point i when split_near()
   * is done on it, and the distance matrix stores this distance
   * so that we don't have to do the distance computation again 
   * when we actually do the split near thing
   *
   * the reason for making the membership matrix a normal matrix 
   * instead of a GenMatrix<index_t> is that we can do a matrix-vector 
   * multiplication with it to obtain the point with the maximum
   * size of the near set.
   *
   * The vector of indices stores the position of a particular point
   * in the point set. As in if a point has index_at_scale[scale_depth] = i, 
   * index_at_point_set[i] gives us the position of that point in 
   * the point set
   */
  /*
  template<typename T>
    void Initialize(const GenMatrix<T> data,
		    ArrayList<NodeDistances<T>*> *point_set, 
		    index_t current_scale, index_t max_scale, 
		    ArrayList<ArrayList<Points<T>*> > *membership,
		    ArrayList<index_t> *index_at_point_set) {

    index_t size = point_set->size();
    T bound = scaled_distance<T>(current_scale);

    membership->Init(size);
    index_at_point_set->Init(size);

    for (index_t i = 0; i < size; i++) {
      (*membership)[i].Init(0);
    }

    NodeDistances<T> **begin = point_set->begin();
    NodeDistances<T> **end = point_set->end();

    for (index_t i = 0, j; begin != end; begin++, i++) {
      NodeDistances<T> **subbegin = begin + 1;
      GenVector<T> p;
      data.MakeColumnVector((*begin)->point(), &p);
      j = i + 1;
      (*begin)->set_index_at_scale(max_scale - current_scale, i);
      (*index_at_point_set)[i] = i;

      for (; subbegin != end; subbegin++, j++) {
	GenVector<T> q;
	data.MakeColumnVector((*subbegin)->point(), &q);

	T dist = pdc::DistanceEuclidean<T>(p, q, bound);
	if (dist <= bound) {
	  Points<T> *x = new Points<T>();
	  Points<T> *y = new Points<T>();
	  x->Init(j, dist);
	  y->Init(i, dist);
	  (*membership)[i].PushBackCopy(x);
	  (*membership)[i].PushBackCopy(y);
	}
      }
    }
  }
  */
  /*
  template<typename T>
    void form_matrices(const GenMatrix<T>& data,
		       ArrayList<NodeDistances<T>*> *point_set,
		       index_t current_scale, index_t max_scale,
		       ArrayList<GenVector<T> > *membership,//Matrix *membership,
		       ArrayList<GenVector<T> > *distances,//GenMatrix<T> *distances,
		       ArrayList<index_t> *index_at_point_set) {
    
    index_t size = point_set->size(), i = 0, j;
    T bound = scaled_distance<T>(current_scale);
    NodeDistances<T> **begin = point_set->begin();
    NodeDistances<T> **end = point_set->end();
    
    // NOTIFY("size = %"LI"d", size);
    membership->Init(size);
    distances->Init(size);
    for (index_t i = 0; i < size; i++) {
      (*membership)[i].Init(size);
      (*membership)[i].SetAll(0.0);
      (*distances)[i].Init(size);
      (*distances)[i].SetAll(0.0);
    }
    index_at_point_set->Init(size);
    
    //membership->SetAll(0.0);
    //distances->SetAll(0.0);
    //index_at_point_set->SetAll(0);
    
    //index_t *ptr = index_at_point_set->ptr();

    for (; begin != end; begin++, i++) {
      NodeDistances<T> **subbegin = begin + 1;
      GenVector<T> p;
      data.MakeColumnVector((*begin)->point(), &p);
      j = i + 1;
      // DEBUG_ASSERT((*begin)->index_at_scale(max_scale - current_scale)
      //	   == -1);
      (*begin)->set_index_at_scale(max_scale - current_scale, i);
      
      //ptr[i] = i;
      (*index_at_point_set)[i] = i;
      //NOTIFY("%"LI"d %"LI"d", i, (*index_at_point_set)[i]);
      //index_at_point_set->PrintDebug("I..");
      for (; subbegin != end; subbegin++, j++) {
	GenVector<T> q;
	data.MakeColumnVector((*subbegin)->point(), &q);

	T dist = pdc::DistanceEuclidean<T>(p, q, bound);
	if (dist <= bound) {
	  //  DEBUG_ASSERT(membership->get(i, j) == 0.0);
	  // membership->set(i, j, 1);
	  // DEBUG_ASSERT(membership->get(j, i) == 0.0);
	  // membership->set(j, i, 1);
	  DEBUG_ASSERT((*membership)[i].get(j) == 0.0);
	  (*membership)[i].ptr()[j] = 1.0;
	  DEBUG_ASSERT((*membership)[j].get(i) == 0.0);
	  (*membership)[j].ptr()[i] = 1.0;
	}
	//DEBUG_ASSERT(distances->get(i, j) == 0.0);
	//distances->set(i, j, dist);
	//DEBUG_ASSERT(distances->get(j, i) == 0.0);
	//distances->set(j, i, dist);

	DEBUG_ASSERT((*distances)[i].get(j) == 0.0);
	(*distances)[i].ptr()[j] = dist;
	DEBUG_ASSERT((*distances)[j].get(i) == 0.0);
	(*distances)[j].ptr()[i] = dist;
      }
    }
    //index_at_point_set->PrintDebug("I");
  }
  */
  /**
   * This function uses the matrices to find the point with the 
   * max sized near set. Once we find the point, we form the 
   * near set of that point without doing the distance computations 
   * again and just using the distances' matrix.
   * 
   * We put the chosen point in the consumed set and update the 
   * membership matrix accordingly.
   * 
   * To put the chosen point in the consumed set, we put the chosen 
   * point at the end of the point set and then pop it out. 
   * So here we need to update the index_at_point_set
   */

  /*
  template<typename T> 
    void GreedyNearSet(ArrayList<ArrayList<Points<T>*> > *membership, 
		       ArrayList<NodeDistances<T>*> *point_set, 
		       ArrayList<NodeDistances<T>*> *near, 
		       ArrayList<NodeDistances<T>*> *consumed_set, 
		       ArrayList<index_t> *index_at_point_set, 
		       index_t current_scale, index_t max_scale, 
		       index_t *new_point, T *new_dist) {

    index_t size = membership->size(), max = 0, index = -1;

    if (size == 1) {
      index = 0;
    }
    else {
      for (index_t i = 0; i < size; i++) {
	if ((*membership)[i].size() >= max 
	    && (*membership)[i].size() != BIG_BAD_NUMBER) {
	  max = (*membership)[i].size();
	  index = i;
	}
      }
    }

    DEBUG_ASSERT(index != -1);
    index_t pivot_point_set = (*index_at_point_set)[index];
    DEBUG_ASSERT(pivot_point_set != -1);

    *new_point = (*point_set)[pivot_point_set]->point();
    *new_dist = (*point_set)[pivot_point_set]->distances()->back();

    // moving the pivot back to the back of the point set
    // to pop it out and put it in the consumed set.
    // The index_at_point_set is edited accordingly
  
    if (point_set->size() != 1) {
      NodeDistances<T> *temp = (*point_set)[pivot_point_set];
      (*point_set)[pivot_point_set] = point_set->back();
      
      DEBUG_ASSERT((*point_set)[pivot_point_set]->index_at_scale(max_scale - current_scale) 
		   != -1);
      
      (*index_at_point_set)[(*point_set)[pivot_point_set]->index_at_scale(max_scale 
									  - current_scale)] 
	= pivot_point_set;
      
      *(point_set->end() - 1) = temp;
      (*index_at_point_set)[index] = -1;
    
      // adding the pivot element to the consumed set
      point_set->PopBackInit(consumed_set->PushBackRaw());
    }
    else {
      (*index_at_point_set)[index] = -1;

      point_set->PopBackInit(consumed_set->PushBackRaw());
      DEBUG_ASSERT(point_set->size() == 0);
    }

    Points<T> **begin = (*membership)[index].begin();
    Points<T> **end = (*membership)[index].end();
    ArrayList<index_t> consumed_indices;
    ArrayList<NodeDistances<T>*> far;
    index_t initial_size = near->size();

    far.Init(0);
    consumed_indices.Init(point_set->size());
    for (index_t i = 0; i < point_set->size(); i++) {
      consumed_indices[i] = 0;
    }

    for(; begin != end; begin++) {
      DEBUG_ASSERT((*begin)->index_at_level() != -1);
      index_t temp_index = (*index_at_point_set)[(*begin)->index_at_level()];
      if (temp_index != -1) {
	(*point_set)[temp_index]->add_distance((*begin)->distance());
	near->PushBackCopy((*point_set)[temp_index]);
	consumed_indices[temp_index] = 1;
	(*index_at_point_set)[(*begin)->index_at_level()] = -1;
      }
    }

    for (index_t i = 0; i < point_set->size(); i++) {
      if (consumed_indices[i] == 0) {
	far.PushBackCopy((*point_set)[i]);
	(*index_at_point_set)[far.back()->index_at_scale(max_scale - current_scale)]
	  = far.size() - 1;
      }
    }
    DEBUG_ASSERT_MSG(point_set->size() ==
		     near->size() - initial_size + far.size(), 
		     "GreedyNear: point sets don't add up");
    point_set->Renew();
    point_set->InitSteal(&far);

    (*membership)[index].Renew();
    //(*membership)[index].Init(0);

  }
  */
  
/*   template<typename T> */
/*     void make_max_new_point_set(ArrayList<GenVector<T> > *membership,//Matrix& membership, */
/* 				ArrayList<GenVector<T> > *distances, //GenMatrix<T>& distances, */
/* 				ArrayList<NodeDistances<T>*> *point_set, */
/* 				ArrayList<NodeDistances<T>*> *near, */
/* 				ArrayList<NodeDistances<T>*> *consumed_set, */
/* 				ArrayList<index_t> *index_at_point_set, */
/* 				index_t current_scale, index_t max_scale, */
/* 				index_t *new_point, T *new_dist) { */

/*     //Vector one_vec, members; */
/*     GenVector<T> one_vec, members; */
/*     index_t size = index_at_point_set->size(); */
/*     index_t index = -1, in = 0, initial_size = near->size(); */
/*     //double max = 0.0; */
/*     T max = 0.0; */
/*     ArrayList<NodeDistances<T>*> far; */

/*     one_vec.Init(size); */
/*     one_vec.SetAll(1.0); */
/*     members.Init(size); */
/*     members.SetAll(0.0); */
    
/*     //la::MulInit(membership, one_vec, &members); */
/*     for (index_t i = 0; i < size; i++) { */
/*       T *x = (*membership)[i].ptr(); */
/*       T *y = one_vec.ptr(); */
/*       for (index_t j = 0; j < size; j++) { */
/* 	members.ptr()[i] += (*x++)*(*y++); */
/*       } */
/*     } */

/*     far.Init(0); */
/*     //members.PrintDebug("Members"); */
/*     // NOTIFY("%"LI"d", point_set->size()) */
/*     if (size == 1) { */
/*       index = 0; */
/*     } */
/*     else { */
/*       //double *begin = members.ptr(); */
/*       //double *end = begin + members.length(); */
/*       T *begin = members.ptr(); */
/*       T *end = begin + members.length(); */
      
/*       for(; begin != end; begin++, in++) { */
/* 	//printf("%f\n", *begin); */
/* 	if (*begin >= max) { */
/* 	  //printf("here"); */
/* 	  max = *begin; */
/* 	  index = in; */
/* 	  //printf(" %"LI"d\n", index); */
/* 	} */
/*       } */
/*     } */
/*     //NOTIFY("index = %"LI"d", index); */
/*     DEBUG_ASSERT(index != -1); */
/*     // NOTIFY("HERE !"); */
/*     index_t pivot_point_set = (*index_at_point_set)[index]; */
/*     DEBUG_ASSERT(pivot_point_set != -1); */

/*     *new_point = (*point_set)[pivot_point_set]->point(); */
/*     *new_dist = (*point_set)[pivot_point_set]->distances()->back(); */

/*     // interchanging the last element in the point set and the */
/*     // pivot element */
/*     if (point_set->size() != 1) { */
/*       NodeDistances<T> *temp = (*point_set)[pivot_point_set]; */
/*       (*point_set)[pivot_point_set] = point_set->back(); */
      
/*       DEBUG_ASSERT((*point_set)[pivot_point_set]->index_at_scale(max_scale - current_scale) != -1); */
      
/*       (*index_at_point_set)[(*point_set)[pivot_point_set]->index_at_scale(max_scale - current_scale)] = pivot_point_set; */
/*       *(point_set->end() - 1) = temp; */
    
/*       (*index_at_point_set)[index] = -1; */
    
/*       // adding the pivot element to the consumed set */
/*       point_set->PopBackInit(consumed_set->PushBackRaw()); */
/*     } */
/*     else { */
/*       (*index_at_point_set)[index] = -1; */

/*       point_set->PopBackInit(consumed_set->PushBackRaw()); */
/*       DEBUG_ASSERT(point_set->size() == 0); */
/*     } */

    
/*     // NOTIFY("HERE !!"); */
/*     // making the near set for the pivot point and using the distance matrix */
/*     T *point_begin = (*membership)[index].ptr(); */
/*     T *dist_begin = (*distances)[index].ptr(); */

/*     if (point_set->size() != 0) { */
/*       for (index_t i = 0; i < size; i++) { */
/* 	if (point_begin[i] == 1.0) { */
/* 	  DEBUG_ASSERT((*index_at_point_set)[i] != -1); */
	  
/* 	  (*point_set)[(*index_at_point_set)[i]]->add_distance(dist_begin[i]); */
/* 	  near->PushBackCopy((*point_set)[(*index_at_point_set)[i]]); */
/* 	  (*index_at_point_set)[i] = -1; */
/* 	} */
/* 	else { */
/* 	  if (i != index && (*index_at_point_set)[i] != -1) { */
/* 	    //DEBUG_ASSERT((*index_at_point_set)[i] != -1); */
	    
/* 	    far.PushBackCopy((*point_set)[(*index_at_point_set)[i]]); */
/* 	    (*index_at_point_set)[i] = far.size() - 1; */
/* 	  } */
/* 	} */
/*       } */
/*     } */


/*  /\*    if (point_set->size() != 0) { *\/ */
/* /\*       for (index_t i = 0; i < size; i++) { *\/ */
/* /\* 	if (membership.get(index, i) == 1.0) { *\/ */
/* /\* 	  DEBUG_ASSERT((*index_at_point_set)[i] != -1); *\/ */
	  
/* /\* 	  (*point_set)[(*index_at_point_set)[i]]->add_distance(distances.get(index, i)); *\/ */
/* /\* 	  near->PushBackCopy((*point_set)[(*index_at_point_set)[i]]); *\/ */
/* /\* 	  (*index_at_point_set)[i] = -1; *\/ */
/* /\* 	} *\/ */
/* /\* 	else { *\/ */
/* /\* 	  if (i != index && (*index_at_point_set)[i] != -1) { *\/ */
/* /\* 	    //DEBUG_ASSERT((*index_at_point_set)[i] != -1); *\/ */
	    
/* /\* 	    far.PushBackCopy((*point_set)[(*index_at_point_set)[i]]); *\/ */
/* /\* 	    (*index_at_point_set)[i] = far.size() - 1; *\/ */
/* /\* 	  } *\/ */
/* /\* 	} *\/ */
/* /\*       } *\/ */
/* /\*     } *\/ */

/*     DEBUG_ASSERT_MSG(point_set->size() == */
/* 		     near->size() - initial_size + far.size(), */
/* 		     "split_near: point set doesn't add up\n"); */
/*     // NOTIFY("HERE !!!"); */
/*     point_set->Renew(); */
/*     point_set->InitSteal(&far); */
    
/*     // updating the membership matrix, making the pivot row and pivot column */
/*     // zero */
/*     for (index_t i = 0; i < size; i++) { */
/*       if (point_begin[i] == 1.0) { */
/* 	(*membership)[i].ptr()[index] = 0.0; */
/*       } */
/*       point_begin[i] = -1.0; */
/*     } */

/* /\*     for (index_t i = 0; i < size; i++) { *\/ */
/* /\*       if (membership.get(index, i) == 1.0) { *\/ */
/* /\* 	membership.set(i, index, 0.0); *\/ */
/* /\*       } *\/ */
/* /\*       membership.set(index, i, -1.0); *\/ */
  
/* /\*     } *\/ */

/*   } */
  
  /**
   * First we need to have the new_point_set separated into ones which go in 
   * the point_set and far set.
   *
   * Then we need to reset the index_at_scale for the next scale for all the points
   * and then for the points to be added back to the point_set, the index_at_point_set 
   * has to be re-established.
   *
   * Then we add the points in the new_consumed_set to the consumed_set
   * and update the membership matrix, removing the corresponding 
   * rows and columns of the consumed sets
   */
  /*
  template<typename T>
    void Update(ArrayList<ArrayList<Points<T>*> > *membership, 
		ArrayList<NodeDistances<T>*> *point_set, 
		ArrayList<NodeDistances<T>*> *far, 
		ArrayList<NodeDistances<T>*> *consumed_set, 
		ArrayList<NodeDistances<T>*> *new_point_set,
		ArrayList<NodeDistances<T>*> *new_consumed_set, 
		index_t current_scale, index_t next_scale, 
		index_t max_scale, 
		ArrayList<index_t> *index_at_point_set) {

    T bound = scaled_distance<T>(current_scale);

    // new point set separated into point_set and far,
    // and points in th new_point_set have reset index_at_scale for
    // the next scale and the index_at_point_set is
    // updated accordingly as the point within the bound is being added
    // to the point_set

    NodeDistances<T> **begin = new_point_set->begin();
    NodeDistances<T> **end = new_point_set->end();

    for (; begin != end; begin++) {

      (*begin)->distances()->PopBack();
      (*begin)->reset_index_at_scale(max_scale - next_scale);

      if ((*begin)->distances()->back() <= bound) {
	point_set->PushBackCopy(*begin);

	DEBUG_ASSERT((*begin)->index_at_scale(max_scale - current_scale) != -1);
	(*index_at_point_set)[(*begin)->index_at_scale(max_scale - current_scale)] 
	  = point_set->size() - 1;
      }
      else {
	far->PushBackCopy(*begin);
      }
    }
    new_point_set->Resize(0);

    // adding the new_consumed_set to the consumed_set
    // and updating the membership matrix accordingly,
    // removing the rows and columns of the corresponding
    // points

    while (new_consumed_set->size() > 0) {
      new_consumed_set->back()->distances()->PopBack();
      
      index_t consumed_index 
	= new_consumed_set->back()->index_at_scale(max_scale - current_scale);

      // checking if it is consumed from the far set 
      // or the near set
      if (consumed_index != -1) {
	
	//updating the membership matrix
	(*membership)[consumed_index].Renew();
	(*index_at_point_set)[consumed_index] = -1;
      }
      // adding to the consumed set
      new_consumed_set->PopBackInit(consumed_set->PushBackRaw());
    }
  }
  */

/*   template<typename T> */
/*     void update_matrices(ArrayList<NodeDistances<T>*> *point_set, */
/* 			 ArrayList<NodeDistances<T>*> *far, */
/* 			 ArrayList<NodeDistances<T>*> *consumed_set, */
/* 			 ArrayList<NodeDistances<T>*> *new_point_set, */
/* 			 ArrayList<NodeDistances<T>*> *new_consumed_set, */
/* 			 index_t current_scale, index_t next_scale, */
/* 			 index_t max_scale, ArrayList<index_t> *index_at_point_set, */
/* 			 ArrayList<GenVector<T> > *membership /\* Matrix *membership *\/){ */

/*     T bound = scaled_distance<T>(current_scale); */
/*     NodeDistances<T> **begin = new_point_set->begin(); */
/*     NodeDistances<T> **end = new_point_set->end(); */
/*     //    index_t size = membership->n_cols(); */
/*     index_t size =  (*membership)[0].length(); */

/*     // new point set separated into point_set and far, */
/*     // and points in th new_point_set have reset index_at_scale for */
/*     // the next scale and the index_at_point_set is */
/*     // updated accordingly as the point within the bound is being added */
/*     // to the point_set */
/*     for (; begin != end; begin++) { */

/*       (*begin)->distances()->PopBack(); */
/*       //(*begin)->reset_index_at_scale(max_scale - next_scale); */

/*       if ((*begin)->distances()->back() <= bound) { */
/* 	point_set->PushBackCopy(*begin); */

/* 	DEBUG_ASSERT((*begin)->index_at_scale(max_scale - current_scale) != -1); */

/* 	(*index_at_point_set)[(*begin)->index_at_scale(max_scale - current_scale)] = point_set->size() - 1; */
/*       } */
/*       else { */
/* 	far->PushBackCopy(*begin); */
/*       } */
/*     } */
    
/*     new_point_set->Resize(0); */

/*     // adding the new_consumed_set to the consumed_set */
/*     // and updating the membership matrix accordingly, */
/*     // removing the rows and columns of the corresponding */
/*     // points */

/*     while (new_consumed_set->size() > 0) { */
/*       new_consumed_set->back()->distances()->PopBack(); */
      
/*       index_t consumed_index = new_consumed_set->back()->index_at_scale(max_scale - current_scale); */
/*       if (consumed_index != -1) { */
	
/* 	//updating the membership matrix */
/* 	DEBUG_ASSERT(consumed_index != -1); */
	
/* 	T *consumed_begin = (*membership)[consumed_index].ptr(); */

/* 	for (index_t i = 0; i < size; i++) { */

/* 	  if (consumed_begin[i] == 1.0) { */

/* 	    (*membership)[i].ptr()[consumed_index] = 0.0; */
/* 	  } */
/* 	  consumed_begin[i] = -1.0; */
/* 	} */

/* /\* 	for (index_t i = 0; i < size; i++) { *\/ */
/* /\* 	  if (membership->get(consumed_index, i) == 1.0) { *\/ */
/* /\* 	    membership->set(i, consumed_index, 0.0); *\/ */
/* /\* 	  } *\/ */
/* /\* 	  membership->set(consumed_index, i, -1.0); *\/ */
/* /\* 	} *\/ */

/* 	(*index_at_point_set)[consumed_index] = -1; */
/*       } */
/*       // adding to the consumed set */
/*       new_consumed_set->PopBackInit(consumed_set->PushBackRaw()); */
/*     } */
/*   } */

  
  template<typename T>
    void max_covering_point(const GenMatrix<T>& data, 
			    ArrayList<NodeDistances<T>*> *point_set, 
			    ArrayList<NodeDistances<T>*> *consumed_set, 
			    ArrayList<NodeDistances<T>*> *near,
			    index_t scale, index_t *new_point, 
			    T *new_dist) {

    if (point_set->size() == 1) {
      *new_point = point_set->front()->point();
      *new_dist = point_set->front()->distances()->back();
      point_set->PopBackInit(consumed_set->PushBackRaw());

      //NOTIFY("returning here");
      return;
    }
    else {
      T bound = scaled_distance<T>(scale);
      index_t max = -1, passed, failed, max_index = -1, total = point_set->size();
      ArrayList<index_t> max_set, current_set;
      ArrayList<T> max_set_distances, current_set_distances;
      index_t near_set_initial_size = near->size();

      max_set.Init(0);
      max_set_distances.Init(0);
      current_set.Init(0);
      current_set_distances.Init(0);

      for (index_t i = 0; i < total; i++) {

	passed = 0; 
	failed = 0;
	NodeDistances<T> **begin = point_set->begin();
	NodeDistances<T> **end = point_set->end();
      
	GenVector<T> p;
	data.MakeColumnVector((*point_set)[i]->point(), &p);
      
	for (index_t j = 0; begin != end && max + failed < total; begin++, j++) {
	  if ( j != i) {
	    GenVector<T> q;
	    data.MakeColumnVector((*begin)->point(), &q);
	  
	    T dist = pdc::DistanceEuclidean<T>(p, q, bound);
	    if (dist > bound) {
	      failed++;
	    }
	    else {
	      passed++;
	      current_set.PushBackCopy(j);
	      current_set_distances.PushBackCopy(dist);
	    }
	  }
	}
	if (passed > max) {
	  max = passed; 
	  max_index = i;
	  max_set.Renew();
	  max_set.InitCopy(current_set);
	  max_set_distances.Renew();
	  max_set_distances.InitCopy(current_set_distances);
	  DEBUG_ASSERT(max_set.size() == max_set_distances.size());
	  //DEBUG_ASSERT(max_set.size() != 0);
	}
	current_set.Resize(0);
	current_set_distances.Resize(0);
      }

      DEBUG_ASSERT(max_index != -1);

      ArrayList<NodeDistances<T>*> far;
      ArrayList<index_t> far_set;
      far.Init(0);
      far_set.Init(point_set->size());
      for (index_t i = 0; i < point_set->size(); i++) {
	far_set[i] = 0;
      }
      far_set[max_index] = 1;

      DEBUG_ASSERT(max == max_set.size());
      for (index_t i = 0; i < max; i++) {
	far_set[max_set[i]] = 1;
	(*point_set)[max_set[i]]->add_distance(max_set_distances[i]);
	near->PushBackCopy((*point_set)[max_set[i]]);
      }

      for (index_t i = 0; i < point_set->size(); i++) {
	if (far_set[i] == 0) {
	  far.PushBackCopy((*point_set)[i]);
	}
      }

      *new_point = (*point_set)[max_index]->point();
      *new_dist = (*point_set)[max_index]->distances()->back();

      consumed_set->PushBackCopy((*point_set)[max_index]);

      DEBUG_ASSERT_MSG(point_set->size() == near->size() - near_set_initial_size
		       + far.size() + 1, "max_covering_point:point sets don't add up");

      point_set->Renew();
      point_set->InitSteal(&far);
      //NOTIFY("returning");
      return;
    }
  }
  

  template<typename TCoverTreeNode, typename T>
    TCoverTreeNode *private_make_tree(index_t point, const GenMatrix<T>& data,
				      index_t current_scale,
				      index_t max_scale, 
				      ArrayList<NodeDistances<T>*> *point_set,
				      ArrayList<NodeDistances<T>*> *consumed_set,
				      index_t *flag = NULL) {
    //NOTIFY("->%"LI"d %"LI"d", current_scale, point_set->size());
    // no other point so leaf in explicit tree
    if (point_set->size() == 0) { 
      TCoverTreeNode *node = new TCoverTreeNode();
      node->MakeLeafNode(point);
      return node;
    }
    else {
      T max_dist = max_set(point_set);
      index_t next_scale = min(current_scale - 1, scale_of_distance(max_dist));
      // At the -INF level so all points are nodes
      // and we have point with zero distances
      if (next_scale == NEG_INF) { 
	ArrayList<TCoverTreeNode*> children;
	NodeDistances<T> **begin = point_set->begin();
	NodeDistances<T> **end = point_set->end();

	children.Init(0);
	//NOTIFY("%"LI"d +", point);
	TCoverTreeNode *self_child = new TCoverTreeNode();
	self_child->MakeLeafNode(point);
	children.PushBackCopy(self_child);
	//NOTIFY("here");
	for (; begin != end; begin++) {
	  //NOTIFY("%"LI"d +", (*begin)->point());
	  TCoverTreeNode *child = new TCoverTreeNode();
	  child->MakeLeafNode((*begin)->point());
	  children.PushBackCopy(child);
	  consumed_set->PushBackCopy(*begin);
	}

	DEBUG_ASSERT(children.size() == point_set->size() + 1);
	point_set->Resize(0);
	TCoverTreeNode *node = new TCoverTreeNode();
	node->MakeNode(point, 0.0, 100, &children);
	//NOTIFY("there");
	return node;
      }

      // otherwise you need to recurse
      else {

	ArrayList<NodeDistances<T>*> far;

	far.Init(0);
	split_far(point_set, &far, current_scale);
	//NOTIFY("HERE");
	TCoverTreeNode *child;
	if (flag != NULL) {
	  child = private_make_tree<TCoverTreeNode>(point, data, next_scale, 
						    max_scale, point_set, 
						    consumed_set, flag);
	}
	else {
	  //NOTIFY("<-%"LI"d", next_scale);
	  child = private_make_tree<TCoverTreeNode>(point, data, next_scale, 
						    max_scale, point_set, 
						    consumed_set);
	}
	
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

	  index_t point_set_size = point_set->size();

	  // greedy child selection, choosing children with max sized near set
	  if (flag != NULL) {

	    // Matrix membership;
	    //ArrayList<GenVector<T> > membership;
	    //ArrayList<ArrayList<Points<T>*> > membership;
	    //GenMatrix<T> distances;
	    //ArrayList<GenVector<T> > distances;
	    //ArrayList<index_t> index_at_point_set;
	    //T new_dist;
	    //index_t new_point;

	    //form_matrices(data, point_set, current_scale, max_scale, 
	    //		  &membership, &distances, &index_at_point_set);
	    //Initialize(data, point_set, current_scale, max_scale, 
	    //       &membership, &index_at_point_set);
	    //membership.PrintDebug("M new");
	    //distances.PrintDebug("D new");
	    
	    //index_t point_set_size = point_set->size();
	    while (point_set->size() != 0) {
	      //NOTIFY("HERE");
	      T new_dist; 
	      index_t new_point;
	      DEBUG_ASSERT(new_point_set.size() == 0);
	      //make_max_new_point_set(&membership, &distances, point_set, 
	      //		     &new_point_set, consumed_set, 
	      //		     &index_at_point_set, current_scale, 
	      //		     max_scale, &new_point, &new_dist);
	      //   membership.PrintDebug("M near removed");
	      
	      // NOTIFY("-----ArrayList I near removed-----");
	      //for(index_t i = 0; i < index_at_point_set.size(); i++) {
	      //	printf("%"LI"d ", index_at_point_set[i]);
	      //}
	      //printf("\n");fflush(NULL);
	      
	      //    GreedyNearSet(&membership, point_set, &new_point_set, 
	      //    consumed_set, &index_at_point_set, 
	      //    current_scale, max_scale, &new_point, 
	      //    &new_dist);
	      max_covering_point(data, point_set, consumed_set, &new_point_set, 
				 current_scale, &new_point, &new_dist);	      

	      split_near(new_point, data, &far, 
			 &new_point_set, current_scale);
	      //index_t new_point_set_size = new_point_set.size();

	      TCoverTreeNode *child_node = 
		private_make_tree<TCoverTreeNode>(new_point, 
						  data, 
						  next_scale,
						  max_scale, 
						  &new_point_set, 
						  &new_consumed_set, 
						  flag);

	      child_node->set_dist_to_parent(new_dist);
	      children.PushBackCopy(child_node);

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
	      
	      //  membership.PrintDebug("M near removed");
	      // update_matrices(point_set, &far, consumed_set, 
	      //	      &new_point_set, &new_consumed_set, 
	      //	      current_scale, next_scale, max_scale, 
	      //	      &index_at_point_set, &membership);
	      // membership.PrintDebug("M updated");
	      
	      //NOTIFY("-----ArrayList I updated-----");
	      //for(index_t i = 0; i < index_at_point_set.size(); i++) {
	      //printf("%"LI"d ", index_at_point_set[i]);
	      //}
	      // printf("\n");fflush(NULL);
	      //Update(&membership, point_set, &far, consumed_set, 
	      //     &new_point_set, &new_consumed_set, 
	      //      current_scale, next_scale, max_scale, 
	      //   &index_at_point_set);

	    }
	  } // end of greedy child select algorithm

	  // original randomized child selection
	  else {
	    while (point_set->size() != 0) {

	      T new_dist = point_set->back()->distances()->back();
	      index_t new_point = point_set->back()->point();
	      
	      // remember to check here what to use, PushBackRaw() or AddBack()
	      // so that we can use PopBackInit(Element *dest)
	      point_set->PopBackInit(consumed_set->PushBackRaw()); 

	      split_near(new_point, data, point_set, 
			 &new_point_set, current_scale);
	      split_near(new_point, data, &far, 
			 &new_point_set, current_scale);
	  
	      //NOTIFY("<-%"LI"d", next_scale);  
	      TCoverTreeNode *child_node = 
		private_make_tree<TCoverTreeNode>(new_point,
						  data,
						  next_scale,
						  max_scale,
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
	  } // end of randomized child selection
	  
	  // NOTIFY("END GREEDY");
	  DEBUG_ASSERT_MSG(point_set_size >= children.size() - 1, 
			   "point set cannot increase");
	  point_set->Renew();
	  point_set->InitSteal(&far);

	  TCoverTreeNode *node = new TCoverTreeNode();
	  if (flag != NULL) {
	    //printf("%"LI"d --> %"LI"d !\n", point_set_size, children.size());
	  }
	  else {
	    //printf("%"LI"d --> %"LI"d\n", point_set_size, children.size());
	  }
	  node->MakeNode(point, max_set(consumed_set), 
			 max_scale - current_scale, &children);
	  
	  return node;
	}
      }
    }
  }

  template<typename TCoverTreeNode, typename T>
    TCoverTreeNode *MakeCoverTree(const GenMatrix<T>& dataset, 
				  index_t *flag = NULL) {

    index_t n = dataset.n_cols();
    DEBUG_ASSERT(n > 0);
    ArrayList<NodeDistances<T>*> point_set, consumed_set;
    GenVector<T> root_point;

    dataset.MakeColumnVector(0, &root_point);
    point_set.Init(0);
    consumed_set.Init(0);

    // speed up possible here by using pointers
    for (index_t i = 1; i < n; i++) {
      NodeDistances<T> *node_distances = new NodeDistances<T>();
      GenVector<T> point;
      T dist;

      dataset.MakeColumnVector(i, &point);
      //dist = sqrt(la::DistanceSqEuclidean(root_point, point));
      dist = pdc::DistanceEuclidean<T>(root_point, point, sqrt(DBL_MAX));

      node_distances->Init(i, dist);

      point_set.PushBackCopy(node_distances);
    }
    DEBUG_ASSERT(point_set.size() == n - 1);

    T max_dist = max_set(&point_set);
    index_t max_scale = scale_of_distance(max_dist);
    
    if (flag != NULL) {
      TCoverTreeNode *root_node = 
	private_make_tree<TCoverTreeNode, T>(0, dataset, max_scale, 
					     max_scale, &point_set, 
					     &consumed_set, flag);

      return root_node;
    }
    else {
      TCoverTreeNode *root_node = 
	private_make_tree<TCoverTreeNode, T>(0, dataset, max_scale,
					     max_scale, &point_set,
					     &consumed_set);

      return root_node;
    }
  }

};
#endif
