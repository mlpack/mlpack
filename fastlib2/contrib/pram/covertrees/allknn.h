/**
 * @file allknn.h
 * 
 * This file defines the class AllKNN. This computes 
 * the nearest neighbors of query set from a 
 * reference set using the dual tree algorithm
 * 
 * It implements the Recursive breadth first search with 
 * the function:
 *
 * void RecursiveBreadthFirstSearch(ArrayList<index_t> *ind,
 *                                  ArrayList<T> *dist);
 *
 * It implements the Depth first search with 
 * the function:
 *
 * void DepthFirstSearch(ArrayList<index_t> *ind,
 *                       ArrayList<T> *dist);
 *
 * It implements the brute search with 
 * the function:
 *
 * void BruteNeighbors(ArrayList<index_t> *ind,
 *                     ArrayList<T> *dist);
 *
 */

#ifndef ALLKNN_H
#define ALLKNN_H

#include <fastlib/fastlib.h>
#include "cover_tree.h"
#include "ctree.h"
#include "distances.h"

const fx_entry_doc allknn_entries[] = {
  {"dim", FX_PARAM, FX_INT, NULL,
   " The dimension of the data we are dealing with.\n"},
  {"qsize", FX_PARAM, FX_INT, NULL,
   " The number of points in the query set.\n"},
  {"rsize", FX_PARAM, FX_INT, NULL, 
   " The number of points in the reference set.\n"},
  {"knns", FX_PARAM, FX_INT, NULL, 
   " The number of nearest neighbors we need to compute"
   " (defaults to 1).\n"},
  {"tree_building", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to record the time taken to build" 
   " the query and the reference tree.\n"},
  {"rbfs", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to record the time taken to do"
   " the recursive breadth first computation.\n"},
  {"dfs", FX_TIMER, FX_CUSTOM, NULL, 
   " The timer to record the time taken to do"
   " the depth first computation.\n"},
  {"brute", FX_TIMER, FX_CUSTOM, NULL, 
   " The timer to record the time taken to do"
   " the brute nearest neighbor computation.\n"},
  {"ec", FX_PARAM, FX_DOUBLE, NULL,
   " The expansion constant we will be using"
   " to make the tree (defaults to 1.3).\n"},
  {"print_tree", FX_PARAM, FX_BOOL, NULL,
   " The variable to decide whether to print"
   " the tree made or not (defaults to false).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc allknn_submodules[] = {
  {"ctree", &tree_construction_doc,
   " Responsible for building the normal cover tree.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc allknn_doc = {
  allknn_entries, allknn_submodules,
  " Performs dual-tree all nearest neighbors computation"
  " - recursive breadth first, depth first, brute.\n"
};

template<typename T>
class AllKNN {
  
  /**
   * This is a statistic class used to store 
   * information on the reference nodes when 
   * doing the dual tree search
   */
  class RefStat {
  private:
    // This is an array of distance from the 
    // query nodes encountered.
    ArrayList <T> distance_to_qnode_;
    
  public:

    RefStat() {
    }

    ~RefStat() {
    }

    // returns the distance to the most recent 
    // query node encountered
    inline T distance_to_qnode() {
      DEBUG_ASSERT(distance_to_qnode_.size() > 0);
      return distance_to_qnode_.back();
    }

    // this pushes the distance to the most recent 
    // query node encountered and is not pruned 
    // for that query node
    void set_distance_to_qnode(T new_dist) {
      distance_to_qnode_.PushBackCopy(new_dist);
    }

    // once you return from a particular query node, 
    // and about to go down another query node of 
    // the same level as this query, it is required 
    // to get rid of the distance to that node since 
    // it wouldn't be traversed again, and next distance
    // is the distance to the parent of that node, 
    // which is also the parent of the node about to 
    // traversed down.
    void pop_last_distance() {
      DEBUG_ASSERT(distance_to_qnode_.size() > 0);
      distance_to_qnode_.PopBack();
    }

    void Init() {
      distance_to_qnode_.Init(0);
    }
  };
  
  /**
   * This is a class which stores the reference points 
   * and their distance to the corresponding query 
   * nodes which were not pruned up until the 
   * leaf level of both the trees
   */
  class LeafNodes {
  private:
    // The reference point
    index_t point_;
    // the distance between the query and the 
    // reference node
    T dist_;

  public:
    LeafNodes() {
    }

    ~LeafNodes() {
    }

    void set_point(index_t point) {
      point_ = point;
    }

    void set_dist(T dist) {
      dist_ = dist;
    }

    index_t point() {
      return point_;
    }

    T dist() {
      return dist_;
    }

    void Init(index_t point, T dist) {
      set_point(point);
      set_dist(dist);
    }
  };
  
  typedef CoverTreeNode<RefStat, T> TreeType;
  
private:
  
  // The query and the references sets
  GenMatrix<T> queries_, references_;
  
  // The query tree
  TreeType *query_tree_;
  
  // The Reference tree
  TreeType *reference_tree_;
  
  // The number of query points and 
  // the number of reference points
  index_t num_queries_, num_refs_;
  
  // The number of nearest neighbors to be computed
  index_t knns_;
  
  // The datanode to store parameters for the object
  // of this class
  struct datanode *module_;

public:

  AllKNN() {
    query_tree_ = NULL;
    reference_tree_ = NULL;
  }

  ~AllKNN() {
    delete query_tree_;
    delete reference_tree_;
  }

  // getters

  TreeType *query_tree() {
    return query_tree_;
  }

  TreeType *reference_tree() {
    return reference_tree_;
  }

private:
  // This function compares the distance of two 
  // reference points to a particular query point
  inline T compare_(TreeType **p, TreeType **q) {
    return (*p)->stat().distance_to_qnode() 
      - (*q)->stat().distance_to_qnode();
  }
  
  // This functions swaps two elements in an array
  // Used as a MACRO in JL's code
  inline void swap_(TreeType **p, TreeType **q) {
    TreeType *temp = NULL;
    temp = *p;
    *p = *q;
    *q = temp;
  }

  // This function halfsorts the cover set in O(n)
  // so that the reference nodes closer to the 
  // query node are descended first so that you can 
  // prune away the further away reference points in 
  // the cover set earlier
  // This functions nonetheless does JUST the sorting
  // and not the pruning
  void halfsort_(ArrayList<TreeType*>*);

  
  // This functions sets the upper bound on the distance to 
  // farthest reference nodes which are possible NN 
  // of this query node....
  // The idea is that the present NN candidates of the 
  // parent of this node can give an upper bound on 
  // the NN candidates of this node
  inline void set_upper_bounds_(ArrayList<T> *upper_bounds, 
				index_t query_index,
				T d) {

    index_t start = query_index * knns_;
    T *begin = upper_bounds->begin() + start;
    T *end = begin + knns_;

    for (; begin != end; begin++) {
      *begin = d;
    }
  }

  // While pruning away reference points, we update the 
  // upper bound when we encounter points which are lesser 
  // the upper bound
  inline void update_upper_bounds_(ArrayList<T> *upper_bounds, 
				   index_t query_index,
				   T d) {

    index_t start = query_index * knns_;
    T *begin = upper_bounds->begin() + start;
    T *end = begin + knns_ - 1;
 
    for (; end != begin; begin++) {

      if (d < *(begin + 1)) {
	*begin = *(begin + 1);
      }
      else {
	*begin = d;
	break;
      }
    }

    if (end == begin) {
      *begin = d;
    }
  }

  // This is used for depth first search when a query node 
  // may be reached more than once, we set the upper_bound
  // according to the parent only if that upper_bound 
  // is less than the present upper_bound of query node
  // presently
  inline void set_update_upper_bounds_(ArrayList<T> *upper_bounds, 
				       index_t query_index,
				       T d){
    index_t start = query_index * knns_;
    T *begin = upper_bounds->begin() + start;
    T *end = begin + knns_;
 
    for (; end != begin; begin++) {
      if (d < *begin) {
	*begin = d;
      }
      else {
	break;
      }
    }
  }

  // This function is used for the recursive breadth first version
  // to remove the last distances of the reference node statistics 
  // so that a new query node can descend down that node
  // Here the nodes are specifically leaf reference nodes
  void reset_leaf_nodes_(ArrayList<TreeType*>*);

  // This function is used for the recursive breadth first version
  // to remove the last distances of the reference node statistics 
  // so that a new query node can descend down that node
  // Here the nodes are specifically non-leaf nodes so this 
  // is done for all the scales which were not pruned 
  // by the previous query node at this level
  void reset_cover_sets_(ArrayList<ArrayList<TreeType*> >*, 
			 index_t, index_t);

  // This function copy all the reference leaf nodes which are 
  // within the upper bound of a particular query node
  // on the basis of the distance os these leaf nodes from 
  // its parent
  // First it upper bounds with the distance of the 
  // reference point to the parent and 
  // its distance to the parent
  // Then it upper bounds with just its distance to the 
  // farthest descendant
  void CopyLeafNodes_(TreeType*, 
		      ArrayList<T>*_bounds,
		      ArrayList<TreeType*>*,
		      ArrayList<TreeType*>*);
  
  // This function copy all the reference non-leaf nodes which are 
  // within the upper bound of a particular query node
  // on the basis of the distance os these leaf nodes from 
  // its parent
  // First it upper bounds with the distance of the 
  // reference point to the parent and 
  // its distance to the parent and the distance of 
  // the farthest descendant of the reference node
  // Then it upper bounds with just its distance to the 
  // farthest descendant
  void CopyCoverSets_(TreeType*, 
		      ArrayList<T>*, 
		      ArrayList<ArrayList<TreeType*> >*, 
		      ArrayList<ArrayList<TreeType*> >*, 
		      index_t, index_t);

  // This function takes the present cover set of a query 
  // node and descends the reference nodes and keeps those 
  // reference nodes which are within the range of the 
  // query node and its descendants, i.e, keeps the 
  // nodes which are potential NN of not only the query node 
  // but also its descendants
  //
  // You descend for just a particular scale and 
  // set max_scale as the one which is the maximum 
  // scale of the any of the children of the 
  // reference cover set
  void DescendTheRefTree_(TreeType*, 
			  ArrayList<T>*, 
			  ArrayList<ArrayList<TreeType*> >*, 
			  ArrayList<TreeType*>*,
			  index_t, index_t*);

  // This function descends the query tree when we have reached
  // the leaf level of the reference tree.
  // if the query is also a leaf, then select its 
  // NN from the leaf set which contains all the 
  // candidate NN
  void ComputeBaseCase_(TreeType*, 
			ArrayList<T>*,
			ArrayList<TreeType*>*, 
			ArrayList<index_t>*);

  // This function perform the whole recursive 
  // breadth first recursion
  void ComputeNeighborRecursion_(TreeType*, 
				 ArrayList<T>*, 
				 ArrayList<ArrayList<TreeType*> >*,
				 ArrayList<TreeType*>*,
				 index_t, index_t,
				 ArrayList<index_t>*);

public:
  /**
   * This function computes the nearest neighbors of te 
   * query set from the reference set using recursive
   * breadth search of the cover tree formed from 
   * the reference set and in a depth first manner 
   * for cover tree formed from the query set.
   * 
   * Use:
   * @code
   * AllKNN<T> allknn;
   * ....
   * ArrayList<T> neighbor_distances;
   * ArrayList<index_t> neighbor_indices;
   * allknn.RecursiveBreadthFirstSearch(&neighbor_indices, &neighbor_distances);
   * @endcode
   */
  void RecursiveBreadthFirstSearch(ArrayList<index_t> *neighbor_indices, 
				   ArrayList<T> *neighbor_distances) {

    GenVector<T> q_root, r_root;
    ArrayList<ArrayList<TreeType*> > cover_sets;
    ArrayList<TreeType*> leaf_nodes;
    index_t current_scale = 0, max_scale = 0;

    queries_.MakeColumnVector(query_tree_->point(), &q_root);
    references_.MakeColumnVector(reference_tree_->point(), &r_root);
    neighbor_indices->Init(knns_* num_queries_);
    for (index_t i = 0; i < neighbor_indices->size(); i++) {
      (*neighbor_indices)[i] = -1;
    }
    neighbor_distances->Init(knns_ * num_queries_);
    cover_sets.Init(101);
    for (index_t i = 0; i < 101; i++) {
      cover_sets[i].Init(0);
    }
    leaf_nodes.Init(0);

    set_upper_bounds_(neighbor_distances, query_tree_->point(), DBL_MAX);
  
    //T dist = sqrt(la::DistanceSqEuclidean(q_root, r_root));
    T dist = pdc::DistanceEuclidean<T>(q_root, r_root, sqrt(DBL_MAX));

    update_upper_bounds_(neighbor_distances, query_tree_->point(), dist);

    reference_tree_->stat().set_distance_to_qnode(dist);
    cover_sets[0].PushBackCopy(reference_tree_);

    // descending the query root along with the reference root

    ComputeNeighborRecursion_(query_tree_, neighbor_distances, &cover_sets, 
			      &leaf_nodes, current_scale, max_scale, 
			      neighbor_indices);
  }

private:
  // This function computes the nearest neighbors 
  // using depth first search
  void DepthFirst_(TreeType*, TreeType*, 
		   ArrayList<ArrayList<LeafNodes> >*,
		   ArrayList<T>*); 
  
public:
  /**
   * This function computes the nearest neighbors of te 
   * query set from the reference set using depth first 
   * search of the cover tree formed from the reference
   * set and the query set.
   * 
   * Use:
   * @code
   * AllKNN<T> allknn;
   * ....
   * ArrayList<T> neighbor_distances;
   * ArrayList<index_t> neighbor_indices;
   * allknn.DepthFirstSearch(&neighbor_indices, &neighbor_distances);
   * @endcode
   */
  void DepthFirstSearch(ArrayList<index_t> *neighbor_indices, 
			ArrayList<T> *neighbor_distances) {
    
    GenVector<T> q_root, r_root;
    ArrayList<ArrayList<LeafNodes> > neighbors;
    
    neighbors.Init(num_queries_);
    for (index_t i = 0; i < neighbors.size(); i++) {
      neighbors[i].Init(0);
    }
    
    queries_.MakeColumnVector(query_tree_->point(), &q_root);
    references_.MakeColumnVector(reference_tree_->point(), &r_root);
    neighbor_indices->Init(knns_* num_queries_);
    neighbor_distances->Init(knns_ * num_queries_);

    for (index_t i = 0; i < num_queries_; i++) {
      set_upper_bounds_(neighbor_distances, i, DBL_MAX);
    }
    
    //T dist = sqrt(la::DistanceSqEuclidean(q_root, r_root));
    T dist = pdc::DistanceEuclidean<T>(q_root, r_root, sqrt(DBL_MAX));
    
    update_upper_bounds_(neighbor_distances, query_tree_->point(), dist);
    
    reference_tree_->stat().set_distance_to_qnode(dist);
    DepthFirst_(query_tree_, reference_tree_, 
		&neighbors, neighbor_distances);

    for (index_t i = 0; i < num_queries_; i++) {
      T *query_ub = neighbor_distances->begin() + i * knns_;
      DEBUG_ASSERT_MSG(neighbors[i].size() >= knns_, 
		       "%"LI"d neighbors candidates found for point %"LI"d",
		       neighbors[i].size(), i+1);
      
      LeafNodes *begin_nn = neighbors[i].begin();
      LeafNodes *end_nn = neighbors[i].end();
      T *end = query_ub + knns_;
      index_t *indices = neighbor_indices->begin() + i * knns_;
      ArrayList<bool> flags;

      flags.Init(knns_);
      for (index_t in = 0; in < knns_; in++) {
	flags[in] = 0;
      }

      for (index_t j = 0; j < knns_ && begin_nn != end_nn; begin_nn++) {
	if (begin_nn->dist() <= *query_ub) {
	  //NOTIFY("%"LI"d %"LI"d -> %lf, ub = %lf", 
	  //i+1, begin_nn->point()+1, begin_nn->dist(), *query_ub);
	  index_t k = 0;
	  while (begin_nn->dist() > *(end - k- 1)) { 
	    k++;
	  }
	  while (flags[k] == 1 && k < knns_ - 1) {
	    k++;
	  }
	  if (k < knns_ && flags[k] == 0) {
	    *(indices + k) = begin_nn->point();
	    flags[k] = 1;
	    j++;	  
	  }
	}
      }
    }
  }

  /**
   * This function computes the nearest neighbors of the 
   * query set from the reference set using brute 
   * computation
   * 
   * Use:
   * @code
   * AllKNN<T> allknn;
   * ....
   * ArrayList<T> neighbor_distances;
   * ArrayList<index_t> neighbor_indices;
   * allknn.BruteNeighbors(&neighbor_indices, &neighbor_distances);
   * @endcode
   */
  void BruteNeighbors(ArrayList<index_t> *neighbor_indices, 
		      ArrayList<T> *neighbor_distances) {

    ArrayList<LeafNodes> neighbors;
    neighbor_indices->Init(num_queries_ * knns_);
    neighbor_distances->Init(num_queries_ * knns_);

    for (index_t i = 0; i < num_queries_; i++) {
      set_upper_bounds_(neighbor_distances, i, DBL_MAX);
      GenVector<T> q;
      T *query_upper_bound = neighbor_distances->begin() 
	+ knns_ * i;
      neighbors.Init(0);

      queries_.MakeColumnVector(i, &q);
      for (index_t j = 0; j < num_refs_; j++) {
	GenVector<T> r;
	references_.MakeColumnVector(j, &r);

	T dist = pdc::DistanceEuclidean<T>(q, r, DBL_MAX);
	if (dist <= *query_upper_bound) {
	  update_upper_bounds_(neighbor_distances, 
			       i, dist);
	  LeafNodes leaf;
	  leaf.Init(j, dist);
	  neighbors.PushBackCopy(leaf);
	}
      }

      DEBUG_ASSERT_MSG(neighbors.size() >= knns_, 
		       "%"LI"d neighbors candidates found for point %"LI"d",
		       neighbors.size(), i+1);
      
      LeafNodes *begin_nn = neighbors.begin();
      LeafNodes *end_nn = neighbors.end();
      T *end = query_upper_bound + knns_;
      index_t *indices = neighbor_indices->begin() + i * knns_;
      ArrayList<bool> flags;

      flags.Init(knns_);
      for (index_t in = 0; in < knns_; in++) {
	flags[in] = 0;
      }

      for (index_t j = 0; j < knns_ && begin_nn != end_nn; begin_nn++) {
	if (begin_nn->dist() <= *query_upper_bound) {
	  index_t k = 0;
	  while (begin_nn->dist() > *(end - k- 1)) { 
	    k++;
	  }
	  while (flags[k] == 1 && k < knns_ - 1) {
	    k++;
	  }
	  if (k < knns_ && flags[k] == 0) {
	    *(indices + k) = begin_nn->point();
	    flags[k] = 1;
	    j++;	  
	  }
	}
      }
      neighbors.Renew();
    }
  }		

  /**
   * If we are doing the all nearest neighbor search for the 
   * bichromatic set, we use this Init()
   * 
   * @code
   * GenMatrix<T> references, queries;
   * .....
   * AllKNN<T> allknn;
   * datanode *allknn_module = fx_submodule(root, "allknn");
   * 
   * allknn.Init(queries, references, allknn_module);
   * @endcode
   */  
  void Init(const GenMatrix<T>& queries, 
	    const GenMatrix<T>& references, 
	    struct datanode *module) {

    module_ = module;
    queries_.Copy(queries);
    num_queries_ = queries_.n_cols();
    references_.Copy(references);
    num_refs_ = references_.n_cols();

    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

    knns_ = fx_param_int(module_, "knns", 1);
    T base = fx_param_double(module_, "ec", 1.3);

    fx_timer_start(module_, "tree_building");
    datanode *ctree_module = fx_submodule(module_, "ctree");
    query_tree_ = ctree::MakeCoverTree<TreeType, T>(queries_, base, 
						    ctree_module);
    reference_tree_ = ctree::MakeCoverTree<TreeType, T>(references_, base,
							ctree_module);
    fx_timer_stop(module_, "tree_building");

    
    if(fx_param_bool(module_, "print_tree", 0)){
      NOTIFY("Query Tree:");
      ctree::PrintTree<TreeType>(query_tree_);
      NOTIFY("Reference Tree:");
      ctree::PrintTree<TreeType>(reference_tree_);
    }
    return;
  }

  /**
   * If we are doing the all nearest neighbor search for the 
   * monochromatic set, we use this Init()
   * 
   * @code
   * GenMatrix<T> reference;
   * .....
   * AllKNN<T> allknn;
   * datanode *allknn_module = fx_submodule(root, "allknn");
   * 
   * allknn.Init(references, allknn_module);
   * @endcode
   */
  void Init(const GenMatrix<T>& references, 
	    struct datanode *module) {

    module_ = module;
    queries_.Copy(references);
    num_queries_ = queries_.n_cols();
    references_.Copy(references);
    num_refs_ = references_.n_cols();

    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

    knns_ = fx_param_int(module_, "knns", 1);
    T base = fx_param_double(module_, "ec", 1.3);

    fx_timer_start(module_, "tree_building");
    datanode *ctree_module = fx_submodule(module_, "ctree");
    reference_tree_ = ctree::MakeCoverTree<TreeType, T>(references_, base,
							ctree_module);
    query_tree_ = reference_tree_;
    fx_timer_stop(module_, "tree_building");

    
    if(fx_param_bool(module_, "print_tree", 0)){
      NOTIFY("Query Tree:");
      ctree::PrintTree<TreeType>(query_tree_);
      NOTIFY("Reference Tree:");
      ctree::PrintTree<TreeType>(reference_tree_);
    }
    return;
  }
};

#include "allknn_impl.h"
#endif
