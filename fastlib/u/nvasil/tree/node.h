#ifndef NODE_H_
#define NODE_H_
#include <new>
#include <limits>
#include "u/nvasil/loki/TypeTraits.h"
#include "u/nvasil/loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "point.h"
#include "point_identity_discriminator.h"
#include "computations_counter.h"
#include "u/nvasil/dataset/binary_dataset.h"

template<typename TYPELIST,
         bool diagnostic>
class Node {
 public:
	typedef typename TYPELIST::Precision_t Precision_t;
	typedef typename TYPELIST::Allocator_t Allocator_t;
	typedef typename TYPELIST::Metric_t    Metric_t;
	typedef typename TYPELIST::BoundingBox_t BoundingBox_t;
	typedef typename TYPELIST::NodeCachedStatistics_t NodeCachedStatistics_t;
	typedef typename TYPELIST::PointIdDiscriminator_t PointIdDiscriminator_t;
	typedef typename Allocator_t::template ArrayPtr<Precision_t> Array_t;
  typedef Node<TYPELIST, diagnostic> Node_t;
	typedef typename Allocator_t::template Ptr<Node> NodePtr_t;
	typedef Point<Precision_t, Allocator_t> Point_t;
	template<typename , bool> friend class NodeTest; 
	struct NNResult {
		NNResult() : point_id_(0),
      distance_(numeric_limits<Precision_t>::max()) {
		}
		bool operator<(const NNResult &other) const {
		  if (distance_==other.distance_ || true) {
			  return point_id_<other.point_id_;  
			}
			return distance_<other.distance_;
		}
		Precision_t get_distance() const {
		  return distance_;
		}
		index_t get_point_id() {
		  return point_id_; 
		}
	  index_t point_id_;
    Point_t nearest_;
		Precision_t	distance_;
	};
  Node();
	// Use this for node
  void Init(const BoundingBox_t &box, 
	          const NodeCachedStatistics_t &statistics,		
			      index_t node_id,
			      index_t num_of_points);  
  // Use this for leaf
  void Init(const BoundingBox_t &box,
			      const NodeCachedStatistics_t &statistics,
			      index_t node_id,
			      index_t start,
			      index_t num_of_points,
			      int32 dimension,
            BinaryDataset<Precision_t> *dataset); 
  ~Node();
  static void *operator new(size_t size);
  static void  operator delete(void *p);
  bool IsLeaf() {
		return !points_.IsNULL();
	}
	template<typename POINTTYPE>                   	
  pair<NodePtr_t, NodePtr_t> 
			 ClosestChild(POINTTYPE point, 
					          int32 dimension,
					          ComputationsCounter<diagnostic> &comp);

  pair<pair<NodePtr_t, Precision_t>, 
	     pair<NodePtr_t, Precision_t> > 
			 ClosestNode(NodePtr_t, 
					         NodePtr_t,
									 int32 dimension,
							     ComputationsCounter<diagnostic> &comp);

// This one is using a custom discriminator  
// We use this for timit experiments so that we exclude points
// from the same speaker
	template<typename POINTTYPE, typename NEIGHBORTYPE>
  void FindNearest(POINTTYPE query_point, 
			             vector<pair<Precision_t, Point_t> >  &nearest, 
									 NEIGHBORTYPE range, 
									 int32 dimension,
									 PointIdDiscriminator_t &discriminator,
									 ComputationsCounter<diagnostic> &comp);

// This one store the results directly on a memmory mapped file
// for k-nearest neighbors and to a normal file for range nearest neighbors
// very efficient for large datasets
// Uses a custom descriminator
  template<typename NEIGHBORTYPE>
  void FindAllNearest(NodePtr_t query_node,
                      Precision_t &max_neighbor_distance,
                      NEIGHBORTYPE range,
                      int32 dimension,
											PointIdDiscriminator_t &discriminator,
                      ComputationsCounter<diagnostic> &comp);
  
  NodePtr_t& get_left() {
  	return left_;
  }
  NodePtr_t& get_right() {
  	return right_;
  }
  BoundingBox_t &get_box() {
	  return box_;
	}
  typename Allocator_t::template ArrayPtr<Point_t>& 	get_points() {
		  return points_;
	}
	index_t get_num_of_points() {
	  return num_of_points_;
	}
	
	NNResult *get_kneighbors() {
	  return kneighbors_;  
	}
	void set_kneighbors(NNResult *chunk, uint32 knns) {
	  kneighbors_=chunk;
		for(index_t i=0; i< num_of_points_; i++) {
		  for(index_t j=0; j<(index_t)knns; j++) {
	      kneighbors_[i*knns+j].point_id_ =
				index_[i];	
			}
		}
	}
	
	void InitKNeighbors(int32 knns);

	void set_range_neighbors(FILE *fp) {
    range_nn_fp_=fp;	
	}
	FILE *get_range_nn_fp() {
	  return range_nn_fp_;
	}
	Precision_t get_min_dist_so_far() {
	  return min_dist_so_far_;
	}
	void set_min_dist_so_far(Precision_t distance) {
	  min_dist_so_far_=distance;
	}

 private:
	BoundingBox_t box_;
  index_t node_id_;
  NodePtr_t left_;
  NodePtr_t right_;
	typename Allocator_t::template ArrayPtr<index_t> index_;
  typename Allocator_t::template ArrayPtr<Precision_t> points_;
	NodeCachedStatistics_t statistics_;
  index_t num_of_points_;
	union {
    NNResult *kneighbors_;
		FILE *range_nn_fp_;
	};
	Precision_t  min_dist_so_far_;	
	class PairComparator {
	 public:
	   bool operator()(const pair<Precision_t, Point_t> &a, 
			               const pair<Precision_t, Point_t>  &b)  {
	     return a.first<b.first;
		}
	};
};

#include "node_impl.h"
#endif /*NODE_H_*/
