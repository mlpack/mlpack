#ifndef KNN_NODE_H_
#define KNN_NODE_H_
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
class KnnNode {
 public:
	typedef typename TYPELIST::Precision_t Precision_t;
	typedef typename TYPELIST::Allocator_t Allocator_t;
	typedef typename TYPELIST::Metric_t    Metric_t;
	typedef typename TYPELIST::BoundingBox_t BoundingBox_t;
	typedef typename TYPELIST::NodeCachedStatistics_t NodeCachedStatistics_t;
	typedef typename TYPELIST::PointIdDiscriminator_t PointIdDiscriminator_t;
	typedef typename Allocator_t::template ArrayPtr<Precision_t> Array_t;
  typedef KnnNode<TYPELIST, diagnostic> Node_t;
	typedef typename Allocator_t::template Ptr<KnnNode> NodePtr_t;
	typedef Point<Precision_t, Allocator_t> Point_t;
	typedef Point<Precision_t, Loki::NullType> NullPoint_t;
	template<typename , bool> friend class KnnNodeTest; 
	struct NNResult {
		NNResult() : point_id_(0),
      distance_(numeric_limits<Precision_t>::max()) {
		}
		bool operator<(const NNResult &other) const {
		  if (point_id_==other.point_id_) {
			  return distance_<other.distance_;  
		  } else {
			  return  point_id_<other.point_id_;
			}
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
  class PairComparator {
	 public:
	   bool operator()(const pair<Precision_t, Point_t>  &a, 
			               const pair<Precision_t, Point_t>  &b)  {
	     return a.first<b.first;
		}
	};
  KnnNode();
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
  ~KnnNode();
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

	template<typename POINTTYPE>
  void FindNearest(POINTTYPE query_point, 
			             vector<pair<Precision_t, Point_t> >  &nearest, 
									 index_t knns, 
									 int32 dimension,
									 PointIdDiscriminator_t &discriminator,
									 ComputationsCounter<diagnostic> &comp);

  void FindAllNearest(NodePtr_t query_node,
                      Precision_t &max_neighbor_distance,
                      index_t knns,
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
	
	void set_kneighbors(index_t knns) {
	  kneighbors_.Reset(Allocator_t::template malloc<Point_t>
			                (num_of_points_*knns));
		distances_.Reset(Allocator_t::template malloc<Precision_t>
			               (num_of_points_*knns));
		distances_.Lock();
		for(index_t i=0; i< num_of_points_*knns; i++) {
		  distances_[i]=numeric_limits<Precision_t>::max();
		}
		distances_.Unlock();
	}

  void OutputNeighbors(NNResult *out, index_t knns); 
	void OutputNeighbors(FILE *fp, index_t knns); 	
	
	Precision_t get_min_dist_so_far() {
	  return min_dist_so_far_;
	}
	
	void set_min_dist_so_far(Precision_t distance) {
	  min_dist_so_far_=distance;
	}
	
	index_t get_node_id() {
	  return node_id_;
	}
  inline void LockPoints() {
	  points_.Lock();
		index_.Lock();
	}
	inline void UnlockPoints() {
	  points_.Unlock();
		index_.Unlock();
	}
	
  string Print(int32 dimension);
 
 private:
	BoundingBox_t box_;
  index_t node_id_;
  NodePtr_t left_;
  NodePtr_t right_;
	typename Allocator_t::template ArrayPtr<index_t> index_;
  typename Allocator_t::template ArrayPtr<Precision_t> points_;
	NodeCachedStatistics_t statistics_;
  index_t num_of_points_;
  typename Allocator_t::template ArrayPtr<Point_t> kneighbors_;
  typename Allocator_t::template ArrayPtr<Precision_t> distances_;

	Precision_t  min_dist_so_far_;	
	
};

#include "knn_node_impl.h"
#endif /*KNN_NODE_H_*/
