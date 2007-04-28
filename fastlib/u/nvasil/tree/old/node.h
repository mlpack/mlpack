#ifndef NODE_H_
#define NODE_H_
#include <new>
#include <limits>
#include "point.h"
#include "point_traits.h"
#include "traits_nearest_neighbor.h"
#include "point_identity_discriminator.h"
#include "computations_counter.h"
#include "data_reader.h"

template<typename PRECISION, 
         typename IDPRECISION,
         template<typename, typename, bool> class BOUNDINGBOX,
         class NODETYPE,
         typename ALLOCATOR,
				 bool diagnostic>
class Node {
 public:	
  typedef BOUNDINGBOX<PRECISION, ALLOCATOR, diagnostic> BoundingBox_t;
  typedef Node<PRECISION, IDPRECISION, BOUNDINGBOX, 
					     NODETYPE, ALLOCATOR, diagnostic> Node_t;
	struct Pivot_t {
		Pivot_t() {
			start_=0;
			num_of_points_=0;
		}
		Pivot_t(IDPRECISION start, IDPRECISION num_of_points, 
				typename BoundingBox_t::PivotData box_pivot_data) :
			start_(start), num_of_points_(num_of_points), 
			box_pivot_data_(box_pivot_data) {};
  	IDPRECISION start_;
  	IDPRECISION num_of_points_;
		typename BoundingBox_t::PivotData box_pivot_data_;
  };
	struct Result {
		Result() : point_id_(0),
               distance_(numeric_limits<PRECISION>::max())	   	{
		}
		bool operator<(const Result &other) const {
		  if (distance_==other.distance_ || true) {
			  return point_id_<other.point_id_;  
			}
			return distance_<other.distance_;
		}
		PRECISION get_distance() const {
		  return distance_;
		}
		IDPRECISION get_point_id() {
		  return point_id_; 
		}
	  IDPRECISION point_id_;
    Point<PRECISION, IDPRECISION, ALLOCATOR> nearest_;
		PRECISION	distance_;
	};
  // Use this for node
  Node(Pivot_t *pivot, IDPRECISION node_id);  
  // Use this for leaf
  Node(Pivot_t *pivot, IDPRECISION node_id,
                DataReader<PRECISION, IDPRECISION> *data); 
  ~Node();
  static void *operator new(size_t size);
  static void  operator delete(void *p);
  bool IsLeaf() {
		return !points_.IsNULL();
	}
	template<typename POINTTYPE>                   	
  pair<typename ALLOCATOR::template Ptr<NODETYPE>, 
		   typename ALLOCATOR::template Ptr<NODETYPE> > 
			 ClosestChild(POINTTYPE point, int32 dimension,
					 ComputationsCounter<diagnostic> &comp);
  pair<pair<typename ALLOCATOR::template Ptr<NODETYPE>, PRECISION>, 
	     pair<typename ALLOCATOR::template Ptr<NODETYPE>, PRECISION> > 
			 ClosestNode(typename ALLOCATOR::template Ptr<NODETYPE>,
					         typename ALLOCATOR::template Ptr<NODETYPE>,
									 int32 dimension,
							     ComputationsCounter<diagnostic> &comp);

// This one is using the default discriminator
// So it only checks if the points have identical id
	template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
  void FindNearest(POINTTYPE query_point, RETURNTYPE &nearest, 
                   PRECISION &distance, NEIGHBORTYPE range, int32 dimension,
                   ComputationsCounter<diagnostic> &comp);
// This one is using a custom discriminator  
// We use this for timit experiments so that we exclude points
// from the same speaker
	template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
  void FindNearest(POINTTYPE query_point, RETURNTYPE &nearest, 
                   PRECISION &distance, NEIGHBORTYPE range, int32 dimension,
                   PointIdentityDiscriminator<IDPRECISION> &discriminator,
									 ComputationsCounter<diagnostic> &comp);
// All nearest with a custom discriminator, this version
// stores the result on a vector on the leaf, very inefficient for
// large datasets
	template<typename NEIGHBORTYPE>
  void FindAllNearest(typename ALLOCATOR::template  Ptr<NODETYPE> query_node,
                      PRECISION &max_neighbor_distance,
                      PRECISION node_distance,
                      NEIGHBORTYPE range,
                      int32 dimension,
                      ComputationsCounter<diagnostic> &comp);
// The same as above with a custom discriminator also used for timit
  template<typename NEIGHBORTYPE>
  void FindAllNearest(typename ALLOCATOR::template  Ptr<NODETYPE> query_node,
                      PRECISION &max_neighbor_distance,
                      PRECISION node_distance,
                      NEIGHBORTYPE range,
                      int32 dimension,
                      PointIdentityDiscriminator<IDPRECISION> &discriminator,
                      ComputationsCounter<diagnostic> &comp);
// This one store the results directly on a memmory mapped file
// very efficient for large datasets
// Uses a default descriminator
  void FindAllNearest(typename ALLOCATOR::template  Ptr<NODETYPE> query_node,
                      PRECISION &max_neighbor_distance,
                      PRECISION node_distance,
                      int32 range,
                      int32 dimension,
                      ComputationsCounter<diagnostic> &comp);
 
	
// This one store the results directly on a memmory mapped file
// very efficient for large datasets
// Uses a custom descriminator
  void FindAllNearest(typename ALLOCATOR::template  Ptr<NODETYPE> query_node,
                      PRECISION &max_neighbor_distance,
                      PRECISION node_distance,
                      int32 range,
                      int32 dimension,
                      PointIdentityDiscriminator<IDPRECISION> &discriminator,
                      ComputationsCounter<diagnostic> &comp);
  
	string Print(int32 dimension);
	void PrintNeighbors(FILE *fp);
 	void InitKNeighbors(int32 range);
	void DeleteNeighbors();
  typename ALLOCATOR::template Ptr<NODETYPE>& get_left() {
  	return left_;
  }
  typename ALLOCATOR::template Ptr<NODETYPE>& get_right() {
  	return right_;
  }
  BoundingBox_t &get_box() {
	  return box_;
	}
  typename ALLOCATOR::template ArrayPtr<Point<PRECISION, IDPRECISION, ALLOCATOR> >&
	get_points() {
		  return points_;
	}
	IDPRECISION get_num_of_points() {
	  return num_of_points_;
	}
  vector<vector<pair<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> > >  *> *
  get_neighbors() {
	  return neighbors_;
	}
	Result *get_kneighbors() {
	  return kneighbors_;  
	}
	void set_kneighbors(Result *chunk) {
	  kneighbors_=chunk;
	}
	PRECISION get_min_dist_so_far() {
	  return min_dist_so_far_;
	}
	void set_min_dist_so_far(PRECISION distance) {
	  min_dist_so_far_=distance;
	}
 private:
  BoundingBox_t  box_;
  IDPRECISION node_id_;
  typename ALLOCATOR::template Ptr<NODETYPE> left_;
  typename ALLOCATOR::template Ptr<NODETYPE> right_;
  typename ALLOCATOR::template ArrayPtr<Point<PRECISION, 
					                              IDPRECISION, ALLOCATOR> > points_;
  IDPRECISION num_of_points_;
  // This one is used for all nearest neighbors
	union {
    vector<vector<pair<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> > >  *> *neighbors_;
    Result *kneighbors_;
  };
  // This one is used as well
	PRECISION min_dist_so_far_;	
};

#include "node_impl.h"
#endif /*NODE_H_*/
