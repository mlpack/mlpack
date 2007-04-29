#ifndef NODE_IMPL_H_
#define NODE_IMPL_H_

#define __TEMPLATE__                                               \
template<typename TYPELIST, bool diagnostic >                                                                  

#define __NODE__                                                   \
	Node<TYPELIST, diagnostic>                  
 
__TEMPLATE__    
__NODE__::Node() {
        	
  left_.SetNULL();
  right_.SetNULL();
  points_.SetNULL();
  kneighbors_=NULL;
	node_id_ = node_id;
	min_dist_so_far_=numeric_limits<Precision_t>::max();
}

__TEMPLATE__
__NODE__::Init(const BoundingBox_t &box, 
	             const NodeCachedStatistics_t &statistics,		
			         index_t node_id,
			         index_t num_of_points) {
  box_=box;
  statistics_=statistics;
  node_id_ = node_id;
	num_of_points_ = num_of_points;
	points_
}

__TEMPLATE__
__NODE__::Init(BoundingBox_t &box,
			         NodeCachedStatistics_t &statistics,
			         index_t node_id,
			         BinaryDataset<Precision_t> *dataset,
               index_t start,
               index_t num_of_points,
							 int32 dimension) {
	box_=box;
	statistics_=statistics;
	node_id_ = node_id;
	num_of_points_ = num_of_points;
	points_.Reset(Allocator_t::malloc<Precision_t>(num_of_points_*dimension_));
  index_.Reset(Allocator_t::malloc<index_t>(num_of_points_));
	for(index_t i=start; i<start+num_of_points; i++) {
	  for(int32 j=0; j<dimension; j++) {
		  points_[i*dimension+j]=dataset_.At(i,j);
	  }
		index_[i]=dataset->get_id(i);
	}
} 

__TEMPLATE__
__NODE__::~Node() {
}

__TEMPLATE__
void *__NODE__::operator new(size_t size) {
  return Allocator_t::allocator->AllignedAlloc(size);
}
     	
__TEMPLATE__
void __NODE__::operator delete(void *p) {
}
__TEMPLATE__
void __NODE__::InitKNeighbors(int32 knns) {
	for(index_t i=0; i<num_of_points_; i++) {
	  for(int32 j=0; j<knns; j++) {
		  kneighbors_[i*range+j].point_id_=index_[i];
		}
	} 
}
                 	
__TEMPLATE__
template<typename POINTTYPE>
pair<Allocator_t::template Ptr<NODETYPE>, 
	   Allocator_t::template Ptr<NODETYPE> >                     
__NODE__::ClosestChild(POINTTYPE point, int32 dimension, 
		                   ComputationsCounter<diagnostic> &comp) {
  return box_.ClosestChild(left_, right_, point, dimension, comp);
}

__TEMPLATE__
inline pair<pair<Allocator_t::template Ptr<NODETYPE>, Precision_t>, 
		 pair<Allocator_t::template Ptr<NODETYPE>, Precision_t> > 
__NODE__::ClosestNode(Allocator_t::template Ptr<NODETYPE> ptr1,
		                  Allocator_t::template Ptr<NODETYPE> ptr2,
									    int32 dimension,
							        ComputationsCounter<diagnostic> &comp) {
	Precision_t dist1 = BoundingBox_t::Distance(box_, ptr1->get_box(), 
			                                      dimension, comp);
	Precision_t dist2 = BoundingBox_t::Distance(box_, ptr2->get_box(), 
			                                      dimension, comp);
  if (dist1<dist2) {
	  return make_pair(make_pair(ptr1, dist1), make_pair(ptr2, dist2));
	} else {
	  return make_pair(make_pair(ptr2,dist2), make_pair(ptr1, dist1));
	}
}

__TEMPLATE__
template<typename POINTTYPE, typename NEIGHBORTYPE>
inline void __NODE__::FindNearest(POINTTYPE query_point, 
		               vector<pair<Precision_t, Point_t> > &nearest, 
                   Precision_t &distance, NEIGHBORTYPE range, int32 dimension,
                   PointIdentityDiscriminator &discriminator,
									 ComputationsCounter<diagnostic> &comp) {
  
	for(index_t i=0; i<num_of_points_; i++) {
  	comp.UpdateDistances();
  	//  we have to check if we are comparing the point with itself
 	  if (unlikely(discriminator(index_[i], 
			PointTraits<POINTTYPE>::GetPointId(query_point, dimension))==true)) {
  	 	continue;
 	  } 

		Precision_t dist = BoundingBox_t::
			template Distance(query_point, 
			                  points_.get()+i*dimension,
			  							  dimension);
		// In case it is range nearest neighbors
		if (boost::is_floating_point<NEIGHBORTYPE>::value==true) {
		  if (dist<=range){
				Point_t point;
				point.Alias(points_.get()+i*dimension, index_[i]);
			  nearest.push_back(make_par(dist, points));
			 }
		}
  // for k-nearest neighbors
	if (boost::is_floating_point<NEIGHBORTYPE>::value==false) {
  	std::partial_sort(nearest[0], nearest[nearest.size]);
		if (nearest.size()>range) {
		  nearest.remove(&nearest[range], nearest.end());
		}
	}
}

__TEMPLATE__
template<typename NEIGHBORTYPE>
inline void __NODE__::FindAllNearest(
		                Allocator_t::template Ptr<NODETYPE> query_node,
                    Precision_t &max_neighbor_distance,
                    NEIGHBORTYPE range,
                    int32 dimension,
                    PointIdentityDiscriminator &discriminator,
                    ComputationsCounter<diagnostic> &comp) {
  
  Precision_t max_local_distance = numeric_limits<Precision_t>::min();
  for(index_t i=0; i<query_node->num_of_points_; i++) {
		// for k nearest neighbors
		if (boost::is_floating_point<NEIGBORTYPE>::value==false) {
     	// get the current maximum distance for the specific point
  	  Precision_t distance = query_node->kneighbors_[i*range+range-1].distance_;
		} else {
		  distance=range;
		}
    // We should check whether this speeds up or slows down 
		// the performance 
    comp.UpdateComparisons();
    if (this->box_.CrossesBoundaries(query_node->points_.get()+i*dimension, 
                                               dimension,
                                               distance, 
                                               comp)) {
		 // for k nearest neighbors
     if (boost::is_floating_point<NEIGHBORTYPE>::value==false) {                                          	
			 vector<pair<Precision_t, Point_t> > > temp(range);
			 for(int32 j=0; j<range; j++) {
			   temp[j].first=query_node->kneighbors_[i*range+j].distance_;
				 temp[j].second=query_node->kneighbors_[i*range+j].nearest_;
			 }
       FindNearest(query_node->points_[i], temp, 
                   distance, range, dimension,
				 				   discriminator,	comp);
			 for(int32 j=range-1; j>=0; j--) {
			   if (query_node->kneighbors_[i*range+j].nearest_==temp[j].second) {
				   break;
			   }
			   query_node->kneighbors_[i*range+j].distance_=temp[j].first;
			   query_node->kneighbors_[i*range+j].nearest_=temp[j].second;
			}
     // Estimate the  maximum nearest neighbor distance
      comp.UpdateComparisons();
      if  (max_local_distance < temp.back().first) {
        max_local_distance = temp.back().first;
      }

		 } else {
		 // for range nearest neighbors
	     vector<pair<Precision_t, Point_t> > > temp;
			 temp.clear();
		   FindNearest(query_node->points_[i], temp, 
                   distance, range, dimension,
				 				   discriminator,	comp);
			 for(index_t j=0; j<temp.size(); j++) {
				 NNResult result;
				 result.point_id_=query_node->points_[i].get_id();
				 result.nearest_=temp[j].first;
				 result.distance_=temp[j].second;
				 FATAL(fwrite(&result, sizeof(NNResult), 1, range_nn_fp_)!=1, 
						   "Error while writing range nearest neighbors: %s\n",
						    strerror(errno));
			 }                                     
  }
	if (boost::is_floating_point<NEIGBORTYPE>::value==true) {
	  max_local_distance=range;
	}	
}

#undef __TEMPLATE__
#undef __NODE__     	               	
#endif /*NODE_IMPL_H_*/
