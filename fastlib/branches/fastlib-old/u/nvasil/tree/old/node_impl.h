#ifndef NODE_IMPL_H_
#define NODE_IMPL_H_

#define __TEMPLATE__                                               \
template<typename PRECISION,                                       \
         typename IDPRECISION,                                     \
         template<typename, typename, bool> class BOUNDINGBOX,     \
         class NODETYPE,                                           \
         typename ALLOCATOR,                                       \
         bool diagnostic >                                                                  

#define __NODE__                                                   \
	Node<PRECISION, IDPRECISION,																		 \
  BOUNDINGBOX, NODETYPE, ALLOCATOR, diagnostic>                  
 
__TEMPLATE__    
__NODE__::Node(Pivot_t *pivot, IDPRECISION node_id) :
        box_(pivot->box_pivot_data_) {
        	
  left_.SetNULL();
  right_.SetNULL();
  points_.SetNULL();
  neighbors_=NULL;
	node_id_ = node_id;
	min_dist_so_far_=numeric_limits<PRECISION>::max();
}
__TEMPLATE__
__NODE__::Node(Pivot_t *pivot, IDPRECISION node_id,
               DataReader<PRECISION, IDPRECISION> *data) : 
         box_(pivot->box_pivot_data_), node_id_(node_id),
         num_of_points_(pivot->num_of_points_),
         min_dist_so_far_(numeric_limits<PRECISION>::max()){

  points_.Reset(ALLOCATOR::allocator-> template Alloc<Point<PRECISION, 
	  			 IDPRECISION, ALLOCATOR> >(num_of_points_));
      	
  for(IDPRECISION i=0; i<num_of_points_; i++) {
  	points_[i].Reset(
				ALLOCATOR::allocator-> template Alloc<PRECISION> (
				pivot->box_pivot_data_.dimension_));
  	for(int32 j=0; j<pivot->box_pivot_data_.dimension_; j++) {
  	  points_[i][j] = data->At(pivot->start_+i)[j];
  	}
  	points_[i].set_id(data->GetId(pivot->start_+i));         	            	
  }
  left_.SetNULL();
  right_.SetNULL();
  neighbors_=NULL;
}             

__TEMPLATE__
__NODE__::~Node() {
}

__TEMPLATE__
void *__NODE__::operator new(size_t size) {
  return ALLOCATOR::allocator->AllignedAlloc(size);
}
     	
__TEMPLATE__
void __NODE__::operator delete(void *p) {
}
__TEMPLATE__
void __NODE__::InitKNeighbors(int32 range) {
	for(uint32 i=0; i<num_of_points_; i++) {
	  for(int32 j=0; j<range; j++) {
		  kneighbors_[i*range+j].point_id_=points_[i].get_id();
			}
	}
  
}
                 	
__TEMPLATE__
template<typename POINTTYPE>
pair<typename ALLOCATOR::template Ptr<NODETYPE>, 
	   typename ALLOCATOR::template Ptr<NODETYPE> >                     
__NODE__::ClosestChild(POINTTYPE point, int32 dimension, 
		                   ComputationsCounter<diagnostic> &comp) {
  return box_.ClosestChild(left_, right_, point, dimension, comp);
}

__TEMPLATE__
inline pair<pair<typename ALLOCATOR::template Ptr<NODETYPE>, PRECISION>, 
		 pair<typename ALLOCATOR::template Ptr<NODETYPE>, PRECISION> > 
__NODE__::ClosestNode(typename ALLOCATOR::template Ptr<NODETYPE> ptr1,
		                  typename ALLOCATOR::template Ptr<NODETYPE> ptr2,
									    int32 dimension,
							        ComputationsCounter<diagnostic> &comp) {
	PRECISION dist1 = BoundingBox_t::Distance(box_, ptr1->get_box(), 
			                                      dimension, comp);
	PRECISION dist2 = BoundingBox_t::Distance(box_, ptr2->get_box(), 
			                                      dimension, comp);
  if (dist1<dist2) {
	  return make_pair(make_pair(ptr1, dist1), make_pair(ptr2, dist2));
	} else {
	  return make_pair(make_pair(ptr2,dist2), make_pair(ptr1, dist1));
	}


}

__TEMPLATE__
template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
inline void __NODE__::FindNearest(POINTTYPE query_point, RETURNTYPE &nearest, 
                   PRECISION &distance, NEIGHBORTYPE range, int32 dimension,
                   ComputationsCounter<diagnostic> &comp) {
	PointIdentityDiscriminator<IDPRECISION> discriminator;
	FindNearest(query_point, nearest, distance, range,
		        	dimension, discriminator, comp);
}

__TEMPLATE__
template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
inline void __NODE__::FindNearest(POINTTYPE query_point, RETURNTYPE &nearest, 
                   PRECISION &distance, NEIGHBORTYPE range, int32 dimension,
                   PointIdentityDiscriminator<IDPRECISION> &discriminator,
									 ComputationsCounter<diagnostic> &comp) {
  for(uint32 i=0; i<num_of_points_; i++) {
  	comp.UpdateDistances();
  	//  we have to check if we are comparing the point with itself
 	  if (discriminator(points_[i].get_id(), 
			PointTraits<POINTTYPE, IDPRECISION>::
      GetPointId(query_point, dimension))==true) {
  	 	continue;
 	  } 

		PRECISION dist = BoundingBox_t:: template Distance(query_point, 
				                                               points_[i],
																											 dimension);
  	TraitsNearestNeighbor<PRECISION, IDPRECISION, ALLOCATOR>::
		Push(nearest, points_[i], dist, distance, range);
  }
  TraitsNearestNeighbor<PRECISION, IDPRECISION, ALLOCATOR>::
		Adjust(nearest, distance, range);
}

__TEMPLATE__
template<typename NEIGHBORTYPE>
inline void __NODE__::FindAllNearest(typename ALLOCATOR::template Ptr<NODETYPE> query_node,
                    PRECISION &max_neighbor_distance,
                    PRECISION node_distance,
                    NEIGHBORTYPE range,
                    int32 dimension,
                    ComputationsCounter<diagnostic> &comp) {
 
	PointIdentityDiscriminator<IDPRECISION> discriminator;
  FindAllNearest(query_node,
                 max_neighbor_distance,
                 node_distance,
                 range,
                 dimension,
								 discriminator,
                 comp);
} 
__TEMPLATE__
template<typename NEIGHBORTYPE>
inline void __NODE__::FindAllNearest(typename ALLOCATOR::template Ptr<NODETYPE> query_node,
                    PRECISION &max_neighbor_distance,
                    PRECISION node_distance,
                    NEIGHBORTYPE range,
                    int32 dimension,
										PointIdentityDiscriminator<IDPRECISION> &discriminator,
                    ComputationsCounter<diagnostic> &comp) {
  if (query_node->neighbors_ == NULL) {
  	query_node->neighbors_ = new  
  	    vector<vector<pair<PRECISION, Point<PRECISION, 
		                       IDPRECISION, 
													 ALLOCATOR> > > *>(query_node->num_of_points_);
  	for(IDPRECISION i=0; i<query_node->num_of_points_; i++) {
  	  query_node->neighbors_->at(i) = 
  	      new vector<pair<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> > >();
  	  TraitsNearestNeighbor<PRECISION, 
				                    IDPRECISION, 
														ALLOCATOR>::
														Prepare(*(query_node->get_neighbors()->at(i)), range);    
  	} 
  }
  PRECISION max_local_distance = numeric_limits<PRECISION>::min();
  for(uint32 i=0; i<query_node->num_of_points_; i++) {
  	// get the current maximum distance
  	PRECISION distance = 
  	    TraitsNearestNeighbor<PRECISION, IDPRECISION, ALLOCATOR>::
				GetTheCurrentMaximum(*(query_node->get_neighbors()->at(i)), range);
    // we might as well replace this with a cross boundary condition
    comp.UpdateComparisons();
    if (this->box_.CrossesBoundaries(query_node->points_[i], 
                                               dimension,
                                               distance, 
                                               comp)) {
                                               	
      
       	                                        
//    if (distance >=  node_distance) { 	                                        
  	  FindNearest(query_node->points_[i], *(query_node->neighbors_->at(i)), 
                  distance, range, 
									discriminator,
									dimension, comp);
    }  
    		
    // Estimate the  maximum nearest neighbor distance
    comp.UpdateComparisons();
    if  (max_local_distance < distance) {
      max_local_distance = distance;
    }           
                    
  }
  if (max_neighbor_distance > max_local_distance) {
  	max_neighbor_distance = max_local_distance;
  }

}

__TEMPLATE__
inline void __NODE__::FindAllNearest(
		                typename ALLOCATOR::template Ptr<NODETYPE> query_node,
                    PRECISION &max_neighbor_distance,
                    PRECISION node_distance,
                    int32 range,
                    int32 dimension,
                    ComputationsCounter<diagnostic> &comp) {
  
	PointIdentityDiscriminator<IDPRECISION> discriminator;
  FindAllNearest(query_node,
                 max_neighbor_distance,
                 node_distance,
                 range,
                 dimension,
                 discriminator,
                 comp); 

} 

__TEMPLATE__
inline void __NODE__::FindAllNearest(
		                typename ALLOCATOR::template Ptr<NODETYPE> query_node,
                    PRECISION &max_neighbor_distance,
                    PRECISION node_distance,
                    int32 range,
                    int32 dimension,
                    PointIdentityDiscriminator<IDPRECISION> &discriminator,
                    ComputationsCounter<diagnostic> &comp) {
  
  PRECISION max_local_distance = numeric_limits<PRECISION>::min();
  for(uint32 i=0; i<query_node->num_of_points_; i++) {
  	// get the current maximum distance for the specific point
  	PRECISION distance = query_node->kneighbors_[i*range+range-1].distance_;
    // we might as well replace this with a cross boundary condition
    comp.UpdateComparisons();
    if (this->box_.CrossesBoundaries(query_node->points_[i], 
                                               dimension,
                                               distance, 
                                               comp)) {
                                               	
      vector<pair<PRECISION, 
			            Point<PRECISION, 
									      IDPRECISION, 
												ALLOCATOR> > > temp(range);
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
			 query_node-> kneighbors_[i*range+j].distance_=temp[j].first;
			 query_node->kneighbors_[i*range+j].nearest_=temp[j].second;
			}
                                       
    }  
    		
    // Estimate the  maximum nearest neighbor distance
    comp.UpdateComparisons();
    if  (max_local_distance < distance) {
      max_local_distance = distance;
    }           
                    
  }
  if (max_neighbor_distance > max_local_distance) {
  	max_neighbor_distance = max_local_distance;
  }

}

__TEMPLATE__
string __NODE__::Print(int32 dimension) {
  char buf[8192];
  string str;
  if (!IsLeaf()) {
    sprintf(buf, "Node: %llu\n", (unsigned long long)node_id_);
    str.append(buf);
  } else {
  	sprintf(buf, "Leaf: %llu\n", (unsigned long long)node_id_);
  	str.append(buf);
  }
  str.append(box_.Print(dimension));  
	str.append("num_of_points: ");
  sprintf(buf,"%llu\n", (unsigned long long)num_of_points_);
  str.append(buf);
	if (IsLeaf()) {
  	for(IDPRECISION i=0; i<num_of_points_; i++) {
  	  for(int32 j=0; j<dimension; j++) {
  	  	sprintf(buf,"%lg ", points_[i][j]);
  	  	str.append(buf);
  	  }
  	  sprintf(buf, "- %llu\n", (unsigned long long)points_[i].get_id());
  	  str.append(buf); 
  	}
  }
  return str;
}    	 

__TEMPLATE__
void __NODE__::PrintNeighbors(FILE *fp) {
  for(IDPRECISION i=0; i<num_of_points_; i++) {
  	for(IDPRECISION j=0; j<neighbors_->at(i)->size(); j++) {
  	  fprintf(fp, "%llu %llu %lg\n", (unsigned long long)points_[i].get_id(), 
  	                                 (unsigned long long)neighbors_->at(i)->
																		                     at(j).second.get_id(),
  	                                 (double)neighbors_->at(i)->at(j).first);
  	}
  }
}

__TEMPLATE__
void __NODE__::DeleteNeighbors() {
  for(IDPRECISION i=0; i<num_of_points_; i++) {
  	delete neighbors_->at(i);
  }
  delete neighbors_;
  neighbors_ = NULL;
}



#undef __TEMPLATE__
#undef __NODE__     	               	
#endif /*NODE_IMPL_H_*/
