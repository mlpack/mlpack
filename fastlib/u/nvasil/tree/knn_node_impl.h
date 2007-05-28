#ifndef KNN_NODE_IMPL_H_
#define KNN_NODE_IMPL_H_

#define TEMPLATE__                                               \
template<typename TYPELIST, bool diagnostic>                                                                  

#define KNN_NODE__                                                   \
	KnnNode<TYPELIST, diagnostic>                  
 
TEMPLATE__    
KNN_NODE__::KnnNode() {
  left_.SetNULL();
  right_.SetNULL();
  points_.SetNULL();
	node_id_ = numeric_limits<index_t>::max();
	min_dist_so_far_=numeric_limits<Precision_t>::max();
}

TEMPLATE__
void KNN_NODE__::Init(const BoundingBox_t &box, 
	                const NodeCachedStatistics_t &statistics,		
			            index_t node_id,
			            index_t num_of_points) {
  box_.Alias(box);
  statistics_.Alias(statistics);
  node_id_ = node_id;
	num_of_points_ = num_of_points;
}

TEMPLATE__
void KNN_NODE__::Init(const typename KNN_NODE__::BoundingBox_t &box,
		      	      const typename KNN_NODE__::NodeCachedStatistics_t &statistics,
			            index_t node_id,
                  index_t start,
                  index_t num_of_points,
							    int32 dimension,
			            BinaryDataset<Precision_t> *dataset) {
	box_.Alias(box);
	statistics_.Alias(statistics);
	node_id_ = node_id;
	num_of_points_ = num_of_points;
	points_.Reset(Allocator_t::template malloc<Precision_t>
			             (num_of_points_*dimension));
  index_.Reset(Allocator_t::template malloc<index_t>(num_of_points_));
	points_.Lock();
	index_.Lock();
	for(index_t i=start; i<start+num_of_points_; i++) {
	  for(int32 j=0; j<dimension; j++) {
		  points_[(i-start)*dimension+j]=dataset->At(i,j);
	  }
		index_[i-start]=dataset->get_id(i);
	}
	points_.Unlock();
	index_.Unlock();
} 

TEMPLATE__
KNN_NODE__::~KnnNode() {
}

TEMPLATE__
void *KNN_NODE__::operator new(size_t size) {
  typename Allocator_t::template Ptr<Node_t> temp;
	temp.Reset(Allocator_t::malloc(size));
  return (void *)temp.get();
}
     	
TEMPLATE__
void KNN_NODE__::operator delete(void *p) {
}
                 	
TEMPLATE__
template<typename POINTTYPE>
pair<typename KNN_NODE__::NodePtr_t, typename KNN_NODE__::NodePtr_t>                     
KNN_NODE__::ClosestChild(POINTTYPE point, int32 dimension, 
		                 ComputationsCounter<diagnostic> &comp) {
  left_.Lock();
	right_.Lock();
	return box_.ClosestChild(left_, right_, point, dimension, comp);
	left_.Unlock();
	right_.Unlock();
}

TEMPLATE__
inline 
pair<pair<typename KNN_NODE__::NodePtr_t, typename KNN_NODE__::Precision_t>, 
		 pair<typename KNN_NODE__::NodePtr_t, typename KNN_NODE__::Precision_t> > 
KNN_NODE__::ClosestNode(typename KNN_NODE__::NodePtr_t ptr1,
		                    typename KNN_NODE__::NodePtr_t ptr2,
									      int32 dimension,
							          ComputationsCounter<diagnostic> &comp) {
	ptr1.Lock();
	ptr2.Lock();
	Precision_t dist1 = BoundingBox_t::Distance(box_, ptr1->get_box(), 
			                                      dimension, comp);
	Precision_t dist2 = BoundingBox_t::Distance(box_, ptr2->get_box(), 
			                                      dimension, comp);
  ptr1.Unlock();
	ptr2.Unlock();
	if (dist1<dist2) {
	  return make_pair(make_pair(ptr1, dist1), make_pair(ptr2, dist2));
	} else {
	  return make_pair(make_pair(ptr2,dist2), make_pair(ptr1, dist1));
	}
}

TEMPLATE__
template<typename POINTTYPE>
inline void KNN_NODE__::FindNearest(POINTTYPE query_point, 
   vector<pair<typename KNN_NODE__::Precision_t, 
	             typename KNN_NODE__::Point_t> > &nearest, 
		index_t knns, 
		int32 dimension,
		typename KNN_NODE__::PointIdDiscriminator_t &discriminator,
    ComputationsCounter<diagnostic> &comp) {
  
	for(index_t i=0; i<num_of_points_; i++) {
  	comp.UpdateDistances();
  	//  we have to check if we are comparing the point with itself
 	  if (unlikely(discriminator.AreTheSame(index_[i], 
		       	     query_point.get_id())==true)) {
  	 	continue;
 	  } 

		Precision_t dist = BoundingBox_t::
			template Distance(query_point, 
			                  points_.get_p()+i*dimension,
			  							  dimension);
	// for k nearest neighbors
		Point_t point;
		point.Alias(points_.get()+i*dimension, index_[i]);
		nearest.push_back(make_pair(dist, point));
	}
	
  // for k-nearest neighbors 
	typename  std::vector<pair<Precision_t, Point_t> >::iterator it;
	it=nearest.begin()+knns;
  std::sort(nearest.begin(), 
			      nearest.end(),
						PairComparator());
	if (likely(nearest.size()>(uint32)knns)) {
	  nearest.erase(it, nearest.end());
	} else {
	  pair<Precision_t, Point_t> dummy;
		dummy.first=numeric_limits<Precision_t>::max();
		index_t extra_size=(index_t)(knns-nearest.size());
		for(index_t i=0; i<extra_size; i++) {
		  nearest.push_back(dummy);
		}
	}	  	
}

TEMPLATE__
inline void KNN_NODE__::FindAllNearest(
		                NodePtr_t query_node,
                    typename KNN_NODE__::Precision_t &max_neighbor_distance,
                    index_t knns,
                    int32 dimension,
										typename KNN_NODE__::PointIdDiscriminator_t &discriminator,
                    ComputationsCounter<diagnostic> &comp) {
  
  points_.Lock();
  index_.Lock();	
	query_node->points_.Lock();
  query_node->index_.Lock();
	query_node->kneighbors_.Lock();
	query_node->distances_.Lock();
	Precision_t max_local_distance = 0; 
	for(index_t i=0; i<query_node->num_of_points_; i++) {
		Precision_t distance;
		// for k nearest neighbors
		// get the current maximum distance for the specific point
  	distance = query_node->distances_[i*knns+knns-1];
		// We should check whether this speeds up or slows down 
		// the performance 
    comp.UpdateComparisons();
   
    Precision_t *temp_point=query_node->points_.get_p()+i*dimension;
		if (this->box_.CrossesBoundaries(temp_point, 
                                     dimension,
                                     distance, 
                                     comp)) {
		  // for k nearest neighbors                       	
			vector<pair<Precision_t, Point_t> > temp(knns);
			for(int32 j=0; j<knns; j++) {
			  temp[j].first=query_node->distances_[i*knns+j];
			  temp[j].second=query_node->kneighbors_[i*knns+j];
			}
			NullPoint_t point;
			point.Alias(query_node->points_.get_p()+i*dimension, 
					        query_node->index_[i]);
      FindNearest(point, temp, 
                  knns, dimension,
				 				  discriminator,	comp);
			DEBUG_ASSERT_MSG((index_t)temp.size()==knns, 
			 		             "During  %i-nn seach, returned %u results",(int)knns, 
						            (unsigned int)temp.size());

				
			for(int32 j=0; j<knns; j++) {
			  query_node->kneighbors_[i*knns+j]=temp[j].second;
			  query_node->distances_[i*knns+j]=temp[j].first;
			}
      // Estimate the  maximum nearest neighbor distance
      comp.UpdateComparisons();
      if (max_local_distance < temp.back().first) {
        max_local_distance = temp.back().first;
		  }
		}		
	  if  (max_local_distance < distance) {
      max_local_distance = distance;
 		}
	} 
  if (max_neighbor_distance>max_local_distance) {
	  max_neighbor_distance=max_local_distance;
	}

  points_.Unlock();
	index_.Unlock();
  query_node->points_.Unlock();
  query_node->index_.Unlock();
	query_node->kneighbors_.Unlock();
	query_node->distances_.Unlock();
}
TEMPLATE__
void KNN_NODE__::OutputNeighbors(NNResult *out, index_t knns) {
    kneighbors_.Lock();
	  distances_.Lock();
		index_.Lock();
    for(index_t i=0; i<num_of_points_; i++) {
		  for(index_t j=0; j<knns; j++) {
			  out[i*knns+j].point_id_ =index_[i];  
			  out[i*knns+j].nearest_ = kneighbors_[i*knns+j];
				out[i*knns+j].distance_ = distances_[i*knns+j];
			}
		}
		index_.Unlock();
	  kneighbors_.Unlock();
	  distances_.Unlock();	
	}	

TEMPLATE__
void KNN_NODE__::OutputNeighbors(FILE *fp, index_t knns) {
    kneighbors_.Lock();
	  distances_.Lock();
		index_.Lock();
		NNResult result;
    for(index_t i=0; i<num_of_points_; i++) {
		  for(index_t j=0; j<knns; j++) {
			  result.point_id_ = index_[i];  
			  result.nearest_ = kneighbors_[i*knns+j];
				result.distance_ = distances_[i*knns+j];
				fwrite(&result, sizeof(NNResult),1 , fp);
			}
		}
		index_.Unlock();
	  kneighbors_.Unlock();
	  distances_.Unlock();	
	}	

TEMPLATE__
string KNN_NODE__::Print(int32 dimension) {
  points_.Lock();
	index_.Lock();
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
  	for(index_t i=0; i<num_of_points_; i++) {
  	  for(int32 j=0; j<dimension; j++) {
  	  	sprintf(buf,"%lg ", points_[i*dimension+j]);
  	  	str.append(buf);
  	  }
  	  sprintf(buf, "-%llu \n",(unsigned long long) index_[i]);
  	  str.append(buf); 
  	}
  }
	points_.Unlock();
	index_.Unlock();
  return str;
}    	 

#undef TEMPLATE__
#undef KNN_NODE__     	               	
#endif /*KNN_NODE_IMPL_H_*/
