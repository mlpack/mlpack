#ifndef BINARY_TREE_IMPL_H_
#define BINARY_TREE_IMPL_H_

#define TEMPLATE__           \
template<typename TYPELIST, bool diagnostic>
#define TREE__ BinaryTree<TYPELIST, diagnostic>

// Straight forward implemantation of the algorithms as described in Andrew Moore's
// paper. For more information regarding traits of nearest neighbors look at 
// traits_nearest_neighbor.h file

TEMPLATE__            
void TREE__::Init(BinaryDataset<Precision_t> *data) {
  data_ = data;
  dimension_  = data->get_dimension();
  num_of_points_ = data->get_num_of_points();
	node_id_=0;
	num_of_leafs_=0;
	current_level_=0;
	max_depth_ = 0;
	min_depth_ = numeric_limits<index_t>::max();
	max_points_on_leaf_ =  30;
	log_progress_=true;
}

TEMPLATE__
TREE__::~BinaryTree(){
}

TEMPLATE__
void TREE__::BuildBreadthFirst() {
  
	progress_.Reset();
  total_points_visited_ = 0;
  list<pair<NodePtrPtr_t, PivotInfo_t *> > fifo;
  PivotInfo_t *pivot = pivoter_(num_of_points_);
  parent_.Reset(new Node_t());
	parent_->Init(pivot->box_, 
			          pivot->statistics_, 
								node_id_,
								pivot->num_of_points_);
	node_id_++;
  pair<PivotInfo_t*, PivotInfo_t*> pivot_pair;
  pivot_pair = pivoter_(pivot);
  delete pivot;     
  
	fifo.push_front(make_pair(parent_->get_left().Reference(), pivot_pair.first));
  fifo.push_front(make_pair(parent_->get_right().Reference(), pivot_pair.second));
  current_level_ =1;                                  
  BuildBreadthFirst(fifo);
  if (log_progress_==true) {
  	printf("\n");
  }
}

TEMPLATE__
void TREE__::BuildBreadthFirst(
		list<pair<typename TREE__::NodePtrPtr_t, 
		          typename TREE__::PivotInfo_t *> > &fifo) {
  
  pair<PivotInfo_t*, PivotInfo_t*> pivot_pair; 
  while (!fifo.empty()) {
  	pair<NodePtrPtr_t, PivotInfo_t*> fifo_pair;
  	fifo_pair = fifo.back();
  	fifo.pop_back();
  	if (fifo_pair.second->num_of_points_ > max_points_on_leaf_) {
  	  (*fifo_pair.first).Reset(new Node_t());
			(*fifo_pair.first)->Init(fifo_pair.second->box_, 
				                       fifo_pair.second->statistics_,
							                 node_id_,
							                 fifo_pair.second->num_of_points_);

  	  node_id_++;
  	  pivot_pair = pivoter_(fifo_pair.second);
  	  delete fifo_pair.second;     
  	  fifo.push_front(make_pair((*fifo_pair.first)->get_left().Reference(), 
						                    pivot_pair.first));
  	  fifo.push_front(make_pair((*fifo_pair.first)->get_right().Reference(),
					                    	pivot_pair.second));
    } else {
      if (log_progress_==true) {
        total_points_visited_ += fifo_pair.second->num_of_points_;
        progress_.Show(total_points_visited_, get_num_of_points());
      }
     (*fifo_pair.first).Reset(new Node_t());
		 (*fifo_pair.first)->Init(fifo_pair.second->box_,
			                        fifo_pair.second->statistics_,
			                        node_id_,
			                        fifo_pair.second->start_,
			                        fifo_pair.second->num_of_points_,
                              dimension_,
                              data_); 

		  num_of_leafs_++;
			node_id_++;
    }
  }  
}      	
                                   	
TEMPLATE__ 
void TREE__::BuildDepthFirst() {
  total_points_visited_ = 0;
	min_depth_=numeric_limits<index_t>::max();
	max_depth_=0;
	current_level_=0;
	progress_.Reset();
  BuildDepthFirst(parent_, pivoter_(num_of_points_));
  if (log_progress_==true) {
  	printf("\n");
  }
}

TEMPLATE__
void TREE__::BuildDepthFirst(typename TREE__::NodePtr_t &ptr, 
    typename TREE__::PivotInfo_t *pivot_info) {
                                 	
  pair<PivotInfo_t *, PivotInfo_t *>  pivot_pair;
  if (pivot_info->num_of_points_ > max_points_on_leaf_) {
  	ptr.Reset(new Node_t());
		ptr->Init(pivot_info->box_, 
				      pivot_info->statistics_,
							node_id_,
							pivot_info->num_of_points_);
		node_id_++;
    pivot_pair = pivoter_(pivot_info);  
	  // There is a case where on all the points are the same
		// so pivoting returns 0 points on the left side
		// In that case we create a gigantic leaf
		if (pivot_pair.first->num_of_points_==0) {
      if (log_progress_==true) {
        total_points_visited_ +=pivot_pair.second->num_of_points_;
        progress_.Show(total_points_visited_, get_num_of_points());
      }
      if (current_level_ > max_depth_) {
		    max_depth_=current_level_;
		  }
		  if (current_level_ < min_depth_) {
		    min_depth_=current_level_;
		  }
      ptr.Reset(new Node_t());
			ptr->Init(pivot_pair.second->box_,
			          pivot_pair.second->statistics_,
			          node_id_,
			          pivot_pair.second->start_,
			          pivot_pair.second->num_of_points_,
			          dimension_,
                data_); 

		  node_id_++;
		  num_of_leafs_++;
      delete pivot_info;    
		  delete pivot_pair.first;
		  delete pivot_pair.second;	
		  return;
		}
  	delete pivot_info;
  	current_level_++; 
  	BuildDepthFirst(ptr->get_left(), pivot_pair.first);
  	BuildDepthFirst(ptr->get_right(), pivot_pair.second);
  	current_level_--; 
  } else {
  	if (log_progress_==true) {
      total_points_visited_ += pivot_info->num_of_points_;
      progress_.Show(total_points_visited_, get_num_of_points());
    }
    if (current_level_ > max_depth_) {
		  max_depth_=current_level_;
		}
		if (current_level_ < min_depth_) {
		  min_depth_=current_level_;
		}
    ptr.Reset(new Node_t());
    ptr->Init(pivot_info->box_,
			        pivot_info->statistics_,
			        node_id_,
			        pivot_info->start_,
			        pivot_info->num_of_points_,
			        dimension_,
              data_); 

		node_id_++;
		num_of_leafs_++;
    delete pivot_info;
  }
}

// This function will return any of the nearest neighbors
// k nearest, range nearest or just nearest
TEMPLATE__
template<typename POINTTYPE, typename NEIGHBORTYPE>
void TREE__::NearestNeighbor(POINTTYPE test_point,
    vector<pair<typename TREE__::Precision_t, 
		            typename TREE__::Point_t> > *nearest_point,
    NEIGHBORTYPE range) {
  bool found = false;
  NearestNeighbor(parent_, 
			            test_point, 
									nearest_point, 
									range, 
									found);
}           

TEMPLATE__
template<typename POINTTYPE, typename NEIGHBORTYPE>
void TREE__::NearestNeighbor(NodePtr_t ptr,
    POINTTYPE &test_point,
    vector<pair<typename TREE__::Precision_t,
	              typename TREE__::Point_t> > *nearest_point,
    NEIGHBORTYPE range,
    bool &found) {
  computations_.UpdateComparisons();
  if (!ptr->IsLeaf()){
  	computations_.UpdateComparisons();
		pair<NodePtr_t, NodePtr_t> child_pair = 
			ptr->ClosestChild(test_point, dimension_, computations_);

		NearestNeighbor(child_pair.first, test_point, nearest_point, 
                     range, found);
    if (child_pair.second->get_box().CrossesBoundaries(test_point, 
					                                       dimension_, 
																								 nearest_point->end()->first,
                                                 computations_)) {
       NearestNeighbor(child_pair.second, 
			                 test_point, 
											 nearest_point, 
                       range, found);
    }
 
      if (found == true) {
      	return;
      } else {
        found = ptr->get_box().IsWithin(test_point, 
						                       dimension_, nearest_point->end()->first,  
                                   computations_)==0;
        if (found == true) {
          return;
        }
      }
  } else {
  	ptr->FindNearest(test_point, *nearest_point, 
				             range, dimension_,
										 discriminator_,
  	                 computations_);
  	found = ptr->get_box().IsWithin(test_point, dimension_, 
			                              nearest_point->end()->first,	
  	                                computations_);
  }  	
}    

TEMPLATE__                  	                                
template<typename NEIGHBORTYPE>
void TREE__::AllNearestNeighbors(typename TREE__::NodePtr_t query, 
                                 NEIGHBORTYPE range) {
   ResetCounters();
   Precision_t distance = numeric_limits<Precision_t>::max();
	 progress_.Reset();
   AllNearestNeighbors(query, parent_, range, distance);                          	                           	                              	    	 	                                 	
	 total_points_visited_=0;
}

TEMPLATE__ 
template<typename NEIGHBORTYPE >
void TREE__::AllNearestNeighbors(typename TREE__::NodePtr_t query, 
                                 typename TREE__::NodePtr_t reference,
                                 NEIGHBORTYPE range, 
                                 typename TREE__::Precision_t distance) {                                               	

  if (distance > query->get_min_dist_so_far()) {
  	return ;
  } else {
  	if (query->IsLeaf() && reference->IsLeaf()) {
  	  Precision_t max_distance=numeric_limits<Precision_t>::max();
			reference->FindAllNearest(query, 
  	                            max_distance,
  	                            range,
  	                            dimension_, 
															  discriminator_,	
  	                            computations_);	                           
		  query->set_min_dist_so_far(max_distance);	
  	} else {
  	  if (query->IsLeaf() && !reference->IsLeaf()) {
  	  	pair<pair<NodePtr_t, Precision_t>,
				     pair<NodePtr_t, Precision_t>  >  closest_child;
  	  	closest_child = query->ClosestNode(reference->get_left(), 
						                               reference->get_right(),
  	  	                                   dimension_,
  	  	                                   computations_);
  	  	AllNearestNeighbors(query, 
						                closest_child.first.first, // child
						                range,                            // range  
														closest_child.first.second // distance of query
												    // from the reference child		
														);
        AllNearestNeighbors(query, closest_child.second.first,
					                         range, closest_child.second.second);
        
  	  } else {
        if (!query->IsLeaf() && reference->IsLeaf()) {
					pair<pair<NodePtr_t, Precision_t>,
				       pair<NodePtr_t, Precision_t>  >  closest_child;
  	  	  closest_child = reference->ClosestNode(query->get_left(), 
						                                     query->get_right(),
  	  	                                         dimension_,
  	  	                                         computations_);

					AllNearestNeighbors(closest_child.first.first, 
							                reference, 
															range, 
															closest_child.first.second);
          AllNearestNeighbors(closest_child.second.first,
						                	reference, 
															range, 
															closest_child.second.second); 
          query->set_min_dist_so_far(
			        std::min<Precision_t>(query->get_min_dist_so_far(),
								 std::max<Precision_t>(query->get_left()->get_min_dist_so_far(),
									    query->get_right()->get_min_dist_so_far())));
        } else {
          if (!query->IsLeaf() && !reference->IsLeaf()) {
          	pair<pair<NodePtr_t, Precision_t>,
				    pair<NodePtr_t, Precision_t>  >  closest_child;
  	  	    closest_child = query->get_left()->ClosestNode(
								reference->get_left(), 
						    reference->get_right(),
  	  	        dimension_,
  	  	        computations_);
          	
						AllNearestNeighbors(query->get_left(), 
								                closest_child.first.first, 
																range, 
								                closest_child.first.second);
            AllNearestNeighbors(query->get_left(), 
								                closest_child.second.first, 
																range, 
								                closest_child.second.second);
            closest_child = query->get_right()->ClosestNode(
								reference->get_left(), 
						    reference->get_right(),
  	  	        dimension_,
  	  	        computations_);
            AllNearestNeighbors(query->get_right(), 
								                closest_child.first.first, 
								                range,  
																closest_child.first.second);
            AllNearestNeighbors(query->get_right(), 
								                closest_child.second.first, 
								                range, 
																closest_child.second.second);   
            query->set_min_dist_so_far(
			        std::min<Precision_t>(query->get_min_dist_so_far(),
								std::max<Precision_t>(query->get_left()->get_min_dist_so_far(),
									    query->get_right()->get_min_dist_so_far())));
 
          }
        }
  	  }
  	}
  }
  
}     	                                                                              	  

TEMPLATE__
void TREE__::InitAllKNearestNeighborOutput(string file, 
		                                        int32 knns) {
  FILE *fp=fopen(file.c_str(), "w");
	const int32 kChunk=8192;
	typename Node_t::NNResult *buffer;
	buffer=new typename Node_t::NNResult[kChunk*knns];
	for(index_t i=0; i<num_of_points_/kChunk; i++) {
	  fwrite(buffer, sizeof(typename Node_t::NNResult),kChunk*knns, fp );
	}
  fwrite(buffer, sizeof(typename Node_t::NNResult),
			   (num_of_points_%kChunk)*knns, fp );
	fclose(fp);
	delete buffer;
	int fd=open(file.c_str(), O_RDWR);
	typename Node_t::NNResult *ptr =(typename Node_t::NNResult *)mmap(NULL,
      sizeof(typename Node_t::NNResult)*knns*num_of_points_,
			PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (ptr==MAP_FAILED) {
	  fprintf(stderr, "Unable to map file: %s", strerror(errno));
		assert(false);
	}
	close(fd);
  all_nn_out_.set_ptr(ptr);
	InitAllKNearestNeighborOutput(parent_, knns);
}


TEMPLATE__
void TREE__::InitAllKNearestNeighborOutput(typename TREE__::NodePtr_t ptr, 
		                                        int32 knns) {
  if (ptr->IsLeaf()) {
  	ptr->set_kneighbors(all_nn_out_.Allocate(ptr->get_num_of_points(), knns),
			                                       knns);
	  ptr->InitKNeighbors(knns);
	} else {
	  InitAllKNearestNeighborOutput(ptr->get_left(), knns);
		InitAllKNearestNeighborOutput(ptr->get_right(), knns);
	}

}

TEMPLATE__
void TREE__::InitAllRangeNearestNeighborOutput(string file) {
	FILE *fp=fopen(file.c_str(), "w");
	if (fp==NULL) {
	  FATAL("Cannot open %s, error: %s\n", 
			    file.c_str(), 
			    strerror(errno));
	}
	parent_->set_range_neighbors(fp);
  InitAllRangeNearestNeighborOutput(parent_, fp);
}


TEMPLATE__
void TREE__::InitAllRangeNearestNeighborOutput(
		typename TREE__::NodePtr_t ptr,
		FILE *fp) {
  if (ptr->IsLeaf()) {
	  ptr->set_range_neighbors(fp);
	} else {
	  InitAllRangeNearestNeighborOutput(ptr->get_left(), fp);
    InitAllRangeNearestNeighborOutput(ptr->get_right(), fp);
	}
}

TEMPLATE__
void TREE__::CloseAllKNearestNeighborOutput(int32 knns) {
  if (munmap(all_nn_out_.get_ptr(), 
			   sizeof(typename Node_t::NNResult)*knns*num_of_points_)<0) {
	  fprintf(stderr, "Failed to umap file: %s", strerror(errno));
		assert(false);
	}
}

TEMPLATE__
void TREE__::CloseAllRangeNearestNeighborOutput() {
  fclose(parent_->get_range_nn_fp());
}

TEMPLATE__
void TREE__::Print() {
  RecursivePrint(parent_);
}


TEMPLATE__
void TREE__::RecursivePrint(typename TREE__::NodePtr_t ptr) {
  string str;
  if (ptr->IsLeaf()) {
 	str = ptr->Print(dimension_);
 	printf("%s\n", str.c_str());
  } else {
 	str = ptr->Print(dimension_);
 	printf("%s\n", str.c_str());
 	RecursivePrint(ptr->get_left());
 	RecursivePrint(ptr->get_right());
  }
}


TEMPLATE__
string TREE__::Statistics() {
  char buff[4096];
  sprintf(buff, "Number of points     : %llu,\n"
                "Number of dimensions : %i\n"
                "Number of nodes      : %llu,\n"
                "Number of leafs      : %llu,\n"
                "Max tree depth       : %llu,\n"
                "Min tree depth       : %llu,\n",
                (unsigned long long)num_of_points_,
                dimension_,
                (unsigned long long)node_id_,
                (unsigned long long)num_of_leafs_,
                (unsigned long long)max_depth_,
                (unsigned long long)min_depth_);
   return string(buff);
}

TEMPLATE__
string TREE__::Computations() {
  char buff[4096];
	sprintf(buff,"number of comparisons:  %llu\n"
			         "number of distances:    %llu\n",
							 (unsigned long long)computations_.get_comparisons(),
							 (unsigned long long)computations_.get_distances());
  return string(buff);
}

TEMPLATE__
void TREE__::set_log_file(const string &log_file) {
  if (log_file_ptr_ != stderr ) {
  	if (fclose(log_file_ptr_)!= 0) {
  	  fprintf(stderr, "Cannot close file %s\n", log_file_.c_str());
  	  assert(false);
  	}
  }
  log_file_ptr_ = fopen(log_file.c_str(), "wb");
  log_file_ = log_file;
 
}

     	           

#undef TREE__
#undef TEMPLATE__
#endif /*BINARY_TREE_IMPL_H_*/
