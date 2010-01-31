#ifndef TREE_IMPL_H_
#define TREE_IMPL_H_

#define __TEMPLATE__           \
template<typename PRECISION,	 \
				 typename IDPRECISION, \
				 typename ALLOCATOR,   \
				 bool     diagnostic,  \
  template<typename PREC,      \
           typename IDPREC,    \
           typename ALLOCATOR, \
           bool     diagnostic> class NODE>                             
#define __TREE__ Tree<PRECISION, IDPRECISION, ALLOCATOR, diagnostic, NODE>

// Straight forward implemantation of the algorithms as described in Andrew Moore's
// paper. For more information regarding traits of nearest neighbors look at 
// traits_nearest_neighbor.h file

__TEMPLATE__            
__TREE__::Tree(DataReader<PRECISION, IDPRECISION> *data, 
               int32 dimension, IDPRECISION num_of_points) {
    	
  data_ = data;
  dimension_  = dimension;
  num_of_points_ = num_of_points;
	node_id_=0;
	num_of_leafs_=0;
	current_level_=0;
	max_depth_ = 0;
	min_depth_ = numeric_limits<IDPRECISION>::max();
	max_points_on_leaf_ =  30;
	log_progress_=true;
	discriminator_.reset(new PointIdentityDiscriminator<IDPRECISION>());
}

__TEMPLATE__
__TREE__::~Tree(){
}

__TEMPLATE__
void __TREE__::SerialBuildBreadthFirst() {
  
	progress_.Reset();
  total_points_visited_ = 0;
  list<pair<Node_ptr_ptr, Pivot_t *> > fifo;
  Pivot_t *pivot = Policy_t::PivotParent(data_, num_of_points_, dimension_);
  parent_.Reset(new Node_t(pivot, node_id_));
	node_id_++;
  pair<Pivot_t*, Pivot_t*> pivot_pair;
  pivot_pair = Policy_t::Pivot(data_, pivot);
  delete pivot;     
  
	fifo.push_front(make_pair(parent_->get_left().Reference(), pivot_pair.first));
  fifo.push_front(make_pair(parent_->get_right().Reference(), pivot_pair.second));
  current_level_ =1;                                  
  SerialBuildBreadthFirst(fifo);
  if (log_progress_==true) {
  	printf("\n");
  }
}

__TEMPLATE__
void __TREE__::SerialBuildBreadthFirst(list<pair<Node_ptr_ptr, Pivot_t *> > &fifo) {
  
  pair<Pivot_t*, Pivot_t*> pivot_pair; 
  while (!fifo.empty()) {
  	pair<Node_ptr_ptr, Pivot_t*> fifo_pair;
  	fifo_pair = fifo.back();
  	fifo.pop_back();
  	if (fifo_pair.second->num_of_points_ > max_points_on_leaf_) {
  	  (*fifo_pair.first).Reset(new Node_t(fifo_pair.second, node_id_));
  	  node_id_++;
  	  pivot_pair = Policy_t::Pivot(data_, fifo_pair.second);
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
     (*fifo_pair.first).Reset(new Node_t(fifo_pair.second, node_id_, data_));
			num_of_leafs_++;
			node_id_++;
    }
  }  
}      	
                                   	
__TEMPLATE__ 
void __TREE__::SerialBuildDepthFirst() {
  total_points_visited_ = 0;
	min_depth_=numeric_limits<IDPRECISION>::max();
	max_depth_=0;
	current_level_=0;
	progress_.Reset();
  SerialBuildDepthFirst(parent_, Policy_t::PivotParent(data_, 
				                                               num_of_points_,
																											 dimension_));
  if (log_progress_==true) {
  	printf("\n");
  }
}

__TEMPLATE__
void __TREE__::SerialBuildDepthFirst(Node_ptr &ptr, 
                               Pivot_t  *pivot_info) {
                                 	
  pair<Pivot_t *, Pivot_t *>  pivot_pair;
  if (pivot_info->num_of_points_ > max_points_on_leaf_) {
  	ptr.Reset(new Node_t(pivot_info, node_id_));
		node_id_++;
    pivot_pair = Policy_t::Pivot(data_, pivot_info);  
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
      ptr.Reset(new Node_t(pivot_pair.second, node_id_, data_));
		  node_id_++;
		  num_of_leafs_++;
      delete pivot_info;    
		  delete pivot_pair.first;
		  delete pivot_pair.second;	
		  return;
		}
  	delete pivot_info;
  	current_level_++; 
  	SerialBuildDepthFirst(ptr->get_left(), pivot_pair.first);
  	SerialBuildDepthFirst(ptr->get_right(), pivot_pair.second);
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
    ptr.Reset(new Node_t(pivot_info, node_id_, data_));
		node_id_++;
		num_of_leafs_++;
    delete pivot_info;
  }
}

// This function will return any of the nearest neighbors
// k nearest, range nearest or just nearest
__TEMPLATE__
template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
void __TREE__::NearestNeighbor(POINTTYPE &test_point,
                               RETURNTYPE *nearest_point,
                               PRECISION    *distance,
                               NEIGHBORTYPE range) {
  bool found = false;
  *distance = numeric_limits<PRECISION>::max();
  TraitsNearestNeighbor<PRECISION, IDPRECISION, ALLOCATOR>::Prepare(*nearest_point, range);                            	
  NearestNeighbor(parent_, test_point, nearest_point, distance, range, found);
}           

__TEMPLATE__
template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
void __TREE__::NearestNeighbor(Node_ptr ptr,
                               POINTTYPE &test_point,
                               RETURNTYPE *nearest_point,
                               PRECISION   *distance,
                               NEIGHBORTYPE range,
                               bool &found) {
  computations_.UpdateComparisons();
  if (!ptr->IsLeaf()){
  	computations_.UpdateComparisons();
		pair<Node_ptr, Node_ptr> child_pair = 
			ptr->ClosestChild(test_point, dimension_, computations_);

		NearestNeighbor(child_pair.first, test_point, nearest_point, distance,
                     range, found);
    if (child_pair.second->get_box().CrossesBoundaries(test_point, 
					                                        dimension_, 
																									*distance,
                                                  computations_)) {
       NearestNeighbor(child_pair.second, 
			                 test_point, 
											 nearest_point, 
											 distance, 
                       range, found);
    }
 
      if (found == true) {
      	return;
      } else {
        found = ptr->get_box().IsWithin(test_point, 
						                       dimension_, *distance,  
                                   computations_)==0;
        if (found == true) {
          return;
        }
      }
  } else {
  	ptr->FindNearest(test_point, *nearest_point, 
				             *distance, range, dimension_,
										 *discriminator_,
  	                 computations_);
  	found = ptr->get_box().IsWithin(test_point, dimension_, *distance, 
  	                           computations_);
  }  	
}    

__TEMPLATE__                  	                                
template<typename NEIGHBORTYPE>
void __TREE__::AllNearestNeighbors(Node_ptr query, 
                                   Node_ptr reference,
                                   NEIGHBORTYPE range) {
   ResetCounters();
   PRECISION distance = numeric_limits<PRECISION>::max();
	 progress_.Reset();
   AllNearestNeighbors(query, reference, range, distance);                          	                           	                              	    	 	                                 	
	 total_points_visited_=0;
}

__TEMPLATE__ 
template<typename NEIGHBORTYPE >
void __TREE__::AllNearestNeighbors(Node_ptr query, 
                                   Node_ptr reference,
                                   NEIGHBORTYPE range, 
                                   PRECISION distance) {                                               	

  if (distance > query->get_min_dist_so_far()) {
  	return ;
  } else {
  	if (query->IsLeaf() && reference->IsLeaf()) {
  	  PRECISION max_distance=numeric_limits<PRECISION>::max();
			reference->FindAllNearest(query, 
  	                            max_distance,
  	                            0, // this has to be fixed in the future 
																   // this value is redundant
  	                            range,
  	                            dimension_, 
															  *discriminator_,	
  	                            computations_);	                           
		  query->set_min_dist_so_far(max_distance);	
  	} else {
  	  if (query->IsLeaf() && !reference->IsLeaf()) {
  	  	pair<pair<Node_ptr, PRECISION>,
				     pair<Node_ptr, PRECISION>  >  closest_child;
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
					pair<pair<Node_ptr, PRECISION>,
				       pair<Node_ptr, PRECISION>  >  closest_child;
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
			        min(query->get_min_dist_so_far(),
								  max(query->get_left()->get_min_dist_so_far(),
									    query->get_right()->get_min_dist_so_far())));
        } else {
          if (!query->IsLeaf() && !reference->IsLeaf()) {
          	pair<pair<Node_ptr, PRECISION>,
				    pair<Node_ptr, PRECISION>  >  closest_child;
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
			        min(query->get_min_dist_so_far(),
								max(query->get_left()->get_min_dist_so_far(),
									    query->get_right()->get_min_dist_so_far())));
 
          }
        }
  	  }
  	}
  }
  
}     	                                                                              	  

__TEMPLATE__
void __TREE__::Print() {
  RecursivePrint(parent_);
}

__TEMPLATE__
void __TREE__::InitAllKNearestNeighborOutput(string file, 
		                                        int32 range
		                                       ) {
  FILE *fp=fopen(file.c_str(), "w");
	const int32 kChunk=8192;
	boost::scoped_array<typename Node_t::Result> buffer;
	buffer.reset(new typename Node_t::Result[kChunk*range]);
	for(IDPRECISION i=0; i<num_of_points_/kChunk; i++) {
	  fwrite(buffer.get(), sizeof(typename Node_t::Result),kChunk*range, fp );
	}
  fwrite(buffer.get(), sizeof(typename Node_t::Result),
			   (num_of_points_%kChunk)*range, fp );
	fclose(fp);
	
	int fd=open(file.c_str(), O_RDWR);
	typename Node_t::Result *ptr =(typename Node_t::Result *)mmap(NULL,
      sizeof(typename Node_t::Result)*range*num_of_points_,
			PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (ptr==MAP_FAILED) {
	  fprintf(stderr, "Unable to map file: %s", strerror(errno));
		assert(false);
	}
	close(fd);
  all_nn_out_.set_ptr(ptr);
	InitAllKNearestNeighborOutput(parent_, range);
}


__TEMPLATE__
void __TREE__::InitAllKNearestNeighborOutput(Node_ptr ptr, 
		                                        int32 range                                          																						
	                                          ) {
  if (ptr->IsLeaf()) {
  	ptr->set_kneighbors(all_nn_out_.Allocate(ptr->get_num_of_points(),
			                                       range));
	  ptr->InitKNeighbors(range);
	} else {
	  InitAllKNearestNeighborOutput(ptr->get_left(), range);
		InitAllKNearestNeighborOutput(ptr->get_right(), range);
	}

}

__TEMPLATE__
void __TREE__::CloseAllKNearestNeighborOutput(int32 range) {
  if (munmap(all_nn_out_.get_ptr(), 
			   sizeof(typename Node_t::Result)*range*num_of_points_)<0) {
	  fprintf(stderr, "Failed to umap file: %s", strerror(errno));
		assert(false);
	}
}

__TEMPLATE__
void __TREE__::RecursivePrint(Node_ptr ptr) {
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

__TEMPLATE__
void __TREE__::PrintNeighbors(string filename) {
  FILE *fp=fopen(filename.c_str(), "w");
  PrintNeighborsRecursive(parent_, fp);
	fclose(fp);
}
__TEMPLATE__
void __TREE__::PrintNeighborsRecursive(Node_ptr ptr, FILE *fp) {
  if (!ptr->IsLeaf()) {
  	PrintNeighborsRecursive(ptr->get_left(), fp);
  	PrintNeighborsRecursive(ptr->get_right(), fp);
  } else {
  	if (ptr->get_neighbors() == NULL) {
  	  return;
  	}
  	ptr->PrintNeighbors(fp);
  	ptr->DeleteNeighbors();
  }
}
 

__TEMPLATE__
string __TREE__::Statistics() {
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

__TEMPLATE__
string __TREE__::Computations() {
  char buff[4096];
	sprintf(buff,"number of comparisons:  %llu\n"
			         "number of distances:    %llu\n",
							 (unsigned long long)computations_.get_comparisons(),
							 (unsigned long long)computations_.get_distances());
  return string(buff);
}

__TEMPLATE__
void __TREE__::set_log_file(const string &log_file) {
  if (log_file_ptr_ != stderr ) {
  	if (fclose(log_file_ptr_)!= 0) {
  	  fprintf(stderr, "Cannot close file %s\n", log_file_.c_str());
  	  assert(false);
  	}
  }
  log_file_ptr_ = fopen(log_file.c_str(), "wb");
  log_file_ = log_file;
 
}

     	           

#undef __TREE__
#undef __TEMPLATE__
#endif /*TREE_IMPL_H_*/
