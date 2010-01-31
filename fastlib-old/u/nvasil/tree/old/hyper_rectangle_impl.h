#ifndef HYPER_RECTANGLE_IMPL_H_
#define HYPER_RECTANGLE_IMPL_H_

#define __TEMPLATE__ template<typename PRECISION, \
                              typename ALLOCATOR, \
															bool diagnostic>
#define __HyperRectangle__ HyperRectangle<PRECISION, ALLOCATOR,    \
	                                        diagnostic>

__TEMPLATE__
__HyperRectangle__::HyperRectangle(PivotData &pivot_data) { 
	min_ = pivot_data.min_;
  max_ = pivot_data.max_;
  pivot_dimension_ = pivot_data.pivot_dimension_;
  pivot_value_=pivot_data.pivot_value_;
}


__TEMPLATE__
HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &__HyperRectangle__::operator=
    (HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr) {
  
  this->min_ = hr.min_;
  this->max_ = hr.max_;
  pivot_dimension_ = hr.pivot_dimension_;
  pivot_value_ = hr.pivot_value_;
  return *this;
}

__TEMPLATE__
void *__HyperRectangle__::operator new(size_t size) {
  return ALLOCATOR::allocator->AllignedAlloc(size);
}

__TEMPLATE__
void __HyperRectangle__::operator delete(void *p) {
}

__TEMPLATE__
template<typename POINTTYPE> 
inline bool __HyperRectangle__::IsWithin(
    POINTTYPE point, int32 dimension, PRECISION range, 
    ComputationsCounter<diagnostic> &comp) {
  // non overlaping at all 
  for(int32 i=0; i<dimension; i++) {
  	  comp.UpdateComparisons();
      if ( point[i] > max_[i] || point[i] < min_[i]) {
  	    return false;
  	  }
    }
  PRECISION closest_projection = max_[0]-min_[0]; 
  for(int32 i=0; i<dimension; i++) {
    PRECISION projection1 = max_[i] - point[i];
    PRECISION projection2 = point[i] - min_[i];
    comp.UpdateComparisons();
    if (closest_projection > projection1 ) {
      closest_projection = projection1;
    }
    comp.UpdateComparisons();
    if (closest_projection > projection2 ) {
      closest_projection = projection2;
    }
    comp.UpdateComparisons();
    if (range >= closest_projection * closest_projection) {
     // Overlapping
 			return  false ;
    }
  }
	// Completelly inside
  return true;
} 

    	
    
__TEMPLATE__
inline PRECISION __HyperRectangle__::IsWithin(
    HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr,
    int32 dimension, 
    PRECISION range,
    ComputationsCounter<diagnostic> &comp) {
  
  PRECISION closest_projection = numeric_limits<PRECISION>::max();
  for(int32 i=0; i<dimension; i++) {
    comp.UpdateComparisons();
    comp.UpdateComparisons();
    PRECISION d1=hr.min_[i] - min_[i];
    PRECISION d2=max_[i] - hr.max_[i];
    if (d1<0 || d2<0) {
      return -1;
    } else {
      PRECISION dist = min(d1,d2);
      if (dist < closest_projection) {
        closest_projection = dist;
      }
    }
  }
  if (closest_projection * closest_projection > range) {
    return 0;
  }
  return (sqrt(range) - closest_projection) * 
         (sqrt(range) - closest_projection); 
}    	

__TEMPLATE__
template<typename POINTTYPE> 
inline bool __HyperRectangle__::CrossesBoundaries(
    POINTTYPE point, int32 dimension, PRECISION range, 
    ComputationsCounter<diagnostic> &comp) {
  
  PRECISION closest_point_coordinate;
  PRECISION dist = 0;
  for(int32 i=0; i<dimension; i++) {
    comp.UpdateComparisons();
    if (point[i] <= min_[i]) {
      closest_point_coordinate = min_[i];
    } else {
      comp.UpdateComparisons();
      if (point[i] < max_[i] && point[i] > min_[i]) {
        closest_point_coordinate = point[i];
      } else {
        closest_point_coordinate = max_[i];
      }
    }
    dist +=(closest_point_coordinate - point[i]) * 
           (closest_point_coordinate - point[i]);
    comp.UpdateComparisons();
    if (dist > range ) {
      return false;
    }
  } 
  return dist <= range;
}   
__TEMPLATE__ 
template<typename POINTTYPE1, typename POINTTYPE2>
inline PRECISION __HyperRectangle__::Distance(POINTTYPE1 point1, 
		                                          POINTTYPE2 point2, 
							         											  int32 dimension) {
  PRECISION distance = 0;
	for(int32 i=0; i< dimension; i++) {
	  distance+=(point1[i]-point2[i]) * (point1[i]-point2[i]);
	}
	return distance;
}
                   	         
__TEMPLATE__
inline PRECISION __HyperRectangle__::Distance(
    HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr1,
    HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr2,
    int32 dimension,
    ComputationsCounter<diagnostic> &comp) {
  
  PRECISION dist=0;
  comp.UpdateDistances();
  for(int32 i=0; i<dimension; i++) {
  	PRECISION d2 = hr1.min_[i] - hr2.max_[i];
  	PRECISION d4 = hr1.max_[i] - hr2.min_[i];
  	if (d2>0) {	
  	  dist += d2*d2 ;
  	  continue;
  	} 
  	if (d4<0) {
  	  dist += d4*d4;
  	}
  }
  return dist;

}

__TEMPLATE__
inline PRECISION __HyperRectangle__::Distance(
    HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr1,
    HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr2,
    PRECISION threshold_distance,
    int32 dimension,
    ComputationsCounter<diagnostic> &comp) {
  
  PRECISION dist=0;
  comp.UpdateDistances();
  for(int32 i=0; i<dimension; i++) {
  	PRECISION d2 = hr1.min_[i] - hr2.max_[i];
  	PRECISION d4 = hr1.max_[i] - hr2.min_[i];
  	if (d2>0) {	
  	  dist += d2*d2;
  	  if (dist > threshold_distance) {
  	  	return numeric_limits<PRECISION>::max();
  	  } else {
  	    continue;
  	  }
  	} 
  	if (d4<0) {
  	  dist += d4*d4;
  	  if (dist > threshold_distance) {
  	  	return numeric_limits<PRECISION>::max();
  	  }
  	}
  }
  return dist;
}

__TEMPLATE__
template<typename POINTTYPE, typename NODETYPE> 
inline pair<typename ALLOCATOR:: template Ptr<NODETYPE>, 
	   typename ALLOCATOR:: template Ptr<NODETYPE> > __HyperRectangle__::
    ClosestChild(typename ALLOCATOR::template Ptr<NODETYPE> left,
		           typename ALLOCATOR::template Ptr<NODETYPE> right,
						 	 POINTTYPE point,
							 int32 dimension,
							 ComputationsCounter<diagnostic> &comp) {
  comp.UpdateComparisons();  	
  if (point[pivot_dimension_] < pivot_value_) {
  	return make_pair(left, right);
  } else {
  	return make_pair(right, left);      	
  }
} 

__TEMPLATE__
string  __HyperRectangle__::Print(int32 dimension) {
  char buf[8192];
  sprintf(buf, "max: ");
  string str;
	str.append(buf);
  for(int32 i=0; i<dimension; i++) {
    sprintf(buf, " %f ", max_[i]);
    str.append(buf);
  }
  sprintf(buf, "\n");
  str.append(buf);
  sprintf(buf, "min: ");
  str.append(buf);
  for(int32 i=0; i<dimension; i++) {
    sprintf(buf, " %f ", min_[i]);
    str.append(buf);
  }
  sprintf(buf, "\n");
  str.append(buf);
  
	return str;
}
#undef __TEMPLATE__
#undef __HyperRectangle__

#endif /*HYPER_RECTANGLE_IMPL_H_*/
