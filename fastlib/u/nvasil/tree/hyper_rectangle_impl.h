#ifndef HYPER_RECTANGLE_IMPL_H_
#define HYPER_RECTANGLE_IMPL_H_

#define TEMPLATE__ template<typename TYPELIST, bool diagnostic>
#define HYPERRECTANGLE__ HyperRectangle<TYPELIST, diagnostic>

TEMPLATE__
HYPERRECTANGLE__::HyperRectangle(){
}

TEMPLATE__
void HYPERRECTANGLE__::Init(int32 dimension) {
  min_.Reset(Allocator_t:: template calloc<Precision_t>
			 (dimension, numeric_limits<Precision_t>::max()));
	max_.Reset(Allocator_t:: template calloc<Precision_t>
			 (dimension, -numeric_limits<Precision_t>::max()));
	pivot_dimension_=0;
	pivot_value_=0;
}
TEMPLATE__
void HYPERRECTANGLE__::Init(Array_t min, Array_t max, int32 pivot_dimension,
		                     Precision_t pivot_value) { 
	min_ = min;
  max_ = max;
  pivot_dimension_ = pivot_dimension;
  pivot_value_= pivot_value;
}


TEMPLATE__
HyperRectangle<TYPELIST, diagnostic> &HYPERRECTANGLE__::operator=
    (HyperRectangle<TYPELIST, diagnostic> &hr) {
  
  this->min_ = hr.min_;
  this->max_ = hr.max_;
  pivot_dimension_ = hr.pivot_dimension_;
  pivot_value_ = hr.pivot_value_;
  return *this;
}

TEMPLATE__
void HYPERRECTANGLE__::Alias(const HyperRectangle_t &hr) {
  
  this->min_ = hr.min_;
  this->max_ = hr.max_;
  pivot_dimension_ = hr.pivot_dimension_;
  pivot_value_ = hr.pivot_value_;
}


TEMPLATE__
void HYPERRECTANGLE__::Copy(const HyperRectangle_t &hr, 
		                              int32 dimension) {
  
  this->min_.Copy(hr.min_, dimension);
  this->max_.Copy(hr.max_, dimension);
  pivot_dimension_ = hr.pivot_dimension_;
  pivot_value_ = hr.pivot_value_;
}

TEMPLATE__
void *HYPERRECTANGLE__::operator new(size_t size) {
	typename Allocator_t::template Ptr<HyperRectangle_t> temp;
	temp.Reset(Allocator_t::malloc(size));
  return (void *)temp.get();
}

TEMPLATE__
void HYPERRECTANGLE__::operator delete(void *p) {
}

TEMPLATE__
template<typename POINTTYPE> 
inline bool HYPERRECTANGLE__::IsWithin(
    POINTTYPE point, int32 dimension, Precision_t range, 
    ComputationsCounter<diagnostic> &comp) {
  // non overlaping at all 
  for(int32 i=0; i<dimension; i++) {
    comp.UpdateComparisons();
    if ( point[i] > max_[i] || point[i] < min_[i]) {
  	  return false;
  	}
  }
  Precision_t closest_projection = max_[0]-min_[0]; 
  for(int32 i=0; i<dimension; i++) {
    Precision_t projection1 = max_[i] - point[i];
    Precision_t projection2 = point[i] - min_[i];
    comp.UpdateComparisons();
    if (closest_projection > projection1) {
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
    	
     
TEMPLATE__
inline typename HYPERRECTANGLE__::Precision_t HYPERRECTANGLE__::IsWithin(
    HyperRectangle_t &hr,
    int32 dimension, 
    Precision_t range,
    ComputationsCounter<diagnostic> &comp) {
  
  Precision_t closest_projection = numeric_limits<Precision_t>::max();
  for(int32 i=0; i<dimension; i++) {
    comp.UpdateComparisons();
    comp.UpdateComparisons();
    Precision_t d1=hr.min_[i] - min_[i];
    Precision_t d2=max_[i] - hr.max_[i];
    if (d1<0 || d2<0) {
      return -1;
    } else {
      Precision_t dist = min(d1,d2);
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

TEMPLATE__
template<typename POINTTYPE> 
inline bool HYPERRECTANGLE__::CrossesBoundaries(
    POINTTYPE point, int32 dimension, HYPERRECTANGLE__::Precision_t range, 
    ComputationsCounter<diagnostic> &comp) {
  
  Precision_t closest_point_coordinate;
  Precision_t dist = 0;
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
TEMPLATE__ 
template<typename POINTTYPE1, typename POINTTYPE2>
inline typename HYPERRECTANGLE__::Precision_t HYPERRECTANGLE__::Distance(
		                                            POINTTYPE1 point1, 
		                                            POINTTYPE2 point2, 
							         											    int32 dimension) {
  Precision_t distance = 0;
	for(int32 i=0; i< dimension; i++) {
	  distance+=(point1[i]-point2[i]) * (point1[i]-point2[i]);
	}
	return distance;
}
                   	         
TEMPLATE__
inline typename HYPERRECTANGLE__::Precision_t HYPERRECTANGLE__::Distance(
    typename HYPERRECTANGLE__::HyperRectangle_t &hr1,
    typename HYPERRECTANGLE__::HyperRectangle_t &hr2,
    int32 dimension,
    ComputationsCounter<diagnostic> &comp) {
  
  Precision_t dist=0;
  comp.UpdateDistances();
  for(int32 i=0; i<dimension; i++) {
  	Precision_t d2 = hr1.min_[i] - hr2.max_[i];
  	Precision_t d4 = hr1.max_[i] - hr2.min_[i];
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

TEMPLATE__
inline typename HYPERRECTANGLE__::Precision_t HYPERRECTANGLE__::Distance(
    typename HYPERRECTANGLE__::HyperRectangle_t &hr1,
    typename HYPERRECTANGLE__::HyperRectangle_t &hr2,
    typename HYPERRECTANGLE__::Precision_t threshold_distance,
    int32 dimension,
    ComputationsCounter<diagnostic> &comp) {
  
  Precision_t dist=0;
  comp.UpdateDistances();
  for(int32 i=0; i<dimension; i++) {
  	Precision_t d2 = hr1.min_[i] - hr2.max_[i];
  	Precision_t d4 = hr1.max_[i] - hr2.min_[i];
  	if (d2>0) {	
  	  dist += d2*d2;
  	  if (dist > threshold_distance) {
  	  	return numeric_limits<Precision_t>::max();
  	  } else {
  	    continue;
  	  }
  	} 
  	if (d4<0) {
  	  dist += d4*d4;
  	  if (dist > threshold_distance) {
  	  	return numeric_limits<Precision_t>::max();
  	  }
  	}
  }
  return dist;
}

TEMPLATE__
template<typename POINTTYPE, typename NODETYPE> 
inline pair<typename HYPERRECTANGLE__::Allocator_t:: template Ptr<NODETYPE>, 
	  typename HYPERRECTANGLE__::Allocator_t:: template Ptr<NODETYPE> > 
		HYPERRECTANGLE__::ClosestChild(
	  typename HYPERRECTANGLE__::Allocator_t::template Ptr<NODETYPE> left,
		typename HYPERRECTANGLE__::Allocator_t::template Ptr<NODETYPE> right,
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

TEMPLATE__
string  HYPERRECTANGLE__::Print(int32 dimension) {
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
#undef TEMPLATE__
#undef HYPERRECTANGLE__

#endif /*HYPER_RECTANGLE_IMPL_H_*/
