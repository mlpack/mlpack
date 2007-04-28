/*
 * =====================================================================================
 * 
 *       Filename:  hyper_ball_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/11/2007 07:08:40 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef HYPER_BALL_IMPL_H_
#define HYPER_BALL_IMPL_H_
#define __TEMPLATE__            \
 template<typename PRECISION,  \
          typename METRIC,      \
          typename ALLOCATOR,   \
          bool     diagnostic>
#define __HYPERBALL__           \
 HyperBall<PRECISION, METRIC, ALLOCATOR, diagnostic>

__TEMPLATE__
__HYPERBALL__::HyperBall(PivotData &pivot) : center_(pivot.center_), 
	radious_(sqrt(pivot.radious_)), pivot_left_(pivot.left_), 
	pivot_right_(pivot.right_) {
}

__TEMPLATE__
static void *__HYPERBALL__::operator new(size_t size) {
  return typename ALLOCATOR::malloc(size);
}

__TEMPLATE__
static void  __HYPERBALL__::operator delete(void *p) {
}

__TEMPLATE__
inline HyperBall<PRECISION, METRIC, ALLOCATOR, diagnostic> &__HYPERBALL__::operator=(
		const HyperBall<PRECISION, METRIC, ALLOCATOR, diagnostic>  &other) {
  center_ = other.center_;
	radious_ = other.radious_;
	pivot_left_ = other.pivot_left_;
  pivot_right_ = other.pivot_right_;
	return *this;
}

__TEMPLATE__
template<typename POINTTYPE> 
inline bool __HYPERBALL__::IsWithin(POINTTYPE point, 
		                                int32 dimension, 
				  													PRECISION range, 
                                    ComputationsCounter<diagnostic> &comp) {
  comp.UpdateDistances();
	PRECISION point_center_distance = 
		 Metric_t::Distance(center_, point, dimension);
  // Inside the ball
	comp.UpdateComparisons();
 	if (radious_ > sqrt(point_center_distance) + sqrt(range)) {
	  return true;
	} else {
		// Completelly outside or crossing
	  return false;
	}
	
}

__TEMPLATE__
inline PRECISION __HYPERBALL__::IsWithin(HyperBall_t &hr,
    int32 dimension, 
    PRECISION range,
    ComputationsCounter<diagnostic> &comp) {
  return IsWithin(hr.center_, dimension, hr.radious_ * hr.radious_, comp);
}

__TEMPLATE__
template<typename POINTTYPE> 
inline bool __HYPERBALL__::CrossesBoundaries(POINTTYPE point, 
    int32 dimension, 
		PRECISION range, 
    ComputationsCounter<diagnostic> &comp) {
	comp.UpdateDistances();
  PRECISION point_center_distance = Metric_t::Distance(center_, point, dimension);
  comp.UpdateComparisons();
	if (point_center_distance < range) {
	  return true;
	}
	comp.UpdateComparisons();
 	if (sqrt(point_center_distance) >  radious_+ sqrt(range)) {
		return false;
	} else {
	  return true;
	}
	
}

__TEMPLATE__
template<typename POINTTYPE1, typename POINTTYPE2>
inline PRECISION __HYPERBALL__::Distance(POINTTYPE1 point1, 
                                         POINTTYPE2 point2, 
                                         int32 dimension) {
  return  Metric_t::Distance(point1, point2, dimension);
}

__TEMPLATE__
inline PRECISION __HYPERBALL__::Distance(HyperBall_t &hr1,
    HyperBall_t &hr2,
    int32 dimension,
    ComputationsCounter<diagnostic> &comp) {
	comp.UpdateDistances();
	PRECISION center_distances=Metric_t::Distance(hr1.center_, 
			                           hr2.center_, dimension);
	comp.UpdateComparisons();
	PRECISION dist=sqrt(center_distances)-(hr2.radious_+hr1.radious_);
	if (dist<=0) {
	  return 0;
	} else {
		return dist*dist;
	}
    
}

__TEMPLATE__
PRECISION __HYPERBALL__::Distance(HyperBall_t &hr1,
                                  HyperBall_t &hr2,
                                  PRECISION threshold_distance,
                                  int32 dimension,
                                  ComputationsCounter<diagnostic> &comp) {
	fprintf(stderr, "Not Implemented yet\n");
	assert("false");
	return 0;
}                           

__TEMPLATE__
template<typename POINTTYPE, typename NODETYPE>                   	
inline pair<typename ALLOCATOR::template Ptr<NODETYPE>, 
	   typename ALLOCATOR::template Ptr<NODETYPE> > 
		 __HYPERBALL__::ClosestChild(typename ALLOCATOR::template Ptr<NODETYPE> left, 
		                             typename ALLOCATOR::template Ptr<NODETYPE> right,
                                 POINTTYPE point,
																 int32 dimension,
																 ComputationsCounter<diagnostic> &comp) {
	comp.UpdateDistances();
  comp.UpdateDistances();	
	PRECISION left_dist = Metric_t::Distance(pivot_left_, point, dimension);
	PRECISION right_dist = Metric_t::Distance(pivot_right_, point, dimension);
	if (left_dist<right_dist) {
	  return make_pair(left, right);
	} else {
	  return make_pair(right, left);  
	}
}                            

__TEMPLATE__
string __HYPERBALL__::Print(int32 dimension) {
  char buf[8192];
  string str("center: ");
  for(int32 i=0; i<dimension; i++){
	  sprintf(buf,"%lg ", (double) center_[i]);
    str.append(buf);
	}
	str.append("\nradious: ");
	sprintf(buf,"%lg\n", (double) radious_);
  str.append(buf);
	str.append("pivot_left: ");
  for(int32 i=0; i<dimension; i++){
	  sprintf(buf,"%lg ", (double)pivot_left_[i]);
    str.append(buf);
	}
  str.append("\npivot_right: ");
  for(int32 i=0; i<dimension; i++){
	  sprintf(buf,"%lg ", (double)pivot_right_[i]);
    str.append(buf);
	}
	str.append("\n");
	return str;
}

#undef __TEMPLATE__
#undef __HYPERBALL__
#endif // HYPER_BALL_IMPL_H_
