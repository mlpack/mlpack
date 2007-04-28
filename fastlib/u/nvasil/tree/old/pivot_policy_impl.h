/*
 * =====================================================================================
 * 
 *       Filename:  pivot_policy_impl.h
 * 
 *    Description: The general case is considered to be hyper_balls 
 *
 * 
 *        Version:  1.0
 *        Created:  02/01/2007 11:29:34 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef PIVOT_POLICY_IMPL_H_
#define PIVOT_POLICY_IMPL_H_
#include <assert.h>

#define __TEMPLATE__  template<typename PRECISION,     \
                      typename IDPRECISION,            \
                      typename ALLOCATOR,              \
                      typename NODE,                   \
                      bool     diagnostic>   

#define  __PivotPolicy__ PivotPolicy<PRECISION,       	\
                         IDPRECISION, ALLOCATOR,			  \
                         NODE,                          \
                         diagnostic>
 	
 
__TEMPLATE__
pair<typename __PivotPolicy__::Pivot_t*, typename __PivotPolicy__::Pivot_t*> 
    __PivotPolicy__::Pivot(DataReader<PRECISION, IDPRECISION> *data,
				typename __PivotPolicy__::Pivot_t *pivot) {
  
  IDPRECISION left_start = pivot->start_;
	IDPRECISION right_start = pivot->start_ + pivot->num_of_points_-1;
  int32 dimension = pivot->box_pivot_data_.dimension_;	
	Array_t pivot_left  = pivot->box_pivot_data_.left_;
	Array_t pivot_right = pivot->box_pivot_data_.right_;
  PRECISION *point_left = data->At(left_start);
  PRECISION *point_right = data->At(right_start); 
  IDPRECISION left_points=0;
  IDPRECISION right_points=0;   
  typename Box_t::PivotData pv_left(dimension), pv_right(dimension);
	
	while (true) {
    while (left_points < pivot->num_of_points_ &&
           IsInLeftPivot(point_left, pivot_left, pivot_right, dimension)) {
      UpdateHyperBall(point_left, pv_left);
      left_points++;
			point_left = data->At(left_start+left_points);
    } 	   
    if (point_left > point_right || left_points == pivot->num_of_points_) {	
      break;
    } 	
    while (right_points < pivot->num_of_points_ &&
           !IsInLeftPivot(point_right, pivot_left, pivot_right, dimension)) {
      UpdateHyperBall(point_right, pv_right);
      right_points++;
      point_right = data->At(right_start - right_points);
    } 	
    if (point_left > point_right || right_points == pivot->num_of_points_) {
      break;
    } 	 
    data->Swap(left_start+left_points, right_start - right_points);
  }
	assert(left_points>0);
	assert(right_points>0);
  assert(left_points+right_points == right_start - left_start+1);
  NormalizeHyperBall(pv_left, left_points);
	NormalizeHyperBall(pv_right, right_points);
	FindPivotPoints(data, pv_left, left_start, left_points);
  FindPivotPoints(data, pv_right, left_start+left_points, right_points);  
  
  Pivot_t* node_pv_left  =  new Pivot_t(left_start, left_points, pv_left);
	Pivot_t* node_pv_right =  new Pivot_t(left_start+left_points, right_points, 
			                             pv_right);
  return make_pair(node_pv_left, node_pv_right);                                       
}
__TEMPLATE__
 typename __PivotPolicy__::Pivot_t* __PivotPolicy__::PivotParent(
		 DataReader<PRECISION, IDPRECISION> *data, 
     IDPRECISION num_of_points, int32 dimension) {
  // make the parent
	typename Box_t::PivotData pivot(dimension) ;
  for(IDPRECISION i=0; i<num_of_points; i++) {
	  UpdateHyperBall(data->At(i), pivot);
	}
	NormalizeHyperBall(pivot, num_of_points);
	FindPivotPoints(data, pivot, 0, num_of_points);
	Pivot_t *pv = new Pivot_t(0, num_of_points, pivot);
	return pv;

}

__TEMPLATE__
void __PivotPolicy__::FindPivotPoints(DataReader<PRECISION, IDPRECISION> *data,
		                                  typename Box_t::PivotData &pivot,
																			IDPRECISION start,
																			IDPRECISION num_of_points) {
 	assert(num_of_points>0);
	PRECISION radious=0;
	PRECISION max_distance=0;
  IDPRECISION furthest_point_index = 0;
	int32 dimension = pivot.dimension_;
	IDPRECISION random_index = IDPRECISION(num_of_points*1.0
			                       *rand()/RAND_MAX)+start;
  for(IDPRECISION i=start; i< start+num_of_points; i++) {
    PRECISION distance1 = Metric_t::Distance(pivot.center_, 
				                                     data->At(i), dimension);
	  PRECISION distance = Metric_t::Distance(data->At(random_index),
			                   data->At(i), dimension);
		if (distance1 > radious) {
		  radious = distance1;
		}
		if (distance>max_distance) {
		  furthest_point_index=i;
			max_distance=distance;
		}
	}
	pivot.radious_ = radious;
	pivot.left_.DeepCopy(data->At(furthest_point_index), dimension);
  max_distance = 0;

  for(IDPRECISION i=start; i< start+num_of_points; i++) {
		PRECISION distance = Metric_t::Distance(pivot.left_, data->At(i), dimension);
	  if (distance > max_distance) {
			furthest_point_index = i;
			max_distance=distance;
		}
	}
  pivot.right_.DeepCopy(data->At(furthest_point_index), dimension);
	assert(Metric_t::Distance(pivot.left_, pivot.right_, dimension)>0);

}


__TEMPLATE__
template<typename POINTTYPE>
void __PivotPolicy__::UpdateHyperBall(POINTTYPE point, 
		                                  typename Box_t::PivotData &pivot) {
	Metric_t::Addition(pivot.center_, pivot.center_, point, pivot.dimension_);
}

__TEMPLATE__
void __PivotPolicy__::NormalizeHyperBall(typename Box_t::PivotData &pivot, 
		                                     IDPRECISION num_of_points) {
	Metric_t::Scale(pivot.center_, pivot.center_, 
			            1.0/num_of_points, pivot.dimension_);
	
}

__TEMPLATE__
template<typename POINTTYPE>
pair<typename __PivotPolicy__::Array_t, PRECISION> 
__PivotPolicy__::FindFurthestPoint(POINTTYPE point,
		DataReader<PRECISION, IDPRECISION> *data, IDPRECISION start, 
		IDPRECISION num_of_points, 
		int32 dimension) {
	PRECISION radious=0;
	IDPRECISION furthest_point_index;
	Array_t furthest_point(dimension);
  for(IDPRECISION i=start; i<start+num_of_points; i++) {
		PRECISION distance = Metric_t::Distance(point, data->At(i), dimension);
	  if (distance > radious) {
		  radious = distance;
			furthest_point_index = i;
		}
	}
  furthest_point.DeepCopy(data->At(furthest_point_index), dimension);
  return make_pair(furthest_point, radious);
}
                                                      
__TEMPLATE__
template<typename POINTTYPE>
bool __PivotPolicy__::IsInLeftPivot(PRECISION *point, POINTTYPE left,
	                                 	POINTTYPE right,
		                                int32 dimension) {
  PRECISION left_distance = Metric_t::Distance(point, left, dimension);
	PRECISION right_distance = Metric_t::Distance(point, right, dimension);
	if (left_distance < right_distance) {
	  return true;
	}
  return false;
}


#undef __TEMPLATE__
#undef __PivotPolicy__
#endif // PIVOT_POLICY_IMPL_H_

