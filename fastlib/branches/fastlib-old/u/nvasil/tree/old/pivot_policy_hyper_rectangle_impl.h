/*
 * =====================================================================================
 * 
 *       Filename:  pivot_policy_impl.h
 * 
 *    Description:  
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
#ifndef PIVO_POLICY_HYPER_RECTANGLE_IMPL_H_
#define PIVO_POLICY_HYPER_RECTANGLE_IMPL_H_
#include <assert.h>

#define __TEMPLATE__  template<typename PRECISION,     \
                      typename IDPRECISION,            \
                      typename ALLOCATOR,              \
                      bool     diagnostic>   

#define  __PivotPolicy__ PivotPolicy<PRECISION,       	\
                         IDPRECISION, ALLOCATOR,			  \
                         KdNode<PRECISION, IDPRECISION, \
									       ALLOCATOR, diagnostic>,        \
                         diagnostic>
 	
 
__TEMPLATE__
pair<typename __PivotPolicy__::Pivot_t*, typename __PivotPolicy__::Pivot_t*> 
    __PivotPolicy__::Pivot(DataReader<PRECISION, IDPRECISION> *data,
				typename __PivotPolicy__::Pivot_t *pivot) {
  
  IDPRECISION left_start = pivot->start_;
	IDPRECISION right_start = pivot->start_ + pivot->num_of_points_-1;
  int32 dimension = pivot->box_pivot_data_.dimension_;	
  int32 max_range_dimension = pivot->box_pivot_data_.pivot_dimension_;
  PRECISION pivot_value = pivot->box_pivot_data_.pivot_value_;                  
  PRECISION *point_left = data->At(left_start);
  PRECISION *point_right = data->At(right_start); 
  IDPRECISION left_points=0;
  IDPRECISION right_points=0;   
  typename Box_t::PivotData hr_left, hr_right;
  hr_left.min_.Reset(ALLOCATOR:: template calloc<PRECISION>(dimension, 
			               numeric_limits<PRECISION>::max()));
  hr_right.min_.Reset(ALLOCATOR::template calloc<PRECISION>(dimension, 
			                numeric_limits<PRECISION>::max()));
  hr_left.max_.Reset(ALLOCATOR::template calloc<PRECISION>(dimension, 
			               -numeric_limits<PRECISION>::max()));
  hr_right.max_.Reset(ALLOCATOR::template calloc<PRECISION>(dimension, 
			                -numeric_limits<PRECISION>::max()));
  hr_left.dimension_ = dimension;
  hr_right.dimension_ = dimension;	

	while (true) {
    while (left_points < pivot->num_of_points_ &&
           point_left[max_range_dimension] < pivot_value) {
      UpdateHyperRectangle(point_left, hr_left);
      left_points++;
			point_left = data->At(left_start+left_points);
    } 	   
    if (point_left > point_right || left_points == pivot->num_of_points_) {	
      break;
    } 	
    while (right_points < pivot->num_of_points_ &&
           point_right[max_range_dimension] >= pivot_value) {
      UpdateHyperRectangle(point_right, hr_right);
      right_points++;
      point_right = data->At(right_start - right_points);
    } 	
    if (point_left > point_right || right_points == pivot->num_of_points_) {
      break;
    } 	 
    data->Swap(left_start+left_points, right_start - right_points);
  }
  assert(left_points+right_points == right_start - left_start+1);
  FindPivotDimensionValue(hr_left);
  FindPivotDimensionValue(hr_right);  
  
  Pivot_t* pv_left  =  new Pivot_t(left_start, left_points, hr_left);
	Pivot_t* pv_right =  new Pivot_t(left_start+left_points, right_points, 
			                             hr_right);
  return make_pair(pv_left, pv_right);                                       
}
__TEMPLATE__
 typename __PivotPolicy__::Pivot_t* __PivotPolicy__::PivotParent(
		 DataReader<PRECISION, IDPRECISION> *data, 
     IDPRECISION num_of_points, int32 dimension) {
  // make the parent
	typename Box_t::PivotData hr;
  hr.min_.Reset(ALLOCATOR::template calloc<PRECISION>(dimension, 
				                                     numeric_limits<PRECISION>::max()));
  hr.max_.Reset(ALLOCATOR::template calloc<PRECISION>(dimension, 
				                                     -numeric_limits<PRECISION>::max()));
  hr.dimension_ = dimension;
 
  for(IDPRECISION i=0; i<num_of_points; i++) {
  	PRECISION *point = data->At(i);
    UpdateHyperRectangle(point, hr);
  }
  FindPivotDimensionValue(hr); 
  Pivot_t *pv = new Pivot_t(0, num_of_points, hr);
	return pv;                               
}



__TEMPLATE__
void __PivotPolicy__::FindPivotDimensionValue(typename Box_t::PivotData &pv) {
  PRECISION max_range = 0;
  int32 max_range_dimension =0;
  for(int32 j=0; j<pv.dimension_; j++) {
    PRECISION range = pv.max_[j] - pv.min_[j];
    if (range > max_range) {
      max_range = range;
      max_range_dimension = j;
    }
  }
  pv.pivot_value_ =  (pv.max_[max_range_dimension] + 
                     pv.min_[max_range_dimension])/2;
  pv.pivot_dimension_ = max_range_dimension; 
}

__TEMPLATE__
void __PivotPolicy__::UpdateHyperRectangle(PRECISION *point,
		                                       typename Box_t::PivotData &pv) {
  for(int32 j=0; j<pv.dimension_; j++) {
    if (point[j] > pv.max_[j]) {
      pv.max_[j] = point[j];
    } 
    if (point[j] < pv.min_[j]) {
        pv.min_[j] = point[j];
    } 
  }
}



#undef __TEMPLATE__
#undef __PivotPolicy__
#endif // PIVOT_POLICY_HYPPER_RECTANGLE_IMPL_H_

