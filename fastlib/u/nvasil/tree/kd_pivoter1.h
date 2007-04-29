/*
 * =====================================================================================
 * 
 *       Filename:  HyperRectanglePivoter.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/28/2007 12:03:05 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifdef KD_PIVOTER1_H_
#define KD_PIVOTER1_H_

#include "loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "dataset/binary_dataset.h"
#include "hyper_rectangle.h"

template<typename TYPELIST, bool diagnostic>
class KdPivoter1 {
 public:
	typedef Loki::TL::TypeAt<TYPELIST, 0>::Result Precision_t;
	typedef Loki::TL::TypeAt<TYPELIST, 1>::Result Allocator_t;
  typedef Loki::TL::TypeAt<TYPELIST, 2>::Result Metric_t;
  typedef HyperBall<TYPELIST, diagnostic> HyperBall_t;
  FORBID_COPY(KdPivoter1)
	struct PivotInfo {
	 public:
	  void Init(index_t start, index_t num_of_points, HyperRectangle_t &box) {
		  box_.Copy(box_);
			start_=start;
			num_of_points_=num_of_points;
		}	 
		HyperRectangle_t box_;
    Loki::NullType statistics_;
		index_t start_;
		index_t num_of_points_;
	}; 
 	Init(BinaryDataset<Precision_t> *data) {
	  data_=data;
	}
	
  pair<PivotInfo*, PivotInfo*> operator()(PivotInfo *pivot) {	
	  index_t left_start = pivot->start_;
	  index_t right_start = pivot->start_ + pivot->num_of_points_-1;
    int32 dimension = data_->get_dimension();	
    int32 max_range_dimension = pivot->box_.get_pivot_dimension();
    Precision_t pivot_value = pivot->box_.get_pivot_value();                  
    Precision_t *point_left = data_->At(left_start);
    Precision_t *point_right = data_->At(right_start); 
    index_t left_points=0;
    index_t right_points=0;   
    HyperRectangle_t  hr_left;
	  HyperRectangle_t  hr_right;
    hr_left.Init(dimension);
    hr_right.Init(dimension);

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
    DEBUG_ASSERT(left_points+right_points == right_start - left_start+1);
    FindPivotDimensionValue(hr_left);
    FindPivotDimensionValue(hr_right);  
  
    PivotInfo* pv_left  =  new PivotInfo();
		pv_left->Init(left_start, left_points, hr_left);
	  PivotInfo* pv_right =  new PivotInfo();
		pv_right->Init(left_start+left_points, right_points, hr_right);
    return make_pair(pv_left, pv_right);                                       
  }
  
	PivotInfo *operator()(index_t num_of_points) {
    // make the parent
	  HyperRectangle_t hr;
    hr.Init(data_->get_dimension());
    for(index_t i=0; i<num_of_points; i++) {
  	  Precision_t *point = data->At(i);
      UpdateHyperRectangle(point, hr);
    }
    FindPivotDimensionValue(hr); 
    PivotInfo *pv = new PivotInfo(0, num_of_points, hr);
	  return pv;                               
  }

 private:
	BinaryDataset<Precision_t> *data_;
  
	void FindPivotDimensionValue(HyperRectangle_t &hr) {
    Precision_t max_range = 0;
    int32 max_range_dimension =0;
    for(int32 j=0; j<data_->get_dimension(); j++) {
      Precision_t range = hr.get_max()[j] - hr.get_min()[j];
      if (range > max_range) {
        max_range = range;
        max_range_dimension = j;
      }
    }
    hr.set_pivot_value((hr.get_max()[max_range_dimension] + 
                        hr.get_min()[max_range_dimension])/2);
    hr.set_pivot_dimension(max_range_dimension); 
  }

  void UpdateHyperRectangle(Precision_t *point,
	  	                      HyperRectangle_t &hr) {
    for(int32 j=0; j<data_.get_dimension(); j++) {
      if (point[j] > hr.get_max()[j]) {
        hr.get_max()[j] = point[j];
      } 
      if (point[j] < hr.get_min()[j]) {
          hr.get_min()[j] = point[j];
      } 
    }
  }

};


#endif // KD_PIVOTER_H_
