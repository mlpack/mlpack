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

#ifndef KD_PIVOTER2_H_
#define KD_PIVOTER2_H_

#include "fastlib/fastlib.h"
#include "u/nvasil/dataset/binary_dataset.h"
#include "u/nvasil/tree/hyper_rectangle.h"
using namespace std;
template<typename TYPELIST, bool diagnostic>
class KdPivoter2 {
 public:
	typedef typename TYPELIST::Precision_t Precision_t;
	typedef typename TYPELIST::Allocator_t Allocator_t;
  typedef typename TYPELIST::Metric_t    Metric_t;
  typedef HyperRectangle<TYPELIST, diagnostic> HyperRectangle_t;
	
	struct PivotInfo {
	 public:
	  void Init(index_t start, index_t num_of_points, HyperRectangle_t &box) {
		  box_.Alias(box);
			start_=start;
			num_of_points_=num_of_points;
		}	 
		HyperRectangle_t box_;
    NullStatistics<> statistics_;
		index_t start_;
		index_t num_of_points_;
	};
  class Comparator {
	 public:	 
	 void Init(index_t pivot_dimension) {
		  pivot_dimension_=pivot_dimension;
		}
		bool operator()(const CompletePoint<Precision_t> &p1,
			              const  CompletePoint<Precision_t> &p2)  const {
		  return p1.At(pivot_dimension_)<p2.At(pivot_dimension_);
		}
	 private:		
		index_t pivot_dimension_;
	};	
 	void Init(BinaryDataset<Precision_t> *data) {
	  data_=data;
	}
	
  pair<PivotInfo*, PivotInfo*> operator()(PivotInfo *pivot) {	
		int32 dimension = data_->get_dimension();	
		index_t left_points  = pivot->num_of_points_/2;
		index_t right_points = pivot->num_of_points_ - left_points;
		typename BinaryDataset<Precision_t>::Iterator it=
			  data_->Begin()+pivot->start_+left_points;
		Comparator comp;
		comp.Init(pivot->box_.get_pivot_dimension());
		std::nth_element(data_->Begin()+pivot->start_, it, 
				             data_->Begin()+pivot->start_+pivot->num_of_points_, comp);

	  HyperRectangle_t  hr_left;
	  HyperRectangle_t  hr_right;
    hr_left.Init(dimension);
    hr_right.Init(dimension);
    index_t left_start=pivot->start_;
		for(index_t i=left_start; i<left_start+left_points; i++) {
		  UpdateHyperRectangle(data_->At(i), hr_left);
		}
    for(index_t i=left_start+left_points; i<left_start+pivot->num_of_points_; i++) {
		  UpdateHyperRectangle(data_->At(i), hr_right);
		} 
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
  	  Precision_t *point = data_->At(i);
      UpdateHyperRectangle(point, hr);
    }
    FindPivotDimensionValue(hr); 
    PivotInfo *pv = new PivotInfo();
		pv->Init(0, num_of_points, hr);
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
    for(int32 j=0; j<data_->get_dimension(); j++) {
      if (point[j] > hr.get_max()[j]) {
        hr.get_max()[j] = point[j];
      } 
      if (point[j] < hr.get_min()[j]) {
          hr.get_min()[j] = point[j];
      } 
    }
  }
  
};


#endif // KD_PIVOTER2_H_
