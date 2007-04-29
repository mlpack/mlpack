/*
 * =====================================================================================
 * 
 *       Filename:  ball_pivoter.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/28/2007 04:15:36 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifdef  BALL_PIVOTER_H_
#define BALL_PIVOTER_H_

#include "loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "dataset/binary_dataset.h"
#include "hyper_ball.h"

template<typename TYPELIST, bool diagnostic>
class BallPivoter {
 public:
	typedef Loki::TL::TypeAt<TYPELIST, 0>::Result Precision_t;
	typedef Loki::TL::TypeAt<TYPELIST, 1>::Result Allocator_t;
  typedef Loki::TL::TypeAt<TYPELIST, 2>::Result Metric_t;
  typedef HyperBall<TYPELIST, diagnostic> HyperBall_t;
  FORBID_COPY(HyperBallPivoter)
	struct PivotInfo {
	 public:
	  void Init(index_t start, index_t num_of_points, HyperBall_t &box) {
		  box_.Copy(box_);
			start_=start;
			num_of_points_=num_of_points;
		}	 
		HyperBall_t box_;
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
	  Array_t pivot_left  = pivot->box_.get_pivot_left();
	  Array_t pivot_right = pivot->box_.get_pivot_right();
    Precision_t *point_left = data->At(left_start);
    Precision_t *point_right = data->At(right_start); 
    index_t left_points=0;
    index_t right_points=0;   
    HyperBall_t ball_left, ball_right;
		ball_left.Init(dimension);
		ball_right.Init(dimension);
	
	  while (true) {
      while (left_points < pivot->num_of_points_ &&
             IsInLeftPivot(point_left, ball_left, ball_right, dimension)) {
        UpdateHyperBall(point_left, ball_left);
        left_points++;
			  point_left = data->At(left_start+left_points);
      } 	   
      if (point_left > point_right || left_points == pivot->num_of_points_) {	
        break;
      } 	
      while (right_points < pivot->num_of_points_ &&
             !IsInLeftPivot(point_right, ball_left, ball_right, dimension)) {
        UpdateHyperBall(point_right, ball_right);
        right_points++;
        point_right = data->At(right_start - right_points);
      } 	
      if (point_left > point_right || right_points == pivot->num_of_points_) {
        break;
      } 	 
      data->Swap(left_start+left_points, right_start - right_points);
    }
	  DEBUG_ASSERT(left_points>0);
	  DEBUG_ASSERT(right_points>0);
    DEBUG_ASSERT(left_points+right_points == right_start - left_start+1);
    NormalizeHyperBall(ball_left, left_points);
	  NormalizeHyperBall(ball_right, right_points);
	  FindPivotPoints(data, ball_left, left_start, left_points);
    FindPivotPoints(data, ball_right, left_start+left_points, right_points);  
   
    PivotInfo* node_pv_left  =  new PivotInfo();
		node_pv_left->Init(left_start, left_points, ball_left);
	  PivotInfo* node_pv_right =  new PivotInfo();
		node_pv_right->Init(left_start+left_points, right_points, ball_right);

    return make_pair(node_pv_left, node_pv_right);                                       
	
	}
  
	PivotInfo *operator()(index_t num_of_points) {
    // make the parent
	  HyperBall_t ball;
		ball.Init(data_->get_dimension());
    for(index_t i=0; i<num_of_points; i++) {
	    UpdateHyperBall(data->At(i), ball);
	  }
	  NormalizeHyperBall(ball, num_of_points);
 	  FindPivotPoints(ball, 0, num_of_points);
	  PivotInfo *pv = new PivoInfo();
		pv->Init(0, num_of_points, ball);
	  return pv;

  }

 private:
	BinaryDataset<Precision_t> *data_;
  void FindPivotPoints(HyperBall_t &ball,
										   index_t start,
											 index_t num_of_points) {
 	DEBUG_ASSERT(num_of_points>0);
	Precision_t radious=0;
	Precision_t max_distance=0;
  index_t furthest_point_index = 0;
	int32 dimension = data->get_dimension();
	index_t random_index = index_t(num_of_points*1.0
			                       *rand()/RAND_MAX)+start;
  for(index_t i=start; i< start+num_of_points; i++) {
    Precision_t distance1 = Metric_t::Distance(pivot.center_, 
				                                       data->At(i), dimension);
	  Precision_t distance = Metric_t::Distance(data->At(random_index),
			                   data->At(i), dimension);
		if (distance1 > radious) {
		  radious = distance1;
		}
		if (distance>max_distance) {
		  furthest_point_index=i;
			max_distance=distance;
		}
	}
	ball.set_radious(radious);
	ball.get_pivot_left().Copy(data->At(furthest_point_index), dimension);
  max_distance = 0;

  for(index_t i=start; i< start+num_of_points; i++) {
		Precision_t distance = Metric_t::Distance(pivot.left_, 
				                                      data->At(i), 
																							dimension);
	  if (distance > max_distance) {
			furthest_point_index = i;
			max_distance=distance;
		}
	}
  ball.get_pivot_right().Copy(data->At(furthest_point_index), dimension);
	DEBUG_ASSERT(Metric_t::Distance(pivot.left_, pivot.right_, dimension)>0);

}

template<typename POINTTYPE>
void UpdateHyperBall(POINTTYPE point, 
		                 HyperBall_T &ball) {
	Metric_t::Addition(ball.get_center(), ball.get_center(), 
			               point, data.get_dimension());
}

__TEMPLATE__
void __PivotPolicy__::NormalizeHyperBall(HyperBall_t &ball, 
		                                     index_t num_of_points) {
	Metric_t::Scale(ball.get_center(), ball.get_center(), 
			            1.0/num_of_points, data->get_dimension());
	
}

template<typename POINTTYPE>
pair<Array_t, Precision_t> FindFurthestPoint(POINTTYPE point,
		                                         index_t start, 
		                                         index_t num_of_points) {
	Precision_t radious=0;
	index_t furthest_point_index;
	Array_t furthest_point(data_->get_dimension());
  for(index_t i=start; i<start+num_of_points; i++) {
		Precision_t distance = Metric_t::Distance(point, 
				                                      data_->At(i), 
				                                      data_->get_dimension());
	  if (distance > radious) {
		  radious = distance;
			furthest_point_index = i;
		}
	}
  furthest_point.Copy(data->At(furthest_point_index), data_->get_dimension());
  return make_pair(furthest_point, radious);
}
                                                      
template<typename POINTTYPE>
bool IsInLeftPivot(Precision_t *point, POINTTYPE left,
	                 POINTTYPE right) {
	int32 dimension=data_->get_dimension();
  Precision_t left_distance = Metric_t::Distance(point, left, dimension);
	Precision_t right_distance = Metric_t::Distance(point, right, dimension);
	if (left_distance < right_distance) {
	  return true;
	}
  return false;
}


 


};


#endif // BALL_PIVOTER
