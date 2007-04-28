/*
 * =====================================================================================
 * 
 *       Filename:  hyper_ball.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/11/2007 05:45:21 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef HYPER_BALL_H_
#define HYPER_BALL_H_
#include <new>
#include <math.h>
#include <string>
#include "base/basic_types.h"
#include "computations_counter.h"


template<typename PRECISION, typename METRIC, 
	       typename ALLOCATOR, bool diagnostic>
class HyperBall {
 public:
  typedef typename ALLOCATOR::template ArrayPtr<PRECISION> Array_t;
	typedef HyperBall<PRECISION, METRIC, ALLOCATOR, diagnostic> HyperBall_t;
	typedef METRIC Metric_t;
	struct PivotData {
		PivotData(){
		}
		PivotData(int32 dimension) : dimension_(dimension), radious_(0),
        center_(dimension_), left_(dimension), right_(dimension) {
		  for(int32 i=0; i<dimension_; i++) {
			  center_[i]=0;
				left_[i]=0;
				right_[i]=0;
			}
		}
  	int32 dimension_;
  	PRECISION radious_; 
		Array_t center_;
	  Array_t	left_;
    Array_t right_;
  };
  HyperBall(PivotData &pivot_data);
  ~HyperBall() {};
  static void *operator new(size_t size);
  static void  operator delete(void *p);
  HyperBall_t &operator=(const HyperBall_t &);
  template<typename POINTTYPE> 
  bool IsWithin(POINTTYPE point, int32 dimension, PRECISION range, 
                     ComputationsCounter<diagnostic> &comp);
  PRECISION IsWithin(HyperBall_t &hr,
                     int32 dimension, 
                     PRECISION range,
                     ComputationsCounter<diagnostic> &comp);
  template<typename POINTTYPE> 
  bool CrossesBoundaries(POINTTYPE point, int32 dimension, PRECISION range, 
                         ComputationsCounter<diagnostic> &comp);
  template<typename POINTTYPE1, typename POINTTYPE2>
	static PRECISION Distance(POINTTYPE1 point1, POINTTYPE2 point2, int32 dimension);
	static PRECISION Distance(HyperBall_t &hr1,
                            HyperBall_t &hr2,
                            int32 dimension,
                            ComputationsCounter<diagnostic> &comp);
  static PRECISION Distance(HyperBall_t &hr1,
                            HyperBall_t &hr2,
                            PRECISION threshold_distance,
                            int32 dimension,
                            ComputationsCounter<diagnostic> &comp);                            
  template<typename POINTTYPE, typename NODETYPE>                   	
  pair<typename ALLOCATOR::template Ptr<NODETYPE>, 
		   typename ALLOCATOR::template Ptr<NODETYPE> > 
				 ClosestChild(typename ALLOCATOR::template Ptr<NODETYPE> left,
          						typename ALLOCATOR::template Ptr<NODETYPE> right,
					             POINTTYPE point, int32 dimension,
											 ComputationsCounter<diagnostic> &comp);                            
  string Print(int32 dimension);

 private:
  Array_t center_;
	// This is the radious, not the square of the radious
	PRECISION radious_;
	Array_t pivot_left_;
	Array_t pivot_right_;
};

#include "hyper_ball_impl.h"
#endif // HYPER_BALL_H_
