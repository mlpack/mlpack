/*
 * =====================================================================================
 * 
 *       Filename:  hyper_rectangle.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/11/2007 05:42:53 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef HYPER_RECTANGLE_H_
#define HYPER_RECTANGLE_H_
#include <new>
#include <math.h>
#include <string>
#include "base/basic_types.h"
#include "computations_counter.h"

template<typename PRECISION, 
				 typename ALLOCATOR, bool diagnostic>
class HyperRectangle {
 public:
	friend class HyperRectangleTest;
  struct PivotData {
		PivotData(){
		}
  	int32 dimension_;
  	typename ALLOCATOR::template ArrayPtr<PRECISION> min_;
  	typename ALLOCATOR::template ArrayPtr<PRECISION> max_;
  	int32 pivot_dimension_;
  	PRECISION pivot_value_;  
  };
  HyperRectangle(PivotData &pivot_data);
  ~HyperRectangle() {};
  static void *operator new(size_t size);
  static void  operator delete(void *p);
  HyperRectangle &operator=(HyperRectangle &);
  template<typename POINTTYPE> 
  bool  IsWithin(POINTTYPE point, int32 dimension, PRECISION range, 
                     ComputationsCounter<diagnostic> &comp);
  PRECISION IsWithin(HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr,
                     int32 dimension, 
                     PRECISION range,
                     ComputationsCounter<diagnostic> &comp);
  template<typename POINTTYPE> 
  bool CrossesBoundaries(POINTTYPE point, int32 dimension, PRECISION range, 
                         ComputationsCounter<diagnostic> &comp);
  template<typename POINTTYPE1, typename POINTTYPE2>
	static PRECISION Distance(POINTTYPE1 point1, POINTTYPE2 point2, int32 dimension);
	static PRECISION Distance(HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr1,
                            HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr2,
                            int32 dimension,
                            ComputationsCounter<diagnostic> &comp);
  static PRECISION Distance(HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr1,
                            HyperRectangle<PRECISION, ALLOCATOR, diagnostic> &hr2,
                            PRECISION threshold_distance,
                            int32 dimension,
                            ComputationsCounter<diagnostic> &comp);                            
  template<typename POINTTYPE, typename NODETYPE>                   	
  pair<typename ALLOCATOR::template Ptr<NODETYPE>,
	     typename ALLOCATOR::template Ptr<NODETYPE> >
  ClosestChild(typename ALLOCATOR::template Ptr<NODETYPE> left,
		           typename ALLOCATOR::template Ptr<NODETYPE> right,
						 	 POINTTYPE point,
							 int32 dimension,
							 ComputationsCounter<diagnostic> &comp);                            
  string Print(int32 dimension);
 private:
	typename ALLOCATOR::template ArrayPtr<PRECISION> min_;
  typename ALLOCATOR::template ArrayPtr<PRECISION> max_;
  int32  pivot_dimension_; 
  PRECISION pivot_value_;                                                      
};                         

#include "hyper_rectangle_impl.h"


#endif // HYPER_RECTANGLE_
