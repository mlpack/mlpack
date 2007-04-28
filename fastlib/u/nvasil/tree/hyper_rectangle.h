/*
 * =====================================================================================
 * 
 *       Filename:  hyper_rectangle.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/17/2007 04:38:03 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef U_NVASIL_HYPER_RECTANGLE_H_
#define U_NVASIL_HYPER_RECTANGLE_H_

#include <new>
#include <math.h>
#include <string>
#include "loki/Typelist.h"
#include "fastib/fastlib.h"
#include "computations_counter.h"

template<typename TYPELIST, bool diagnostic>
class HyperRectangle {
 public:
	typedef Loki::TL::TypeAt<TYPELIST, 0>::Result  Precision_t;  
	typedef Loki::TL::TypeAt<TYPELIST, 1>::Result  Allocator_t;
	typedef Loki::TL::TypeAt<TYPELIST, 2>::Result  Metric_t;
	typedef HyperRectangle<TYPELIST, dignostic> HyperRectangle_t;
	typedef Allocator_t:: template ArrayPtr<Precision_t> ArrayPtr_t;
	friend class HyperRectangleTest;
  
	HyperRectangle();
	void Init(int32 dimension);
  void Init(ArrayPtr_t min, ArrayPtr_t max, int32 pivot_dimension,
			      Precision_t pivot_value);
  ~HyperRectangle() {};
  static void *operator new(size_t size);
  static void  operator delete(void *p);
  HyperRectangle_t &operator=(HyperRectangle_t &);
	void Alias(const HyperRectangle_t &other);
	void Copy(const HyperRectangle_t &other, int32 dimension);
  template<typename POINTTYPE> 
  bool  IsWithin(POINTTYPE point, int32 dimension, Precision_t range, 
                     ComputationsCounter<diagnostic> &comp);
	typename<POINTTYPE>
	bool IsWithin(POINTTYPE point, int32 dimension, Precision_t metric_matrix,
			          Precision_t range, ComputationsCounter<diagnostic> &comp);
  Precision_t IsWithin(HyperRectangle_t &hr,
                       int32 dimension, 
                       Precision_t range,
                       ComputationsCounter<diagnostic> &comp);
  template<typename POINTTYPE> 
  bool CrossesBoundaries(POINTTYPE point, int32 dimension, Precision_t range, 
                         ComputationsCounter<diagnostic> &comp);
  template<typename POINTTYPE1, typename POINTTYPE2>
	static Precision_t Distance(POINTTYPE1 point1, POINTTYPE2 point2,
		                        	int32 dimension);
  template<typename POINTTYPE1, typename POINTTYPE2>
	static Precision_t Distance(POINTTYPE1 point1, POINTTYPE2 point2, 
			                        int32 dimension, Precision_t **metric_matrix)
	static Precision_t Distance(HyperRectangle_t &hr1,
                            HyperRectangle_t &hr2,
                            int32 dimension,
                            ComputationsCounter<diagnostic> &comp);
  static Precision_t Distance(HyperRectangle_t &hr1,
                              HyperRectangle_t &hr2,
                              Precision_t threshold_distance,
                              int32 dimension,
                              ComputationsCounter<diagnostic> &comp);                            
  template<typename POINTTYPE, typename NODETYPE>                   	
  pair<Allocator_t::template Ptr<NODETYPE>,
	     Allocator_t::template Ptr<NODETYPE> >
  ClosestChild(Allocator_t::template Ptr<NODETYPE> left,
		           Allocator_t::template Ptr<NODETYPE> right,
						 	 POINTTYPE point,
							 int32 dimension,
							 ComputationsCounter<diagnostic> &comp);                            
  string Print(int32 dimension);
 private:
	Allocator_t::template ArrayPtr<Precision_t> min_;
  Allocator_t::template ArrayPtr<Precision_t> max_;
  int32  pivot_dimension_; 
  Precision_t pivot_value_;                                                      
};                         


