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
#include <limits>
#include <math.h>
#include <string>
#include "u/nvasil/loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "computations_counter.h"

using namespace std;
template<typename TYPELIST, bool diagnostic>
class HyperRectangle {
 public:
	typedef typename Loki::TL::TypeAt<TYPELIST, 0>::Result  Precision_t;  
	typedef typename Loki::TL::TypeAt<TYPELIST, 1>::Result  Allocator_t;
	typedef typename Loki::TL::TypeAt<TYPELIST, 2>::Result  Metric_t;
	typedef HyperRectangle<TYPELIST, diagnostic> HyperRectangle_t;
	typedef typename Allocator_t:: template ArrayPtr<Precision_t> ArrayPtr_t;
  typedef typename Allocator_t:: template ArrayPtr<Precision_t> Array_t;
	template<typename, bool> friend	class  HyperRectangleTest;
  
	HyperRectangle();
	void Init(int32 dimension);
  void Init(ArrayPtr_t min, ArrayPtr_t max, int32 pivot_dimension,
			      Precision_t pivot_value);
  ~HyperRectangle() {};
	void Destruct(){
	}
  static void *operator new(size_t size);
  static void  operator delete(void *p);
  HyperRectangle_t &operator=(HyperRectangle_t &);
	void Alias(const HyperRectangle_t &other);
	void Copy(const HyperRectangle_t &other, int32 dimension);
  template<typename POINTTYPE> 
  bool  IsWithin(POINTTYPE point, int32 dimension, Precision_t range, 
                  ComputationsCounter<diagnostic> &comp);
	template <typename POINTTYPE>
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
			                        int32 dimension, Precision_t **metric_matrix);
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
  pair<typename Allocator_t::template Ptr<NODETYPE>,
	     typename Allocator_t::template Ptr<NODETYPE> >
  ClosestChild(typename Allocator_t::template Ptr<NODETYPE> left,
		           typename Allocator_t::template Ptr<NODETYPE> right,
						 	 POINTTYPE point,
							 int32 dimension,
							 ComputationsCounter<diagnostic> &comp);                            
  string Print(int32 dimension);
  Array_t &get_min() {
	  return min_;
	}
	Array_t &get_max() {
	  return max_;
	}
	int32 get_pivot_dimension() {
	  return pivot_dimension_;
	}
	Precision_t get_pivot_value() {
	  return pivot_value_;
	}
	void set_pivot_dimension(int32 pivot_dimension) {
	  pivot_dimension_=pivot_dimension;
	}
	void set_pivot_value(Precision_t pivot_value) {
	  pivot_value_=pivot_value;
	}
 private:
	Array_t min_;
  Array_t max_;
  int32  pivot_dimension_; 
  Precision_t pivot_value_;                                                      
};                         

#include "hyper_rectangle_impl.h"

#endif 
