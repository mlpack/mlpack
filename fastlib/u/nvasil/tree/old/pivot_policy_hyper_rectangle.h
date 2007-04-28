/*
 * =====================================================================================
 * 
 *       Filename:  pivot_policy_hyper_rectangle.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/11/2007 05:37:05 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef PIVOT_POLICY_HYPER_RECTANGLE_H_
#define PIVOT_POLICY_HYPER_RECTANGLE_H_

#include "kdnode.h"
// Specialize for Kdnodes
template<typename PRECISION, 
         typename IDPRECISION, 
         typename ALLOCATOR,
				 bool     diagnostic>
class PivotPolicy<PRECISION, IDPRECISION, ALLOCATOR, 
                  KdNode<PRECISION, IDPRECISION, 
									       ALLOCATOR, diagnostic>, diagnostic> {
 public:                  
  friend class HyperRectanglePivotTest;
	typedef  KdNode<PRECISION, IDPRECISION, ALLOCATOR, diagnostic> Node_t;
	typedef  typename Node_t::Pivot_t Pivot_t;
	typedef  typename Node_t::BoundingBox_t Box_t; 
 	static pair<Pivot_t*, Pivot_t*> Pivot(DataReader<PRECISION, IDPRECISION> *data,
			                                  Pivot_t *pivot);
  static Pivot_t* PivotParent(DataReader<PRECISION, IDPRECISION> *data, 
                              IDPRECISION num_of_points, int32 dimension);
  private:
   static void  FindPivotDimensionValue(typename Box_t::PivotData &pv);               
   static void  UpdateHyperRectangle(PRECISION *point, 
			                               typename Box_t::PivotData &pv);
};                   

#include "pivot_policy_hyper_rectangle_impl.h"

#endif // PIVOT_POLICY_HYPER_RECTANGLE_H_
