/*
 * =====================================================================================
 * 
 *       Filename:  metric.h
 * 
 *    Description: Definition of different metrics 
 * 
 *        Version:  1.0
 *        Created:  02/11/2007 04:52:54 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#include "fastlib/fastlib.h"
template<typename PRECISION>
class EuclideanMetric {
 public:
	typedef PRECISION Precision_t; 
	template<typename POINTTYPE1, typename POINTTYPE2>
	static inline Precision_t Distance(POINTTYPE1 p1, 
			                               POINTTYPE2 p2, int32 dimension) {
	  // we need to do some type checking for the POINTTYPE
		Precision_t dist=0;
		for(int32 i=0; i<dimension; i++) {
		  dist+=(p1[i]-p2[i])*(p1[i]-p2[i]);
		}
		return dist;
	}

	template<typename POINTTYPE1, typename POINTTYPE2, typename POINTTYPE3>
  static inline void Addition(POINTTYPE1 &result, 
			                        POINTTYPE2 p1, 
											        POINTTYPE3 p2, 
											        int32 dimension) {
		for(int32 i=0; i<dimension; i++) {
		  result[i] = p1[i]+p2[i];
		}
	}

	template<typename POINTTYPE1, typename POINTTYPE2>
	static inline void Scale(POINTTYPE1 &result, POINTTYPE2 point,
		                       Precision_t scale_factor, int32 dimension) {
	  for(int32 i=0; i<dimension; i++) {
		  result[i]= point[i] * scale_factor;
		}
	}

};
