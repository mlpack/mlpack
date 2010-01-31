#ifndef PIVO_POLICY_H_
#define PIVO_POLICY_H_
#include "base/basic_types.h"
#include "data_reader.h"

// The general case is the ball(metric) node, just because it is more generall
// it accomodates more metrics
template<typename PRECISION, 
         typename IDPRECISION,
         typename ALLOCATOR,
         class NODE,
				 bool diagnostic>
class PivotPolicy {
 public:
	 friend class EuclideanBallPivotTest;
	typedef typename ALLOCATOR::template ArrayPtr<PRECISION> Array_t;
	typedef typename NODE::Pivot_t Pivot_t;
	typedef typename NODE::BoundingBox_t Box_t;
	typedef typename NODE::Metric_t Metric_t;
  static pair<Pivot_t*, Pivot_t*>	Pivot(DataReader<PRECISION, IDPRECISION> *data,
			                                  Pivot_t *pivot);
  static Pivot_t* PivotParent(DataReader<PRECISION, IDPRECISION> *data,
			                        IDPRECISION num_of_data,
															int32 dimension);
 
 private:
  static	void FindPivotPoints(DataReader<PRECISION, IDPRECISION> *data,
		                   typename Box_t::PivotData &pivot,
											 IDPRECISION start,
											 IDPRECISION num_of_points);
	template<typename POINTTYPE>	
  static void UpdateHyperBall(POINTTYPE point, 
		                   typename Box_t::PivotData &pivot);

  static void NormalizeHyperBall(typename Box_t::PivotData &pivot, 
		                      IDPRECISION num_of_points);
	template<typename POINTTYPE>	
  static bool IsInLeftPivot(PRECISION *point, POINTTYPE left, POINTTYPE right, 
			               int32 dimension);

  template<typename POINTTYPE>
  static pair<Array_t, PRECISION> FindFurthestPoint(POINTTYPE point,
		                       DataReader<PRECISION, IDPRECISION> *data, 
   		                     IDPRECISION start, 
		                       IDPRECISION num_of_points, 
													 int32 dimension);
};
#include "pivot_policy_impl.h"

#include "pivot_policy_hyper_rectangle.h"

#endif /*PIVOT_POLICY_H_*/
