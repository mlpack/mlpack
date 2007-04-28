#ifndef POINT_TRAITS_
#define POINT_TRAITS_
#include "base/basic_types.h"
#include "point.h"
template<typename POINTTYPE, typename IDPRECISION>
struct PointTraits {
  static IDPRECISION GetPointId(POINTTYPE point, 
			int32 size) {
  	return point.get_id();
  }
};

template<typename POINTTYPE, typename IDPRECISION>
struct PointTraits;

template<typename POINTTYPE, typename IDPRECISION>
struct PointTraits<POINTTYPE *, IDPRECISION> {
  static IDPRECISION GetPointId(POINTTYPE* point, int32 size) {
  	return *((IDPRECISION*)((char*)point + size *sizeof(POINTTYPE)));
  }
};



#endif /*POINT_TRAITS_*/
