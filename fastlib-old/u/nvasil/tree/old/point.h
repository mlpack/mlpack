#ifndef POINT_H_
#define POINT_H_
#include <string>
#include "base/basic_types.h"

template<typename PRECISION, typename IDPRECISION, typename ALLOCATOR>
class Point : public ALLOCATOR::template Ptr<PRECISION> {
 public:
  Point() {
	  this->p_=NULL;
		this->id_=0;
	};
  Point<PRECISION, IDPRECISION, ALLOCATOR> &operator=(const 
			Point<PRECISION, IDPRECISION, ALLOCATOR> &point)  
  {
		this->p_ = point.p_;
		this->id_ = point.id_;
    return *this;
  }
  Point<PRECISION, IDPRECISION, ALLOCATOR> &DeepCopy(
			Point<PRECISION, IDPRECISION, ALLOCATOR> &point, int32 dimension) {
  	for(int32 i=0; i<dimension; i++) {
  		this->operator[](i) = point[i];
  	}
		id_ = point.id_;
  	return *this;
  }
  void Print(int32 dimension) {
  	printf("values= ");
  	for(int32 i=0; i< dimension; i++) {
  	  printf("%lf ", (double)this->operator[](i));
  	}
  	printf("; id=%llu\n", (unsigned long long)id_);
  } 
  IDPRECISION get_id()	{
    return id_;
  }
  void set_id(IDPRECISION id) {
    id_ = id;
  }

 private:
  IDPRECISION id_;
};

#endif /*POINT_H_*/
