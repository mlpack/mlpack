#ifndef POINT_H_
#define POINT_H_
#include <new>
#include <string>
#include "u/nvasil/loki/NullType.h"
#include "fastlib/fastlib.h"

template<typename PRECISION, typename ALLOCATOR>
class Point {
 public:
	typedef PRECISION Precision_t;
	typedef ALLOCATOR Allocator_t;
	typedef Point<Precision_t, Allocator_t> Point_t;
  Point() {
	  this->p_.Reset(NULL);
		this->id_=0;
	};
  void *operator new(size_t size) {
    return Allocator_t::allocator->AllignedAlloc(size);
  }   	
  void operator delete(void *p) {
  }

	void Init(int32 dim) {
	  p_.Reset(Allocator_t::template calloc<Precision_t>(dim, 0));
	}
	Precision_t &operator[](index_t i) {
	  return p_[i];
	}
  void Alias(const Point_t &point) {
		this->p_ = point.p_;
		this->id_ = point.id_;
  }

	void Alias(Precision_t *ptr, index_t point_id) {
	  this->p_.Reset(ptr);
		this->id_=point_id;
	}
	template<typename POINTTYPE>
  void Copy(POINTTYPE point, int32 dimension) {
  	for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
  }
	
	void Copy(Point_t point, int32 dimension) {
	  for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
    id_=point.id_;
	}
	void SetNULL() {
	  p_=NULL;
	}
  void Print(int32 dimension) {
  	printf("values= ");
  	for(int32 i=0; i< dimension; i++) {
  	  printf("%lg ", (double)this->operator[](i));
  	}
  	printf("; id="LI, id_);
  } 
  index_t get_id()	{
    return id_;
  }
  void set_id(index_t id) {
    id_ = id;
  }

 private:
	typename Allocator_t::template ArrayPtr<Precision_t> p_;
  index_t id_;
};

// use this for points where you don't care about the allocator
// this class stands only as alias to other types
template<typename PRECISION>
class Point<PRECISION, Loki::NullType> {
 public:
	typedef PRECISION Precision_t;
	typedef Point<Precision_t, Loki::NullType> Point_t;
  Point() {
	  this->p_=NULL;
		this->id_=0;
	};
	Precision_t &operator[](index_t i) {
	  return p_[i];
	}
  void Alias(const Point_t &point) {
		this->p_ = point.p_;
		this->id_ = point.id_;
  }
	 void Alias(Precision_t *ptr, index_t id) {
	  p_=ptr;
		id_=id;
	}
	template<typename POINTTYPE>
  void Copy(POINTTYPE point, int32 dimension) {
  	for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
  }
	void Copy(Point_t point, int32 dimension) {
	  for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
    id_=point.id_;
	}
	void SetNULL() {
	  p_=NULL;
	}
  void Print(int32 dimension) {
  	printf("values= ");
  	for(int32 i=0; i< dimension; i++) {
  	  printf("%lg ", (double)this->operator[](i));
  	}
  	printf("; id="LI, id_);
  } 
  index_t get_id()	{
    return id_;
  }
  void set_id(index_t id) {
    id_ = id;
  }

 private:
	Precision_t *p_;
  index_t id_;
};

#endif /*POINT_H_*/
