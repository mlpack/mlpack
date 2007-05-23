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
  inline Point() {
	  this->p_.Reset(Allocator_t::NullValue);
		this->id_=0;
	};
  inline void Lock() {
	  this->p_.Lock();
	}
	inline void Unlock() {
	  this->p_.Unlock();
	}
  inline Point(const Point_t &other) {
	  Alias(other);
	}
  /*void *operator new(size_t size) {
    return Allocator_t::allocator->AllignedAlloc(size);
  }   	
  void operator delete(void *p) {
  }
  */
	inline void Init(int32 dim) {
	  p_.Reset(Allocator_t::template calloc<Precision_t>(dim, 0));
	}
	Precision_t &operator[](index_t i) {
	  return p_[i];
	}
  inline void Alias(const Point_t &point) {
		this->p_ = point.p_;
		this->id_ = point.id_;
  }

	inline void Alias(Precision_t *ptr, index_t point_id) {
	  this->p_.Reset(ptr);
		this->id_=point_id;
	}
	template<typename POINTTYPE>
  inline void Copy(POINTTYPE point, int32 dimension) {
  	for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
  }
	
	inline void Copy(Point_t point, int32 dimension) {
	  for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
    id_=point.id_;
	}
	inline void SetNULL() {
	  p_=NULL;
	}
  void Print(int32 dimension) {
  	printf("values= ");
  	for(int32 i=0; i< dimension; i++) {
  	  printf("%lg ", (double)this->operator[](i));
  	}
  	printf("; id="LI, id_);
  } 
  inline index_t get_id()	{
    return id_;
  }
  inline void set_id(index_t id) {
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
  inline Point() {
	  this->p_=NULL;
		this->id_=0;
	};

	inline Point(const Point_t &other) {
	  Alias(other);
	}
	inline Precision_t &operator[](index_t i) {
	  return p_[i];
	}
  inline void Alias(const Point_t &point) {
		this->p_ = point.p_;
		this->id_ = point.id_;
  }
  inline void Alias(Precision_t *ptr, index_t id) {
	  p_=ptr;
		id_=id;
	}
	template<typename POINTTYPE>
  inline void Copy(POINTTYPE point, int32 dimension) {
  	for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
  }
	inline void Copy(Point_t point, int32 dimension) {
	  for(int32 i=0; i<dimension; i++) {
  	  p_[i] = point[i];
  	}
    id_=point.id_;
	}
	inline void SetNULL() {
	  p_=NULL;
	}
  inline void Print(int32 dimension) {
  	printf("values= ");
  	for(int32 i=0; i< dimension; i++) {
  	  printf("%lg ", (double)this->operator[](i));
  	}
  	printf("; id="LI, id_);
  } 
  inline index_t get_id()	{
    return id_;
  }
  inline void set_id(index_t id) {
    id_ = id;
  }

 private:
	Precision_t *p_;
  index_t id_;
};

template<class PRECISION>
class CompletePoint {
 public:
  typedef PRECISION Precision_t;
	typedef CompletePoint<Precision_t> CompletePoint_t;
	CompletePoint() {}
	~CompletePoint(){}
	CompletePoint(const CompletePoint_t &other) {
	  this->p_=other.p_;
		this->id_=other.id_;
		this->dimension_=other.dimension_;
	}
  CompletePoint_t &operator=(const CompletePoint_t &other) {
	  DEBUG_ASSERT_MSG(this->dimension_==other.dimension_, 
				             "Points have different dimensions "LI"!="LI"", 
										  this->dimension_, other.dimension_);
		memcpy(this->p_, other.p_, dimension_*sizeof(Precision_t));
		this->id_=other.id_;
	}
	void Alias(Precision_t *ptr, index_t id, int32 dimension) {
	  p_=ptr;
		id_=id;
		dimension_=dimension;
	}
 private:
	Precision_t *p_;
	index_t id_;
	int32 dimension_;
};

#endif /*POINT_H_*/
