/*
 * =====================================================================================
 *
 *       Filename:  hyper_rectangle_unit.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/22/2007 09:45:28 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "u/nvasil/loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "u/nvasil/mmanager/memory_manager.h"
#include "point.h"
#include "euclidean_metric.h"
#include "hyper_rectangle.h"

template<typename TYPELIST, bool diagnostic>
class HyperRectangleTest {
 friend class HyperRectangle<TYPELIST, diagnostic>;
 public:
  typedef typename Loki::TL::TypeAt<TYPELIST, 0>::Result Precision_t;
  typedef typename Loki::TL::TypeAt<TYPELIST, 1>::Result Allocator_t;
	typedef typename Loki::TL::TypeAt<TYPELIST, 2>::Result Metric_t;
	typedef HyperRectangle<TYPELIST, diagnostic> HyperRectangle_t;
  HyperRectangleTest() {
	}
	~HyperRectangleTest() {
	}
	void Init() {
		dimension_=2;
		Allocator_t::allocator_ = new Allocator_t();
		Allocator_t::allocator_->Initialize();
	  hyper_rectangle_= new HyperRectangle_t();
   	hyper_rectangle_->Init(dimension_);
		hyper_rectangle_->min_[0]=-1;
		hyper_rectangle_->min_[1]=-1;
		hyper_rectangle_->max_[0]=1;
		hyper_rectangle_->max_[1]=1;
	}
	void Destruct() {
	  delete hyper_rectangle_;
		delete Allocator_t::allocator_;
	}
	
  void AliasTest() {
		printf("Alias Test\n");
	  Init();
    HyperRectangle_t other;
    other.Alias(*hyper_rectangle_);		
	  DEBUG_ASSERT_MSG(other.min_==hyper_rectangle_->min_, 
				             "Min don't match\n");
		DEBUG_ASSERT_MSG(other.max_==hyper_rectangle_->max_,
				             "Max don't match\n");
		DEBUG_ASSERT_MSG(other.pivot_dimension_==hyper_rectangle_->pivot_dimension_,
				             "Pivot dimension doesn't match\n");
		DEBUG_ASSERT_MSG(other.pivot_value_==hyper_rectangle_->pivot_value_,
				             "Pivot value don't match\n");
	  Destruct();
	}
  void CopyTest() {
		printf("Copy Test\n");
	  Init();
    HyperRectangle_t other;
		other.Init(dimension_);
    other.Copy(*hyper_rectangle_, dimension_);	
    DEBUG_ASSERT_MSG(other.min_==hyper_rectangle_->min_,
				             "Min don't match\n");
    DEBUG_ASSERT_MSG(other.max_!=hyper_rectangle_->max_, 
				             "Max are the same \n");
		for(index_t i=0; i<dimension_; i++) {
	    DEBUG_ASSERT_MSG(other.min_[i]==hyper_rectangle_->min_[i], 
				             "Min don't match\n");
			DEBUG_ASSERT_MSG(other.max_[i]==hyper_rectangle_->max_[i],
				             "Max left doesn't match\n");
		}
	  Destruct();
	}
	void IsWithinTest() {
		Init();
	  Point<Precision_t, Allocator_t> point;
		point.Init(dimension_);
	  point[0]=-0.1;
		point[1]=0.3;
		Precision_t range=0.03;
		DEBUG_ASSERT_MSG(hyper_rectangle_->IsWithin(point, dimension_, 
					                                 range, comp_)==true, 
				             "IsWithin doesn't work\n");
		range=2;
    DEBUG_ASSERT_MSG(hyper_rectangle_->IsWithin(point, dimension_, 
					                                 range, comp_)==false, 
				             "IsWithin doesn't work\n");

    Destruct();	  
	}
	void CrossesBoundariesTest() {
	  Init();
		Point<Precision_t, Allocator_t> point;
		point.Init(dimension_);
	  point[0]=2;
		point[1]=-4;
		Precision_t range=11;
		DEBUG_ASSERT_MSG(hyper_rectangle_->CrossesBoundaries(point, dimension_, 
					                                 range, comp_)==true, 
				             "CrossesBoundary doesn't work\n");
		range=0.25;
    DEBUG_ASSERT_MSG(hyper_rectangle_->CrossesBoundaries(point, dimension_, 
					                                 range, comp_)==false, 
				             "CrossesBoundary doesn't work\n");

    Destruct();
	}
	void DistanceTest(){
    Init();
		Point<Precision_t, Allocator_t> point1;
	  Point<Precision_t, Allocator_t> point2;
	  point1.Init(dimension_);
    point2.Init(dimension_);
		point1[0]=0;
		point1[1]=1;
		point2[0]=-1;
		point2[1]=-2;
		DEBUG_ASSERT_MSG(HyperRectangle_t::Distance(point1, point2, dimension_)==10,
				             "Distance between points doesn't work\n");
		HyperRectangle_t other;
		other.Init(dimension_);
		other.min_[0]=2;
		other.min_[1]=2;
		other.max_[0]=5;
		other.max_[1]=5;
		
		DEBUG_ASSERT_MSG(HyperRectangle_t::Distance(*hyper_rectangle_, other,
				                                   dimension_, comp_)==2,
				                                   "Distance doesn't work\n");
		other.min_[0]=-0.5;
		other.min_[1]=-0.5;
		DEBUG_ASSERT_MSG(HyperRectangle_t::Distance(*hyper_rectangle_, other,
				                                   dimension_, comp_)==0,
				                                   "Distance doesn't work\n");
		Destruct();             
	}
	
 private: 
  HyperRectangle<TYPELIST, diagnostic> *hyper_rectangle_;
  int32 dimension_; 
  ComputationsCounter<diagnostic> comp_;	
};

typedef LOKI_TYPELIST_3(float32, 
		                    MemoryManager<false>, 
		                    EuclideanMetric<float32>) UserTypeParameters_t;

int main(int argc, char *argv[]) {
	HyperRectangleTest<UserTypeParameters_t, false> hyper_rectangle_test;
	hyper_rectangle_test.AliasTest();
	hyper_rectangle_test.IsWithinTest();
	hyper_rectangle_test.CrossesBoundariesTest();
	hyper_rectangle_test.DistanceTest();
}
