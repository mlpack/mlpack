/*
 * =====================================================================================
 *
 *       Filename:  bounding_box_unit.cc
 *
 *    Description: Unit tests for bounding box  
 *
 *        Version:  1.0
 *        Created:  01/30/2007 11:09:21 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab
 *
 * =====================================================================================
 */
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <limits>
#include "boost/scoped_ptr.hpp"
#include "base/basic_types.h"
#include "smart_memory/src/memory_manager.h" 
#include "kdnode.h"

class HyperRectangleTest: public CppUnit::TestFixture  {
  CPPUNIT_TEST_SUITE(HyperRectangleTest);
  CPPUNIT_TEST(TestAll);
  CPPUNIT_TEST_SUITE_END();
 public:
  static const int32 kDimension_=10;
  static const float32 kPointIn_[kDimension_];
  static const float32 kPointOut_[kDimension_];
	typedef MemoryManager<false> Allocator;

  void setUp() {
		MemoryManager<false>::allocator=new MemoryManager<false>(); 
    MemoryManager<false>::allocator->Initialize();
  	pivot_.dimension_=kDimension_;
  	pivot_.min_.Reset(Allocator::allocator->Alloc<float32>(kDimension_));
  	pivot_.max_.Reset(Allocator::allocator->Alloc<float32>(kDimension_));
  	pivot_.pivot_dimension_=1;
  }
  
  void tearDown() {
    if(remove("temp_mem")<0) {
	    fprintf(stderr,"Unable to delete file, error %s\n", strerror(errno));
	  }
		delete MemoryManager<false>::allocator;
		delete hr_;
  }
  void TestAll() {
   	hr_= new HyperRectangle<float32, MemoryManager<false>, true>(pivot_);
   	CPPUNIT_ASSERT(!hr_->min_.IsNULL());
   	CPPUNIT_ASSERT(!hr_->max_.IsNULL());
   	for(int32 i=0; i<kDimension_; i++) {
   	  hr_->min_[i] = -1;
   	  hr_->max_[i] =  1;
   	}
   	for(int32 i=0; i<kDimension_; i++) {
   	  CPPUNIT_ASSERT(hr_->min_[i] == -1);
   	  CPPUNIT_ASSERT(hr_->max_[i] ==  1);
   	}
   	CPPUNIT_ASSERT(hr_->IsWithin(kPointIn_, kDimension_, 0.9,
   	               comp_)==0);
    CPPUNIT_ASSERT(hr_->IsWithin(kPointIn_, kDimension_, 4,
   	               comp_) == 1); 
   	                  	                
   	CPPUNIT_ASSERT(hr_->IsWithin(kPointOut_, kDimension_, 1.5,
   	               comp_)==-1); 
    CPPUNIT_ASSERT(hr_->CrossesBoundaries(kPointOut_, kDimension_, 0.24,
   	               comp_)==false); 
    CPPUNIT_ASSERT(hr_->CrossesBoundaries(kPointOut_, kDimension_, 2.6,
   	               comp_)==true);
    HyperRectangle<float32, MemoryManager<false>, true> hr1 = *hr_;
    for(int32 i=0; i<kDimension_; i++) {
   	  CPPUNIT_ASSERT(hr1.min_[i] == -1);
   	  CPPUNIT_ASSERT(hr1.max_[i] ==  1);
    } 	
		pivot1_.dimension_=kDimension_;
  	pivot1_.min_.Reset(Allocator::allocator->Alloc<float32>(kDimension_));
  	pivot1_.max_.Reset(Allocator::allocator->Alloc<float32>(kDimension_));
  	pivot1_.pivot_dimension_=1;

    HyperRectangle<float32, MemoryManager<false>, true>  hr2(pivot1_);
    for(int32 i=0; i<kDimension_; i++) {
   	  hr2.min_[i] = -0.5;
   	  hr2.max_[i] =  0.5;
   	}
   	CPPUNIT_ASSERT(hr_->IsWithin(hr2, kDimension_, 0.26,
   	                            comp_) > 0);
		CPPUNIT_ASSERT(hr_->IsWithin(hr2, kDimension_, 0.24,
   	                            comp_) == 0);
    for(int32 i=0; i<kDimension_; i++) {
   	  hr2.min_[i] = -1.5;
   	  hr2.max_[i] =  0.5;
   	}   	                           
   	CPPUNIT_ASSERT(hr_->IsWithin(hr2, kDimension_, 0.26,
   	                             comp_) == -1.0);
                      	               
  }
 private:
  HyperRectangle<float32, MemoryManager<false>, true>::PivotData pivot_; 
 	HyperRectangle<float32, MemoryManager<false>, true>::PivotData pivot1_; 
	HyperRectangle<float32, MemoryManager<false>, true>  *hr_;
	ComputationsCounter<true> comp_;
};

const float32 HyperRectangleTest::kPointIn_[
    HyperRectangleTest::kDimension_] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const float32 HyperRectangleTest::kPointOut_ [
    HyperRectangleTest::kDimension_] =
    {1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5};


CPPUNIT_TEST_SUITE_REGISTRATION(HyperRectangleTest);
int main( int argc, char **argv)
{
  // Create the event manager and test controller
  CPPUNIT_NS::TestResult controller;

  // Add a listener that colllects test result
  CPPUNIT_NS::TestResultCollector result;
  controller.addListener( &result );        

  // Add a listener that print dots as test run.
  CPPUNIT_NS::BriefTestProgressListener progress;
  controller.addListener( &progress );      

  // Add the top suite to the test runner
  CPPUNIT_NS::TestRunner runner;
  runner.addTest( CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest() );
  runner.run( controller );

  // Print test in a compiler compatible format.
  CPPUNIT_NS::CompilerOutputter outputter( &result, CPPUNIT_NS::stdCOut() );
  outputter.write(); 

  return result.wasSuccessful() ? 0 : 1;
}










