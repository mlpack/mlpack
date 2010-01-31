#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <sys/stat.h>
#include <float.h>
#include <math.h>
#include "base/basic_types.h"
#include "smart_memory/src/memory_manager.h"
#include "pivot_policy.h"
#include "kdnode.h"
#include "data_file.h"
#include "data_reader.h"
#include "tree.h"


class TreeTest : public CppUnit::TestFixture  {
  CPPUNIT_TEST_SUITE(TreeTest);
  CPPUNIT_TEST(TestBuildDepthFirst);
  CPPUNIT_TEST(TestBuildBreadthFirst);
  CPPUNIT_TEST(TestNearestNeighbor);
  CPPUNIT_TEST(TestKNearestNeighbors);
  CPPUNIT_TEST(TestAllNearestNeighbors);
  CPPUNIT_TEST_SUITE_END();
 public:
  static const char  *kInputFile_; 
  static const char  *kAllNnFile1_;
  static const char  *kAllNnFile2_;
   
  void setUp() { 
    MemoryManager<false>::allocator = new MemoryManager<false>();
		MemoryManager<false>::allocator->set_capacity(1000 * 65536);
		MemoryManager<false>::allocator->Initialize();
    void *temp;
  	OpenDataFile(string(kInputFile_), &dimension_, 
                 &num_of_points_, 
                 &temp,
                 &map_size_);
    data_ = new DataReader<Precision_t, IdPrecision_t>(temp, dimension_);
    tree_ = new Tree_t(data_, dimension_, num_of_points_);

  }
  void tearDown() {
    delete MemoryManager<false>::allocator;
		if(remove("temp_mem")<0) {
	    fprintf(stderr,"Unable to delete file, error %s\n", strerror(errno));
	  }
		CloseDataFile(data_->get_source(), map_size_);
		delete data_;
		delete tree_;
  }
 private:
  typedef float32 Precision_t;
	typedef uint64  IdPrecision_t;
	typedef MemoryManager<false> Allocator_t;
  typedef Tree<Precision_t, IdPrecision_t, Allocator_t, false,
				       KdNode> Tree_t;
	typedef Tree_t::Node_t::Result Result_t;
	typedef Point<Precision_t, IdPrecision_t, Allocator_t> Point_t;
	DataReader<Precision_t, IdPrecision_t> *data_;
  Tree_t *tree_;	
	int32 dimension_;
	IdPrecision_t num_of_points_;
  IdPrecision_t map_size_;
 protected:

  void TestBuildBreadthFirst() {
  	printf("TestBuildBreadthFirst...\n");
  	tree_->SerialBuildBreadthFirst();
  	//tree_->Print();
  	string stats = tree_->Statistics();
    printf("%s\n", stats.c_str());
  }
	void TestBuildDepthFirst() {
  	printf("TestBuildDepthFirst...\n");
    tree_->SerialBuildDepthFirst();
    //tree_->Print();
    string stats = tree_->Statistics();
    printf("%s\n", stats.c_str());
    
  }
  
  void TestNearestNeighbor() {
  	printf("TestNearestNeighbor...\n");
  	tree_->SerialBuildDepthFirst();
  	// tree_->Print();
  	printf("%s\n", tree_->Statistics().c_str());
  	Precision_t *test_point;
    for(IdPrecision_t i=0; i<num_of_points_; i++) {
      test_point = data_->At(i);
      Precision_t mindist = numeric_limits<float32>::max();
      IdPrecision_t minid = 0;
      for(IdPrecision_t j = 0; j<num_of_points_; j++) {
        if (j==i) {
      	  continue;
        }
        Precision_t *point = data_->At(j);
        Precision_t  dist=0;
        dist = Tree_t::BoundingBox_t::Distance(test_point, point, dimension_);
        if (dist < mindist) {
      	  mindist = dist;
      	  minid = data_->GetId(j);
        }
      }
      Point_t nearest;
      Precision_t tree_mindist;
      tree_->NearestNeighbor(test_point, &nearest, &tree_mindist, 1);

      CPPUNIT_ASSERT_DOUBLES_EQUAL(tree_mindist, mindist, 
                                   numeric_limits<float32>::epsilon());
    
      if (minid!=nearest.get_id()) {
      	printf("%llu %llu %llu\n", (unsigned long long) data_->GetId(i),
						                       (unsigned long long) nearest.get_id(),
																	 (unsigned long long) minid);
      	printf("%f %f\n", tree_mindist, mindist);
      }
      CPPUNIT_ASSERT_EQUAL(minid, nearest.get_id());
    }
    
  }
  
  void TestKNearestNeighbors() {
  	printf("TestKNearestNeighbor...\n");
  	tree_->SerialBuildDepthFirst();
  	// tree_->Print();
  	printf("%s\n", tree_->Statistics().c_str());
  	Precision_t *test_point;
  	int32 k=15;
  	multimap<Precision_t, IdPrecision_t> neighbors;
    for(IdPrecision_t i=0; i<0.1*num_of_points_; i++) {
      test_point = data_->At(i);
      neighbors.clear();
      for(IdPrecision_t j = 0; j<num_of_points_; j++) {
        if (j==i) {
      	  continue;
        }
        Precision_t *point = data_->At(j);
        Precision_t dist=0;
        dist = Tree_t::BoundingBox_t::Distance(test_point, point, dimension_);
        neighbors.insert(make_pair(dist, data_->GetId(j))); 
      }
      vector<pair<Precision_t, Point_t> > nearest;
      nearest.clear();
      Precision_t tree_mindist;
      tree_->NearestNeighbor(test_point, &nearest, &tree_mindist, k);
      multimap<Precision_t, IdPrecision_t>::iterator it = neighbors.begin();
      for(int32 m=0; m<k; m++) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(it->first, nearest[m].first, 
                                   numeric_limits<float32>::epsilon());
        CPPUNIT_ASSERT_EQUAL(it->second, nearest[m].second.get_id());
        it++;
      }
    }
  }
  
  void TestAllNearestNeighbors() {
  	printf("TestAllNearestNeighbors...\n");
  	tree_->SerialBuildDepthFirst();
    tree_->InitAllKNearestNeighborOutput(kAllNnFile1_,1);
  	tree_->AllNearestNeighbors(tree_->get_parent(), tree_->get_parent(), 1);
	  tree_->CloseAllKNearestNeighborOutput(1);
  	FILE *fp=fopen(kAllNnFile2_, "w");
  	
  	Precision_t *test_point;
  	for(IdPrecision_t i=0; i<num_of_points_; i++) {
      test_point = data_->At(i);
      Precision_t mindist = numeric_limits<Precision_t>::max();
      IdPrecision_t minid = 0;
      for(IdPrecision_t j = 0; j<num_of_points_; j++) {
        if (j==i) {
      	  continue;
        }
        Precision_t *point = data_->At(j);
        Precision_t dist=0;
        dist = Tree_t::BoundingBox_t::Distance(test_point, point, dimension_);
        if (dist < mindist) {
      	  mindist = dist;
      	  minid = data_->GetId(j);
        }
      }
			Result_t temp;
			temp.point_id_=data_->GetId(i);
			temp.distance_=mindist;
			temp.nearest_.set_id(minid);
      fwrite(&temp, sizeof(Result_t), 1, fp);
    }
    fclose(fp);
    // compare files
    struct stat info1;
    struct stat info2;
    CPPUNIT_ASSERT(stat(kAllNnFile1_, &info1) == 0);
    CPPUNIT_ASSERT(stat(kAllNnFile2_, &info2) == 0);
    CPPUNIT_ASSERT(info1.st_size == info2.st_size);
    FILE *fp1 = fopen(kAllNnFile1_, "r");
    FILE *fp2 = fopen(kAllNnFile2_, "r");
		uint32 size=info1.st_size/sizeof(Result_t);
		CPPUNIT_ASSERT(size==num_of_points_);
    Result_t  allnn1[size]; 
    Result_t  allnn2[size]; 
    fread(allnn1, sizeof(Result_t), size, fp1);
    fread(allnn2, sizeof(Result_t), size, fp2);
    fclose(fp1);
		fclose(fp2);
		
		sort(allnn1, allnn1+size);
		sort(allnn2, allnn2+size);
		for(uint32 i=0; i<size; i++) {
      CPPUNIT_ASSERT(allnn1[i].point_id_ == allnn2[i].point_id_);
      CPPUNIT_ASSERT(allnn1[i].nearest_.get_id() == 
					           allnn2[i].nearest_.get_id() );
		}
	}

};

 const char *TreeTest::kInputFile_ = "./data/data";
 const char *TreeTest::kAllNnFile1_ = "./data/allnn1.txt";
 const char *TreeTest::kAllNnFile2_ = "./data/allnn2.txt";

CPPUNIT_TEST_SUITE_REGISTRATION(TreeTest);
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
