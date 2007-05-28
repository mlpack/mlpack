/*
 * =====================================================================================
 *
 *       Filename:  binary_dataset.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/24/2007 07:36:16 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <string>
#include <unistd.h>
#include "fastlib/fastlib.h"
#include "u/nvasil/test/test.h"
#include "u/nvasil/dataset/binary_dataset.h"

using namespace std;
template<typename PRECISION>
class BinaryDatasetTest {
 public:
	typedef PRECISION Precision_t;
  void Init() {
		data_file_="data";
		dimension_=10;
		num_of_points_=8000;
		dataset_.Init(data_file_, 
			            num_of_points_,
								 	dimension_);
	  for(uint64 i=0; i<num_of_points_; i++) {
			for(int32 j=0; j<dimension_; j++) {
		    dataset_.At(i,j)=1.0*16.3;
		  }
			dataset_.set_id(i,i);
		}
	}
	void Destruct() {
	  dataset_.Destruct();
		unlink(data_file_.c_str());
    unlink(data_file_.append(".ind").c_str());
    data_file_="";		
	}
  void FillDataset() {
	 	for(uint64 i=0; i<num_of_points_; i++) {
			for(int32 j=0; j<dimension_; j++) {
		    TEST_ASSERT(dataset_.At(i,j)==1.0*16.3)	;
		  }
		}
		dataset_.Destruct();
		dataset_.Init(data_file_);
    for(uint64 i=0; i<num_of_points_; i++) {
			for(int32 j=0; j<dimension_; j++) {
		    TEST_ASSERT(dataset_.At(i,j)==1.0*16.3)	;
		  }
		}
  }	

  void Swap() {
		index_t i=140;
		index_t j=1456;
		Precision_t point_i[dimension_];
		memcpy(point_i, dataset_.At(i), dimension_*sizeof(Precision_t));
		uint64 id_i=dataset_.get_id(i);
		Precision_t point_j[dimension_];
		uint64 id_j=dataset_.get_id(j);
		memcpy(point_j, dataset_.At(j), dimension_*sizeof(Precision_t));
	  dataset_.Swap(i,j);
		TEST_ASSERT(memcmp(dataset_.At(i), point_j,
				       	dimension_*sizeof(Precision_t))==0);
	  TEST_ASSERT(memcmp(dataset_.At(j), point_i, 
					           dimension_*sizeof(Precision_t))==0);
	  TEST_ASSERT(dataset_.get_id(i)==id_j);
    TEST_ASSERT(dataset_.get_id(j)==id_i);

	}	
	void TestAll() {
    Init();
	  FillDataset();
		Destruct();
		Init();
    Swap();
    Destruct();		
	}
 private:
	string data_file_;
	uint64 num_of_points_;
	int32 dimension_;
  BinaryDataset<Precision_t> dataset_;	
};

int main(int argc, char *argv[]) {
  BinaryDatasetTest<double> binary_dataset_tester;
	binary_dataset_tester.TestAll();
}
