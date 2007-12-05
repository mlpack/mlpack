/*
 * =====================================================================================
 *
 *       Filename:  sparse_vector_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/04/2007 08:20:52 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "fastlib/fastlib.h"
#include "u/nvasil/sparse_matrix/sparse_vector.h"

class SparseVectorTest {
 public:
	void Init() {
    dim_  = 100;
	  ind1_	= new index_t[10];
		ind2_ = new index_t[5];
		val1_ = new double[10];
		val2_ = new double[5];
		dim_  = 40;
		for(index_t i=0; i<10; i++) {
		  ind1_[i] = 2*i;
			val1_[i] = 3*i;
		}
		
		for(index_t i=0; i<5; i++) {
		  ind1_[i] = 2*i;
			val1_[i] = 4*i;
		}
    v1_.Init(ind1_, val1_, 10, dim_);
    v2_.Init(ind2_, val2_,  5, dim_);
    
		
	}
  
	void Destruct() {
	  v1_.Destruct();
		v2_.Destruct();
		delete []ind1_;
		delete []ind2_;
	}

	void TestInit1();
	void TestInit2();
	void TestInit3();
  void TestCopyConstructor();
  void TestGet();
	void TestSet();
	void TestAdd();
	void TestSubtract();
	void TestPointProduct();
	void TestDotProduct();
	void TestDistance();
	
	void TestAll() {
		TestInit1();
	  TestInit2();
	  TestInit3();
    TestCopyConstructor();
    TestGet();
	  TestSet();
	  TestAdd();
	  TestSubtract();
	  TestPointProduct();
	  TestDotProduct();
	  TestDistance();
	}

 private:
  SparseVector v1_;
	SparseVector v2_;
	SparseVector v3_;
	index_t   *ind1_;
	index_t   *ind2_;
  double     *val1;	
	double     *val2;
	index_t     dim_;
};

int main() {
  SparseVectorTest test;
	test.TestAll();
}
