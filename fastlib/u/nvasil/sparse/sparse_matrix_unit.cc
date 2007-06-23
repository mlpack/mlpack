/*
 * =====================================================================================
 *
 *       Filename:  sparse_matrix_unit.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06/22/2007 09:35:38 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include "fastlib/fastlib.h"
#include "base/test.h"
#include "u/nvasil/sparse/sparse_matrix.h"

namespace sparse {
template<typename T>
class MatrixTest {
 public:
	void Init(){
	  size_=100;
		non_zero_=10;
		matrix_.Init(100, 10);
		for(index_t i=0; i<size_; i++) {
		  for(index_t j=0; j<non_zero_; j++) {
			  matrix_.set(i, j, float32(i*size_+j));
			}
		}
		matrixfile_="sparse_matrix.txt";
		xfile_="xfile.txt";
		bfile_="bfile.txt";
		tolerance_=0.01;
	};
	void Destruct(){
	  matrix_.Destruct();
	}
  void TestAccess(){
	  for(index_t i=0; i<size_; i++) {
		  for(index_t j=0; j<non_zero_; j++) {
			  TEST_ASSERT(matrix_.get(i, j)==float32(i*size_+j));
			}
		}
	}
  
  void TestConjugateGradient() {
		matrix_.Destruct();
		matrix_.Init(matrixfile_);
		T *b=ReadVectorFromFile<T>(bfile_);
		T *x=ReadVectorFromFile<T>(xfile_);
	  T *xconjg=NewVector<T>(size_);	
	  ConjugateGradient(matrix_, b,  xconjg, tolerance_);
		for(index_t i=0; i<size_; i++) {
		  TEST_DOUBLE_APPROX(xconjg[i], x[i], 0.0001);
		}
	}
  void TestAll() {
	  Init();
		TestAccess();
		Destruct();
		Init();
    TestConjugateGradient();
		Destruct();
	}	
 private:
	Matrix<T>  matrix_;
	index_t size_;
	index_t non_zero_;
  string matrixfile_;
  string xfile_;
  string bfile_;	
	T tolerance_;
};
};

int main(int argc, char *argv[]) {
	sparse::MatrixTest<float32> matrix_test;
	matrix_test.TestAll();
}
