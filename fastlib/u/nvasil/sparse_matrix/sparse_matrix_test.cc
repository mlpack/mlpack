/*
 * =====================================================================================
 *
 *       Filename:  sparse_matrix_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/08/2007 03:50:44 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <stdio.h>
#include <limits>
#include <vector>
#include <map>
#include "fastlib/fastlib.h"
#include "u/nvasil/test/test.h"
#include "u/nvasil/sparse_matrix/sparse_matrix.h"

class SparseMatrixTest {
 public:
	SparseMatrixTest() {
	}
	~SparseMatrixTest() {
	}
	void Init() {
	  for(index_t i=0; i<num_of_cols_; i++) {
		  for(index_t j=0; j<num_of_rows_; j++) {
			  mat_[i][j] = (i+j) * ((i+j) % 2);
			}
		}
	}
	void Destruct() {
    delete smat_;		
	}
  void TestInit1() {
	  smat_ = new SparseMatrix(num_of_rows_, num_of_cols_);
    smat_->StartLoadingRows();
		std::vector<index_t> ind;
	  std::vector<double>  val;
    for(index_t i=0; i<num_of_cols_; i++) {
			ind.clear();
			val.clear();
      for(index_t j=0; j<num_of_cols_; j++) {
				if (mat_[i][j] != 0) {
			    ind.push_back(i);
				  val.push_back(mat_[i][j]);
				}
			}
		  smat_->LoadRow(i, ind, val);
		}
		for(index_t i=0; i<num_of_rows_; i++) {
		  for(index_t j=0; j<num_of_cols_; j++) {
			  TEST_DOUBLE_APPROX(smat_->get(i,j), 
						               mat_[i][j], 
						               std::numeric_limits<double>::epsilon());
			}
		}
	}
	void TestInit2() {
		std::vector<index_t> rows;
	  std::vector<index_t> cols;
	  std::vector<double>  vals;
		std::vector<index_t> nnz(num_of_rows_);
		for(index_t i=0; i<num_of_cols_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
				if (mat_[i][j] != 0) {
			    rows.push_back(i);
					rows.push_back(j);
				  vals.push_back(mat_[i][j]);
					nnz[i]++;
				}
			}
		}
		smat_->Init(rows, cols, vals, 
				        *(std::max_element(nnz.begin(), nnz.end())), num_of_rows_);
		for(index_t i=0; i<num_of_rows_; i++) {
		  for(index_t j=0; j<num_of_cols_; j++) {
			  TEST_DOUBLE_APPROX(smat_->get(i,j), mat_[i][j],
						               std::numeric_limits<double>::epsilon());
			}
		}
	}
  void TestInit3() {
		FILE *fp = fopen("temp.txt", "w");
    if (fp==NULL) {
		  FATAL("Cannot open temp.txt error %s", strerror(errno));
		}
		for(index_t i=0; i<num_of_cols_; i++) {
      for(index_t j=0; j<num_of_cols_; j++) {
				if (mat_[i][j] != 0) {
				  fprintf(fp, "%i %i %g", i, j, mat_[i][j]);
				}
			}
		}
    smat_->Init("temp.txt");
    unlink("temp.txt");
    for(index_t i=0; i<num_of_rows_; i++) {
		  for(index_t j=0; j<num_of_cols_; j++) {
			  TEST_DOUBLE_APPROX(smat_->get(i,j), 
						               mat_[i][j], 
						               std::numeric_limits<double>::epsilon());
			}
		}
	}
	void TestCopyConstructor() {
	  	}
	void TestMakeSymmetric() {
	  TestInit1();
		smat_->set(2, 3, 1.44);
		smat_->set(3, 2, 0.74);
		smat_->set(7, 8, 4.33);
		smat_->set(8, 7, 0.22);
		smat_->MakeSymmetric();
		for(index_t i=0; i<num_of_rows_; i++) {
		  for(index_t j=0; j<num_of_cols_; j++) {
			  TEST_DOUBLE_APPROX(smat_->get(i,j), 
						               smat_->get(j, i),
													 std::numeric_limits<double>::epsilon());
			}
		}
	}
  void TestAll(){
		Init();
    TestInit1();
		Destruct();
		Init();
		TestInit2();
		Destruct();
		Init();
		TestInit3();
		Destruct();
		Init();
		TestCopyConstructor();
		Destruct();
		Init();
		TestMakeSymmetric();
		Destruct();
	}
		
 private:
	SparseMatrix *smat_;
  static const index_t num_of_cols_  = 40;
	static const index_t num_of_rows_          = 40;
  double                mat_[num_of_rows_][num_of_cols_];
	std::vector<index_t>  indices_;
	std::vector<index_t>  rows_;
	Vector                values_;
};

int main() {
  SparseMatrixTest test;
	test.TestAll();
}
