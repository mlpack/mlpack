/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file test_infomax_ica.h
 * @author Chip Mappus
 *
 * Unit tests for infomax ica method.
 */

#include "infomax_ica.h"
#include "fastlib/base/test.h"

class TestInfomaxICA {

 public:
  void Init(){
    // load some test data that has been verified using the matlab
    // implementation of infomax
    data::Load("../../fastlib/fake.arff",&testdata_);
    lambda_=0.001;
    b_=5;
    epsilon_=0.001;
    ica_ = new InfomaxICA(lambda_,b_,epsilon_);
  }

  void Destruct(){
    delete ica_;
  }    

  void TestCov(){
    ica_->sampleCovariance(testdata_);
  }
  
  void TestSqrtm(){
    Matrix intermediate = ica_->sampleCovariance(testdata_);
    ica_->sqrtm(intermediate);
  }

  void TestICA(){
    Matrix unmixing;
    ica_->applyICA(testdata_);
    ica_->getUnmixing(unmixing);
    ica_->displayMatrix(unmixing);
  }

  void TestAll() {
    TestSqrtm();
    TestCov();
    TestICA();
  }

 private:
  InfomaxICA *ica_;
  Matrix testdata_;
  double lambda_;
  int b_;
  double epsilon_;
};
