/*
 * =====================================================================================
 *
 *       Filename:  multiscale_mvu_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/07/2008 01:22:02 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "multiscale_mvu.h"

class MultiscaleMVUTest {
 public: 
  void Init() {
    engine_=new MultiscaleMVU<MaxFurthestNeighbors>();
  }
  void Destruct() {
    delete engine_;
  }
  void Test1() {
    Init();
    Matrix points;
    data::Load("/net/hg200/nvasil/dataset/swiss_roll/swiss_roll_10000.csv" , &points);
    datanode *module=fx_submodule(NULL, "/", "temp"); 
    engine_->Init(points, module);
    Destruct();
  }
  void TestAll() {
   Test1();
  }  
 private:
  MultiscaleMVU<MaxFurthestNeighbors> *engine_;
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  MultiscaleMVUTest test;
  test.TestAll();
  fx_done();
}
