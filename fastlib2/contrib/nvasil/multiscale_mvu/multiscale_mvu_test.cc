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
    data::Load("/net/hg200/nvasil/dataset/swiss_roll/swiss_roll_20000.csv" , &points);
    char buffer[128];
    sprintf(buffer, "%lg", 0.01);
    fx_set_param(NULL, "/l_bfgs/feasibility_tolerance", buffer);
    sprintf(buffer, "%lg", 50.0);
    fx_set_param(NULL, "/l_bfgs/desired_feasibility", buffer);
    sprintf(buffer, "%lg", 100.0);
    fx_set_param(NULL, "/l_bfgs/norm_grad_tolerance", buffer);
    sprintf(buffer, "%i" , 11);
    fx_set_param(NULL, "/l_bfgs/mem_bfgs", buffer);
    sprintf(buffer, "%i" , 9);
    fx_set_param(NULL, "/start_scale", buffer);
    sprintf(buffer, "%i" , 5);
    fx_set_param(NULL, "/scaler/leaf_size", buffer);
    sprintf(buffer, "%i", 6);
    fx_set_param(NULL, "/optfun/knns", buffer);
    sprintf(buffer, "%lg", 50.0);

    datanode *module=fx_submodule(NULL, "/", "temp"); 
    engine_->Init(points, module);
    engine_->ComputeOptimum();
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
