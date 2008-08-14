/*
 * =====================================================================================
 *
 *       Filename:  gop_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/30/2008 11:35:48 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "fastlib/fastlib.h"
#include "gop_nmf.h"

class GopNmfEngineTest {
 public:
  GopNmfEngineTest(fx_module *module) {
    module_=module;
  }
  void Init() {
    engine_ = new GopNmfEngine();
  }
  void Destruct() {
    delete engine_;
  }
  void Test1() {
    Matrix data_points;
    data::Load("/net/hg200/nvasil/dataset/orl_faces/orl_test_faces_100.csv", &data_points);
    fx_set_param_int(module_, "new_dimension",30);
    engine_->Init(module_, data_points); 
    engine_->ComputeGlobalOptimum();
  }
  void Test2() {
    Matrix data_points;
    data::Load("/net/hg200/nvasil/dataset/orl_faces/orl_test_faces_100.csv", &data_points);
    fx_set_param_int(module_, "new_dimension",30);
    engine_->Init(module_, data_points); 
    engine_->ComputeTighterGlobalOptimum();
  }
 void TestAll() {
    Init();
    Test1();
    Destruct();
  }

 private:
  fx_module *module_;
  GopNmfEngine *engine_;
   
};

int main(int argc, char *argv[]) {
  fx_module *fx_root=fx_init(argc, argv, &gop_nmf_engine_doc);
  GopNmfEngineTest test(fx_root);
  test.TestAll();
  fx_done(fx_root);
}
