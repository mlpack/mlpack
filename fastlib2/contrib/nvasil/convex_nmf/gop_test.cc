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
#include "splitter.h"

class GopNmfEngineTest {
 public:
  GopNmfEngineTest(fx_module *module) {
    module_=module;
  }
  void Init() {
    engine_ = new GopNmfEngine<SimpleSplitter>();
  }
  void Destruct() {
    delete engine_;
    delete splitter_;
  }
  void Test1() {
    Matrix data_points;
    data::Load(
        //"/net/hg200/nvasil/dataset/orl_faces/orl_test_faces_100.csv"
        "6.csv"
        , &data_points);
    fx_set_param_int(module_, "new_dimension", 2);
    fx_set_param_double(module_, "opt_gap", 0.001);
    fx_set_param_double(module_, "/relaxed_nmf/grad_tolerance", 1e-4);
    fx_set_param_double(module_, "/relaxed_nmf/scale_factor", 1);
    splitter_ = new SimpleSplitter();
    splitter_->Init();
    engine_->Init(module_, splitter_, data_points); 
    engine_->ComputeGlobalOptimum();
  }
void TestAll() {
    Init();
    Test1();
    Destruct();
  }

 private:
  fx_module *module_;
  GopNmfEngine<SimpleSplitter> *engine_;
  SimpleSplitter *splitter_; 
};

int main(int argc, char *argv[]) {
  fx_module *fx_root=fx_init(argc, argv, &gop_nmf_engine_doc);
  GopNmfEngineTest test(fx_root);
  test.TestAll();
  fx_done(fx_root);
}
