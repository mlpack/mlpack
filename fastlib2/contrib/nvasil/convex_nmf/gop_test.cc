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
  }
  
  void Destruct() {
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

    GopNmfEngine<SimpleSplitter> engine;
    SimpleSplitter splitter; 
    splitter.Init();
    engine.Init(module_, &splitter, data_points); 
    engine.ComputeGlobalOptimum();
  }
  
  void Test2() {
    Matrix data_points;
    data::Load("100_1_40_rand.csv", &data_points);
    fx_set_param_int(module_, "new_dimension", 3);
    fx_set_param_double(module_, "opt_gap", 0.001);
    fx_set_param_double(module_, "/relaxed_nmf/grad_tolerance", 1e-4);
    fx_set_param_double(module_, "/relaxed_nmf/scale_factor", 1);
    fx_set_param_int(module_, "/splitter/w_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/h_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/w_offset", data_points.n_rows());
    fx_set_param_int(module_, "/splitter/h_offset", 0);
    fx_set_param_int(module_, "/splitter/h_length", data_points.n_rows());
    
    fx_module *splitter_module=fx_submodule(module_, "splitter");
    GopNmfEngine<TreeOnWandTreeOnHSplitter> engine; 
    TreeOnWandTreeOnHSplitter splitter; 
    splitter.Init(splitter_module,  data_points);
    engine.Init(module_, &splitter, data_points); 
    engine.ComputeGlobalOptimum();
    
  }
  
  void Test3() {
    Matrix data_points;
    data::Load("100_1_40_rand.csv", &data_points);
    fx_set_param_int(module_, "new_dimension", 2);
    fx_set_param_double(module_, "opt_gap", 0.001);
    fx_set_param_double(module_, "/relaxed_nmf/grad_tolerance", 1e-4);
    fx_set_param_double(module_, "/relaxed_nmf/scale_factor", 1);
    
    GopNmfEngine<SimpleSplitter> engine; 
    SimpleSplitter splitter; 
    splitter.Init();
    engine.Init(module_, &splitter, data_points); 
    engine.ComputeGlobalOptimum();
    
  }
  void Test4() {
    Matrix data_points;
    data::Load("100_1_40_rand.csv", &data_points);
    fx_set_param_int(module_, "new_dimension", 1);
    fx_set_param_double(module_, "opt_gap", 0.001);
    fx_set_param_double(module_, "/relaxed_nmf/grad_tolerance", 1e-4);
    fx_set_param_double(module_, "/relaxed_nmf/scale_factor", 1);
    fx_set_param_int(module_, "/splitter/w_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/h_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/w_offset", data_points.n_rows());
    fx_set_param_int(module_, "/splitter/h_offset", 0);
    fx_set_param_int(module_, "/splitter/h_length", data_points.n_rows());
    fx_set_param_double(module_, "/splitter/minimum_interval_length", 1e-3); 
    fx_module *splitter_module=fx_submodule(module_, "splitter");
    GopNmfEngine<TreeOnWandBuildTreeOnHSplitter> engine; 
    TreeOnWandBuildTreeOnHSplitter splitter; 
    splitter.Init(splitter_module,  data_points);
    engine.Init(module_, &splitter, data_points); 
    engine.ComputeGlobalOptimum();
    
  }
  void Test5() {
    Matrix data_points;
    data::Load("100_1_40_rand.csv", &data_points);
    fx_set_param_int(module_, "new_dimension", 3);
    fx_set_param_double(module_, "opt_gap", 0.001);
    fx_set_param_double(module_, "/relaxed_nmf/grad_tolerance", 1e-4);
    fx_set_param_double(module_, "/relaxed_nmf/scale_factor", 1);
    fx_set_param_int(module_, "/splitter/w_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/h_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/w_offset", data_points.n_rows());
    fx_set_param_int(module_, "/splitter/h_offset", 0);
    fx_set_param_int(module_, "/splitter/h_length", data_points.n_rows());
    fx_set_param_double(module_, "/splitter/minimum_interval_length", 1e-3); 
    fx_module *splitter_module=fx_submodule(module_, "splitter");
    GopNmfEngine<BuildTreeOnWandBuildTreeOnHSplitter> engine; 
    BuildTreeOnWandBuildTreeOnHSplitter splitter; 
    splitter.Init(splitter_module,  data_points);
    engine.Init(module_, &splitter, data_points); 
    engine.ComputeGlobalOptimum();
    
  }
  void Test6() {
    Matrix data_points;
    data::Load(
        "/net/hu15/gtg739c/fastlib3/fastlib2/contrib/nvasil/convex_nmf/v_5_1_4.csv", &data_points);
    fx_set_param_int(module_, "new_dimension", 1);
    fx_set_param_double(module_, "opt_gap", 0.001);
    fx_set_param_double(module_, "/l_bfgs/sigma", 1);
    fx_set_param_double(module_, "/l_bfgs/gamma", 1.5); 
    fx_set_param_double(module_, "/l_bfgs/silent", true);
    fx_set_param_double(module_, "/l_bfgs/show_warnings", false);
    fx_set_param_double(module_, "/relaxed_nmf/grad_tolerance", 1e-2);
    fx_set_param_double(module_, "/relaxed_nmf/scale_factor", 1);
    fx_set_param_double(module_, "/relaxed_nmf/opt_gap", 1e-9);
    fx_set_param_int(module_, "/splitter/w_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/h_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/w_offset", data_points.n_rows());
    fx_set_param_int(module_, "/splitter/h_offset", 0);
    fx_set_param_int(module_, "/splitter/h_length", data_points.n_rows());
    fx_set_param_int(module_, "/splitter/w_length", data_points.n_cols());
    fx_set_param_double(module_, "/splitter/minimum_interval_length", 1e-5); 
    fx_module *splitter_module=fx_submodule(module_, "splitter");
    GopNmfEngine<BuildTreeOnWandBuildTreeOnHSplitter, RelaxedRescaledNmfL1> engine; 
    BuildTreeOnWandBuildTreeOnHSplitter splitter; 
    splitter.Init(splitter_module,  data_points);
    engine.Init(module_, &splitter, data_points); 
    engine.ComputeGlobalOptimum();
    
  }
 
   void Test7() {
    Matrix data_points;
    data::Load(
        "/net/hu15/gtg739c/fastlib3/fastlib2/contrib/nvasil/convex_nmf/v_5_1_4.csv", &data_points);
    Matrix h_solution;
    Matrix w_solution;
    data::Load(
        "/net/hu15/gtg739c/fastlib3/fastlib2/contrib/nvasil/convex_nmf/w_5_1_4.csv", 
        &w_solution);
    data::Load(
        "/net/hu15/gtg739c/fastlib3/fastlib2/contrib/nvasil/convex_nmf/h_5_1_4.csv", 
        &h_solution);
    fx_set_param_int(module_, "new_dimension", 1);
    fx_set_param_double(module_, "opt_gap", 0.001);
    fx_set_param_double(module_, "/l_bfgs/sigma", 1);
    fx_set_param_double(module_, "/l_bfgs/gamma", 1.5); 
    fx_set_param_double(module_, "/l_bfgs/silent", true);
    fx_set_param_double(module_, "/l_bfgs/show_warnings", false);
    fx_set_param_double(module_, "/relaxed_nmf/grad_tolerance", 1e-2);
    fx_set_param_double(module_, "/relaxed_nmf/opt_gap", 1e-9);
    fx_set_param_double(module_, "/relaxed_nmf/scale_factor", 1);
    fx_set_param_int(module_, "/splitter/w_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/h_leaf_size", 1);
    fx_set_param_int(module_, "/splitter/w_offset", data_points.n_rows());
    fx_set_param_int(module_, "/splitter/h_offset", 0);
    fx_set_param_int(module_, "/splitter/h_length", data_points.n_rows());
    fx_set_param_int(module_, "/splitter/w_length", data_points.n_cols());
    fx_set_param_double(module_, "/splitter/minimum_interval_length", 1e-5); 
    fx_module *splitter_module=fx_submodule(module_, "splitter");
    GopNmfEngine<TreeSplitterDemo, RelaxedRescaledNmfL1> engine; 
    TreeSplitterDemo splitter; 
    splitter.Init(splitter_module,  data_points, w_solution, h_solution);
    engine.Init(module_, &splitter, data_points); 
    engine.ComputeGlobalOptimum();
    
  }
 
void TestAll() {
    Init();
    Test6();
    Destruct();
  }

 private:
  fx_module *module_;
};

int main(int argc, char *argv[]) {
  fx_module *fx_root=fx_init(argc, argv, &gop_nmf_engine_doc);
  GopNmfEngineTest test(fx_root);
  test.TestAll();
  fx_done(fx_root);
}
