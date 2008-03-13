/*
 * =====================================================================================
 *
 *       Filename:  test_l_bfgs.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/12/2008 01:27:16 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include "l_bfgs.h"
#include "../mvu/mvu_objectives.h"
#include <string>

class LBfgsTest {
 public:
  LBfgsTest(){
    data_file_="swiss_roll_1000.csv";
  }
  void TestMaxVar1() {
    Matrix test_data;
    data::Load(data_file_.c_str(), &test_data);
    LBfgs<MaxVariance> engine;
    MaxVariance opt_object;
    opt_object.Init(NULL, test_data);
    engine.Init(&opt_object, NULL);
    engine.ComputeLocalOptimumBFGS();
    Matrix results;
    engine.GetResults(&results);
    data::Save("max_var", results);
    engine.Destruct();
  }
  void TestMaxVar2() {
    Matrix test_data;
    data::Load(data_file_.c_str(), &test_data);
    LBfgs<MaxVarianceInequalityOnFurthest> engine;
    MaxVarianceInequalityOnFurthest opt_object;
    opt_object.Init(NULL, test_data);
    engine.Init(&opt_object, NULL);
    engine.ComputeLocalOptimumBFGS();
    Matrix results;
    engine.GetResults(&results);
    data::Save("max_var_ineq", results);
    engine.Destruct();
  }
  void TestMaxVar3() {
    Matrix test_data;
    data::Load(data_file_.c_str(), &test_data);
    LBfgs<MaxFurthestNeighbors> engine;
    MaxFurthestNeighbors  opt_object;
    opt_object.Init(NULL, test_data);
    engine.Init(&opt_object, NULL);
    engine.ComputeLocalOptimumBFGS();
    Matrix results;
    engine.GetResults(&results);
    data::Save("max_furth", results);
    engine.Destruct();
  }
  void TestAll() {
   // TestMaxVar1();
   // TestMaxVar2();
    TestMaxVar3();
  }
   
 private:
  std::string data_file_;
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  LBfgsTest test;
  test.TestAll();
  fx_done();
}

