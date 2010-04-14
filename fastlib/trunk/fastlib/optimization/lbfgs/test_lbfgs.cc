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
/*
 * =====================================================================================
 *
 *       Filename:  test_lbfgs.cc
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
#include "lbfgs.h"
#include "mlpack/mvu/mvu_objectives.h"
#include <string>

class LbfgsTest {
 public:
  LbfgsTest(){
    data_file_="swiss_roll_1000.csv";
  }
  void TestMaxVar1() {
    Matrix test_data;
    data::Load(data_file_.c_str(), &test_data);
    Lbfgs<MaxVariance> engine;
    MaxVariance opt_object;
    opt_object.Init(NULL, test_data);
    engine.Init(&opt_object, NULL);
    engine.ComputeLocalOptimumBFGS();
    Matrix results;
    engine.CopyCoordinates(&results);
    data::Save("max_var", results);
    engine.Destruct();
  }
  void TestMaxVar2() {
    /*
    Matrix test_data;
    data::Load(data_file_.c_str(), &test_data);
    Lbfgs<MaxVarianceInequalityOnFurthest> engine;
    MaxVarianceInequalityOnFurthest opt_object;
    opt_object.Init(NULL, test_data);
    engine.Init(&opt_object, NULL);
    engine.ComputeLocalOptimumBFGS();
    Matrix results;
    engine.CopyCoordinates(&results);
    data::Save("max_var_ineq", results);
    engine.Destruct();
    */
  }
  void TestMaxVar3() {
    Matrix test_data;
    data::Load(data_file_.c_str(), &test_data);
    Lbfgs<MaxFurthestNeighbors> engine;
    MaxFurthestNeighbors  opt_object;
    opt_object.Init(NULL, test_data);
    engine.Init(&opt_object, NULL);
    engine.ComputeLocalOptimumBFGS();
    Matrix results;
    engine.CopyCoordinates(&results);
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
  fx_init(argc, argv, NULL);
  LbfgsTest test;
  test.TestAll();
  fx_done(NULL);
}

