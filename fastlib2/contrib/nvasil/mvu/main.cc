/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/20/2008 12:34:14 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <string>
#include "fastlib/fastlib.h"
#include "mvu_objectives.h"
#include "../l_bfgs/l_bfgs.h"

int main(int argc, char *argv[]){
  fx_init(argc, argv);
  std::string optimized_function=fx_param_str(NULL, "opts/optfun", "mvu");
  std::string data_file=fx_param_str_req(NULL, "opts/data_file");
  Matrix data_mat;
  if (data::Load(data_file.c_str(), &data_mat)==SUCCESS_FAIL) {
    FATAL("Didn't manage to load %s", data_file.c_str());
  }
  datanode *optfun_node;
  datanode *l_bfgs_node;
  l_bfgs_node=fx_submodule(NULL, "opts/l_bfgs", "l_bfgs");
  optfun_node=fx_submodule(NULL, "opts/optfun", "optfun");
 
  //we need to insert the number of points
  char buffer[128];
  sprintf(buffer, "%i", data_mat.n_cols());
  fx_set_param(l_bfgs_node, "num_of_points", buffer);
  std::string result_file=fx_param_str(NULL, "opts/result_file", "result.csv");
  bool done=false;
    
  if (optimized_function == "mvu") {
    MaxVariance opt_function;
    opt_function.Init(optfun_node, data_mat);
    LBfgs<MaxVariance> engine;
    engine.Init(&opt_function, l_bfgs_node);
    engine.ComputeLocalOptimumBFGS();
    Matrix result;
    engine.GetResults(&result);
    if (data::Save(result_file.c_str(), result)==SUCCESS_FAIL) {
      FATAL("Didn't manage to save %s", result_file.c_str());
    }
    engine.Destruct();
    done=true;
  }
  if (optimized_function=="mvuineq") {
    MaxVarianceInequalityOnFurthest opt_function;
    opt_function.Init(optfun_node, data_mat);
    LBfgs<MaxVarianceInequalityOnFurthest> engine;
    engine.Init(&opt_function, l_bfgs_node);
    engine.ComputeLocalOptimumBFGS();
    Matrix result;
    engine.GetResults(&result);
    if (data::Save(result_file.c_str(), result)==SUCCESS_FAIL) {
      FATAL("Didn't manage to save %s", result_file.c_str());
    }
    engine.Destruct();
    done=true;
  }
  if (optimized_function == "mvfu"){
    MaxFurthestNeighbors opt_function;
    opt_function.Init(optfun_node, data_mat);
    LBfgs<MaxFurthestNeighbors> engine;
    engine.Init(&opt_function, l_bfgs_node);
    engine.ComputeLocalOptimumBFGS();
    Matrix result;
    engine.GetResults(&result);
    if (data::Save(result_file.c_str(), result)==SUCCESS_FAIL) {
      FATAL("Didn't manage to save %s", result_file.c_str());
    }
    engine.Destruct();
    done=true;
  }
  if (done==false) {
    FATAL("The method you provided %s is not supported", 
        optimized_function.c_str());
  }
  fx_done();
}
