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
  fx_module *fx_root=fx_init(argc, argv, NULL);
  std::string optimized_function=fx_param_str(NULL, "opts/optfun", "mvu");
  std::string data_file=fx_param_str_req(NULL, "opts/data_file");
  Matrix data_mat;
  if (data::Load(data_file.c_str(), &data_mat)==SUCCESS_FAIL) {
    FATAL("Didn't manage to load %s", data_file.c_str());
  }
  NOTIFY("Removing the mean., centering data...");
  OptUtils::RemoveMean(&data_mat);
  
  fx_module *optfun_node;
  fx_module *l_bfgs_node;
  l_bfgs_node=fx_submodule(fx_root, "opts/l_bfgs");
  optfun_node=fx_submodule(fx_root, "opts/optfun");
  
  bool pca_preprocess=fx_param_bool(fx_root, "opts/pca_pre", false);
  index_t pca_dimension=fx_param_int(fx_root, "opts/pca_dim", 5);
  bool pca_init=fx_param_bool(fx_root, "opts/pca_init", false);
  Matrix *initial_data=NULL;
  if (pca_preprocess==true) {
    NOTIFY("Preprocessing with pca");
    Matrix temp;
    OptUtils::SVDTransform(data_mat, &temp, pca_dimension);
    data_mat.Destruct();
    data_mat.Own(&temp);
  }
  if (pca_init==true) {
    NOTIFY("Preprocessing with pca");
    initial_data = new Matrix();
    index_t new_dimension=fx_param_int(l_bfgs_node, "new_dimension", 2);
    OptUtils::SVDTransform(data_mat, initial_data, new_dimension);
 }
 
  
  //we need to insert the number of points
  fx_set_param_int(l_bfgs_node, "num_of_points", data_mat.n_cols());
  std::string result_file=fx_param_str(NULL, "opts/result_file", "result.csv");
  bool done=false;
    
  if (optimized_function == "mvu") {
    MaxVariance opt_function;
    opt_function.Init(optfun_node, data_mat);
    LBfgs<MaxVariance> engine;
    engine.Init(&opt_function, l_bfgs_node);
    if (pca_init==true) {
      engine.set_coordinates(*initial_data);
    }
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
    if (pca_init==true) {
      engine.set_coordinates(*initial_data);
    }
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
    //opt_function.set_lagrange_mult(0.0);
    LBfgs<MaxFurthestNeighbors> engine;
    engine.Init(&opt_function, l_bfgs_node);
    if (pca_init==true) {
      la::Scale(1e-1, initial_data);
      engine.set_coordinates(*initial_data);
    }
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
  if (pca_init==true) {
    delete initial_data;
  }
  fx_done(fx_root);
}
