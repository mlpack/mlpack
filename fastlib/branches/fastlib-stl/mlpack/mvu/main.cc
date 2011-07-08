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
#include <fastlib/fx/io.h>
#include "mvu_objectives.h"
#include "fastlib/optimization/lbfgs/lbfgs.h"
/**
 * @author Nikolaos Vasiloglou (nvasil@ieee.org)
 * @file main.cc
 * 
 * This program computes the Maximum Variance Unfolding
 * or Maximum Furthest Neighbor Unfolding
 * as defined in the paper
 * @conference{vasiloglou2008ssm,
 *     title={{Scalable semidefinite manifold learning}},
 *     author={Vasiloglou, N. and Gray, A.G. and Anderson, D.V.},
 *     booktitle={Machine Learning for Signal Processing, 2008. MLSP 2008. IEEE Workshop on},
 *     pages={368--373},
 *     year={2008}
 * }
 * usage: 
 * >>ncmvu --/optimized_function=[mvu,mfnu] 
 *         --/new_dimension=[number of the new dimensions for MVU/MFNU]
 *         --/data_file=foo.csv[every row is a point]
 *         --/result_file=bar.csv [the results after the projection of MVU/MFNU]
 *         --/pca_pre=[true or false] (pca preprocessing of the data)
 *         --/pca_dim=[number of the PCA projection]
 *         --/pca_init=[true or false] 
 */

PARAM_STRING("optimized_function", "choose the method, MVU or MFNU.", 
  "lbfgs", "mvfu"); //may be a bug, but default value is mvfu...

PARAM_STRING_REQ("data_file", "the csv file with the data", "lbfgs");
PARAM_STRING("result_file", "the results of the method are exported to a\
 csv file.", "lbfgs", "result.csv");

PARAM_INT("new_dimension", "The dimension of the nonlinear projection\
 (MVU, or MFNU).", "lbfgs", 2);

PARAM_INT("pca_dim", "the projection dimension with PCA if chosen", "lbfgs", 5);
PARAM_FLAG("pca_pre", "sometimes it is good to do pca preprocessing and then\
 MVU/MFNU.", "lbfgs");
PARAM_FLAG("pca_init", "if this flag is true then the optimization of MVU/MFNU\
 is initialized.", "lbfgs");

PARAM_MODULE("lbfgs", "Responsible for the Limited BFGS module.");
PARAM_MODULE("optfun", "Responsible for initializing MVU/MFNU.");

PROGRAM_INFO("MVU", "This program computes the Maximum Variance Unfolding and\
 Maximum Furthest Neighbor Unfolding");

using namespace mlpack;

int main(int argc, char *argv[]){
  IO::ParseCommandLine(argc, argv);

  std::string optimized_function=
    IO::GetParam<std::string>("lbfgs/optimized_function");
  // this is sort of a hack and it has to be eliminated in the final version
  index_t new_dimension=IO::GetParam<int>("lbfgs/new_dimension");
  IO::GetParam<int>("lbfgs/new_dimension") = new_dimension;
  
  if (!IO::HasParam("optfun/nearest_neighbor_file")) {
    Matrix data_mat;
    std::string data_file=IO::GetParam<std::string>("lbfgs/data_file");
    if (data::Load(data_file.c_str(), &data_mat)==SUCCESS_FAIL) {
      IO::Fatal << "Didn't manage to load " << data_file.c_str()) << std::endl;
    }
    IO::Info << "Removing the mean., centering data..." << std::endl;
    OptUtils::RemoveMean(&data_mat);
  
 
    bool pca_preprocess=IO::HasParam("lbfgs/pca_pre");
    index_t pca_dimension=IO::GetParam<int>("lgfgs/pca_dim");
    bool pca_init=IO::HasParam("lbfgs/pca_init");
    Matrix *initial_data=NULL;
    if (pca_preprocess==true) {
      IO::Info << "Preprocessing with pca") << std::endl;
      Matrix temp;
      OptUtils::SVDTransform(data_mat, &temp, pca_dimension);
      data_mat.Destruct();
      data_mat.Own(&temp);
    }
    if (pca_init==true) {
      IO::Info << "Preprocessing with pca" << std::endl;
      initial_data = new Matrix();
      index_t new_dimension=IO::GetParam<int>("lbfgs/new_dimension");
      OptUtils::SVDTransform(data_mat, initial_data, new_dimension);
    }
  
    //we need to insert the number of points
    IO::GetParam<int>("lbfgs/num_of_points") = data_mat.n_cols();
    std::string result_file=IO::GetParam<std::string>("lbfgs/result_file"); 
    bool done=false;
    
    if (optimized_function == "mvu") {
      MaxVariance opt_function;
      opt_function.Init(optfun_node, data_mat);
      Lbfgs<MaxVariance> engine;
      engine.Init(&opt_function, l_bfgs_node);
      if (pca_init==true) {
        engine.set_coordinates(*initial_data);
      }
      engine.ComputeLocalOptimumBFGS();
      if (data::Save(result_file.c_str(), *engine.coordinates())==SUCCESS_FAIL) {
        IO::Fatal << "Didn't manage to save " << result_file.c_str() << std::endl;
      }
      done=true;
    }
   if (optimized_function == "mvfu"){
      MaxFurthestNeighbors opt_function;
      opt_function.Init(optfun_node, data_mat);
      //opt_function.set_lagrange_mult(0.0);
      Lbfgs<MaxFurthestNeighbors> engine;
      IO::GetParam<bool>("lbfgs/use_default_termination") = false;
      engine.Init(&opt_function, l_bfgs_node);
      if (pca_init==true) {
        la::Scale(1e-1, initial_data);
        engine.set_coordinates(*initial_data);
      }
      engine.ComputeLocalOptimumBFGS();
      if (data::Save(result_file.c_str(), *engine.coordinates())==SUCCESS_FAIL) {
        IO::Fatal << "Didn't manage to save " << result_file.c_str() << std::endl;
      }
      done=true;
    }
    if (done==false) {
      IO::Fatal << "The method you provided " << optimized_function.c_str() <<
        " is not supported" << std::endl;
    }
    if (pca_init==true) {
      delete initial_data;
    }
    
  } else {
    // This is for nadeem
    
    std::string result_file=IO::GetParam<std::string>("lbfgs/result_file");
    bool done=false;
    
    if (optimized_function == "mvu") {
      MaxVariance opt_function;
      opt_function.Init(optfun_node);
      Matrix init_mat;
      //we need to insert the number of points
      IO::GetParam<int>("lbfgs/num_of_points") = opt_function.num_of_points();

      Lbfgs<MaxVariance> engine;
      engine.Init(&opt_function, l_bfgs_node);
      engine.ComputeLocalOptimumBFGS();
      if (data::Save(result_file.c_str(), *engine.coordinates())==SUCCESS_FAIL) {
        IO::Fatal << "Didn't manage to save " << result_file << std::endl;
      }
      done=true;
    }
    if (optimized_function == "mvfu"){
      MaxFurthestNeighbors opt_function;
      opt_function.Init(optfun_node);
      //we need to insert the number of points
      IO::GetParam<int>("lbfgs/num_of_points") = opt_function.num_of_points();
      IO::GetParam<bool>("lbfgs/use_default_termination") = false;

      Lbfgs<MaxFurthestNeighbors> engine;
      engine.Init(&opt_function, l_bfgs_node);
      engine.ComputeLocalOptimumBFGS();
      if (data::Save(result_file.c_str(), *engine.coordinates())==SUCCESS_FAIL) {
        IO::Fatal << "Didn't manage to save " << result_file.c_str() << std::endl;
      }
      done=true;
    }
    if (done==false) {
       IO::Fatal << "The method you provided " << optimized_function.c_str() <<
        " is not supported" << std::endl;
    }
  }
}
