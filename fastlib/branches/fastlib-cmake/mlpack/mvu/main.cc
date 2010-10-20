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
//#include "mlpack/optimization/lbfgs/lbfgs.h"
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

const fx_entry_doc main_entries[] = {
  {"optimized_function", FX_PARAM, FX_STR, NULL,
   " choose the method MVU or MFNU .\n"},
  {"new_dimension", FX_PARAM, FX_INT, NULL,
   " The dimension of the nonlinear projection (MVU or MFNU).\n"},
  {"data_file", FX_REQUIRED, FX_STR, NULL,
   " the csv file with the data.\n"},
  {"result_file", FX_PARAM, FX_STR, NULL,
   " the results of the method are exported to a csv file.\n"},
  {"pca_pre", FX_PARAM, FX_BOOL, NULL,
   " sometimes it is good to do pca preprocessing and then MVU/MFNU.\n"},
  {"pca_dim", FX_PARAM, FX_INT, NULL,
   " the projection dimension with PCA if chosen.\n"},
  {"pca_init", FX_PARAM, FX_BOOL, NULL,
   " if this flag is true then the optimization of MVU/MFNU is initialized .\n"},
 FX_ENTRY_DOC_DONE
};

const fx_submodule_doc main_submodules[] = {
  {"optfun", &mvu_doc,
   " Responsible for intializing MVU/MFNU.\n"},
   {"lbfgs", &lbfgs_doc,
    " Responsible for the Limited BFGS module.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc main_doc = {
  main_entries, main_submodules,
  " This program computes Maximum Variance Unfolding \n"
  " and Maximum Furthest Neighbor Unfolding\n"
};

int main(int argc, char *argv[]){
  fx_module *fx_root=fx_init(argc, argv, &main_doc);

  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("new_dimension", boost_po::value<int>(), "  the numbe of dimensions for the unfolded\n")
      ("nearest_neighbor_file", boost_po::value<std::string>(), " file with the nearest neighbor pairs and the squared distances defaults to nearest.txt\n")
      ("furthest_neighbor_file", boost_po::value<std::string>(), " file with the nearest neighbor pairs and the squared distances")
      ("knns", boost_po::value<int>(), " number of nearest neighbors to build the graph if you choose the option with the nearest file you don't need to specify it")
      ("leaf_size", boost_po::value<int>(), " leaf size for the tree. if you choose the option with the nearest file you don't need to specify it\n")
      ("optimized_function", boost_po::value<std::string>(), " choose the method MVU or MFU")
      ("new_dimension", boost_po::value<int>(), " The dimension of the nonlinear projection (MVU or MFNU)")
      ("data_file", boost_po::value<int>(), " the csv file with the data.\n")
      ("result_file", boost_po::value<std::string>(), " the results of the method are exported to a csv file.\n")
      ("pca_pre", boost_po::value<bool>(), " sometimes it is good to do pca preprocessing and then MVU/MFNU")
      ("pca_dim", boost_po::value<int>(), " the projection dimension with PCA if chosen.\n")
      ("pca_init", boost_po::value<bool>(), " if this flag is true then the optimization of MVU/MFNU is initialized.\n");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

  //std::string optimized_function=fx_param_str(fx_root, "/optimized_function", "mvfu");
  std::string optimized_function= vm["/optimized_function"].as<std::string>();
  fx_module *optfun_node;
  fx_module *l_bfgs_node;
  l_bfgs_node=fx_submodule(fx_root, "/lbfgs");
  optfun_node=fx_submodule(fx_root, "/optfun");
  // this is sort of a hack and it has to be eliminated in the final version
  //index_t new_dimension=fx_param_int(l_bfgs_node, "/new_dimension", 2);
  index_t new_dimension = vm["/new_dimension"].as<int>();

  fx_set_param_int(optfun_node, "new_dimension", new_dimension); 
  
  if (!fx_param_exists(fx_root, "/optfun/nearest_neighbor_file")) {
    Matrix data_mat;
    //std::string data_file=fx_param_str_req(fx_root, "/data_file");
    std::string data_file = vm["/data_file"].as<std::string>();
    if (data::Load(data_file.c_str(), &data_mat)==SUCCESS_FAIL) {
      FATAL("Didn't manage to load %s", data_file.c_str());
    }
    NOTIFY("Removing the mean., centering data...");
    OptUtils::RemoveMean(&data_mat);
  
 
    //bool pca_preprocess=fx_param_bool(fx_root, "/pca_pre", false);
    //index_t pca_dimension=fx_param_int(fx_root, "/pca_dim", 5);
    bool pca_preprocess = vm["/pca_pre"].as<bool>();
    index_t pca_dimension = vm["/pca_dim"].as<index_t>();
    bool pca_init = vm["/pca_init"].as<bool>();

    //bool pca_init=fx_param_bool(fx_root, "/pca_init", false);
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
      //index_t new_dimension=fx_param_int(l_bfgs_node, "new_dimension", 2);
      index_t new_dimension = vm["new_dimension"].as<index_t>();
      OptUtils::SVDTransform(data_mat, initial_data, new_dimension);
    }
  
    //we need to insert the number of points
    fx_set_param_int(l_bfgs_node, "num_of_points", data_mat.n_cols());
    //std::string result_file=fx_param_str(fx_root, "/result_file", "result.csv");
    std::string result_file = vm["/result_file"].as<std::string>();
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
        FATAL("Didn't manage to save %s", result_file.c_str());
      }
      done=true;
    }
   if (optimized_function == "mvfu"){
      MaxFurthestNeighbors opt_function;
      opt_function.Init(optfun_node, data_mat);
      //opt_function.set_lagrange_mult(0.0);
      Lbfgs<MaxFurthestNeighbors> engine;
      fx_set_param_bool(l_bfgs_node, "use_default_termination", false);
      engine.Init(&opt_function, l_bfgs_node);
      if (pca_init==true) {
        la::Scale(1e-1, initial_data);
        engine.set_coordinates(*initial_data);
      }
      engine.ComputeLocalOptimumBFGS();
      if (data::Save(result_file.c_str(), *engine.coordinates())==SUCCESS_FAIL) {
        FATAL("Didn't manage to save %s", result_file.c_str());
      }
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
  } else {
    // This is for nadeem
    
    //std::string result_file=fx_param_str(NULL, "/result_file", "result.csv");
    std::string result_file = vm["/result_file"].as<std::string>();

    bool done=false;
    
    if (optimized_function == "mvu") {
      MaxVariance opt_function;
      opt_function.Init(optfun_node);
      Matrix init_mat;
      //we need to insert the number of points
      fx_set_param_int(l_bfgs_node, "num_of_points", opt_function.num_of_points());

      Lbfgs<MaxVariance> engine;
      engine.Init(&opt_function, l_bfgs_node);
      engine.ComputeLocalOptimumBFGS();
      if (data::Save(result_file.c_str(), *engine.coordinates())==SUCCESS_FAIL) {
        FATAL("Didn't manage to save %s", result_file.c_str());
      }
      done=true;
    }
    if (optimized_function == "mvfu"){
      MaxFurthestNeighbors opt_function;
      opt_function.Init(optfun_node);
      //we need to insert the number of points
      fx_set_param_int(l_bfgs_node, "num_of_points", opt_function.num_of_points());
      fx_set_param_bool(l_bfgs_node, "use_default_termination", false);

      Lbfgs<MaxFurthestNeighbors> engine;
      engine.Init(&opt_function, l_bfgs_node);
      engine.ComputeLocalOptimumBFGS();
      if (data::Save(result_file.c_str(), *engine.coordinates())==SUCCESS_FAIL) {
        FATAL("Didn't manage to save %s", result_file.c_str());
      }
      done=true;
    }
    if (done==false) {
      FATAL("The method you provided %s is not supported", 
          optimized_function.c_str());
    }
    fx_done(fx_root);
  }
}
