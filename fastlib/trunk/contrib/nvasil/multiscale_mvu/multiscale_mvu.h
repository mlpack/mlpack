/*
 * =====================================================================================
 * 
 *       Filename:  multiscale_mvu.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/06/2008 12:51:32 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */


#ifndef MULTISCALE_MVU_H_
#define MULTISCALE_MVU_H_
#include "fastlib/fastlib.h"
#include "../all_centroid_knn/all_centroid_knn.h"
#include "../mvu/mvu_objectives.h"
#include "../l_bfgs/l_bfgs.h"

template<typename OptimizedFunction>
class MultiscaleMVU {
 public:
  MultiscaleMVU(){
  };
  void Init(Matrix &points, datanode *module) {
    module_=module;
    datanode *scaler_module=fx_submodule(module_, "/scaler", "scaler");
    scaler_.Init(points, scaler_module); 
    start_scale_=fx_param_int(module_, "start_scale", scaler_.tree_min_depth()-1);
    end_scale_=fx_param_int(module_, "end_scale", scaler_.tree_max_depth());
    step_scale_=fx_param_int(module_, "scale_step", 1);
    dimension_=points.n_rows();
    num_of_points_=points.n_cols();
    points_.Copy(points);
  }
  void Destruct() {
  
  }
  void ComputeOptimum() {
    datanode *l_bfgs_node=fx_submodule(module_, "/l_bfgs", "opt");
    datanode *optfun_node=fx_submodule(module_, "/optfun", "optfun");
    new_dimension_=fx_param_int(l_bfgs_node, "new_dimension", 2);
    // get all the centroids from the tree
    Matrix centroids;
    scaler_.ComputeCentroids(&centroids); 
    index_t num_of_centroids=centroids.n_cols();
    // for all the scales do the optimization
    ArrayList<index_t> centroid_ids;
    // This will store the results of the optimization in the intermediate
    // steps for the centroids
    Matrix result;
    result.Init(new_dimension_, num_of_centroids);
    for(index_t i=0; i<result.n_cols(); i++) {
      for(index_t j=0; j<result.n_rows(); j++) {
        result.set(j, i, math::Random(0,1));
      }
    }
    
    double last_sigma=fx_param_double(l_bfgs_node, "sigma", 10);
    // First you finish for all inermidiate steps
 
    for(index_t level=start_scale_; level<=end_scale_; level+=step_scale_) {
      LBfgs<OptimizedFunction> l_bfgs;
      OptimizedFunction optfun;
      // This will keep the results that also serve as initializations for the
      // next step in a compact matrix
      Matrix init_data;
      // This will keep the centroids at a certain level
      Matrix interim_data;
  
      scaler_.RetrieveCentroids(level, &centroid_ids, &result); 
      NOTIFY("Scale:%i Number_of_points:%i\n", level, centroid_ids.size());
      // These are the intermediate data for optimization
      // the centroids and the resulting optimized coordinates
      interim_data.Init(dimension_, centroid_ids.size());
      // These coordinates will be used as a starting point for optimization
      init_data.Init(new_dimension_, centroid_ids.size());
      for(index_t j=0; j<centroid_ids.size(); j++) {
        init_data.CopyColumnFromMat(j, centroid_ids[j], result);
        interim_data.CopyColumnFromMat(j, centroid_ids[j], centroids);
      }
      data::Save("init.csv", init_data);
      data::Save("centroids.csv", interim_data);
      optfun.Init(optfun_node, interim_data);
      char buffer[128];
      sprintf(buffer, "%i", init_data.n_cols());
      fx_set_param(l_bfgs_node, "num_of_points", buffer);
      l_bfgs.Init(&optfun, l_bfgs_node);
      l_bfgs.set_sigma(std::max(last_sigma/100, 10.0));
      l_bfgs.set_coordinates(init_data);
      l_bfgs.ComputeLocalOptimumBFGS();
      init_data.Destruct();
      l_bfgs.GetResults(&init_data);
      last_sigma=l_bfgs.sigma();     
      data::Save("result.csv", init_data);
     
      // Now put the results back to interim data
      init_data.Destruct();
      l_bfgs.GetResults(&init_data);
      for(index_t j=0; j<centroid_ids.size(); j++) {
        result.CopyColumnFromMat(centroid_ids[j], j, init_data);
      }

      centroid_ids.Renew();
   }
    
    // After you finish the intermediate steps we have to run it for 
    // the points
    Matrix final_init;
//    scaler_.FromCentroidsToPointsRecurse(result, 
//       end_scale_, &final_init);
    scaler_.RetrieveCentroids(end_scale_, &centroid_ids, &result); 
    scaler_.FromCentroidsToPoints1(centroids, points_, result,
        centroid_ids,   &final_init); 
    LBfgs<OptimizedFunction> l_bfgs;
    OptimizedFunction optfun;
    optfun.Init(optfun_node, points_);
    char buffer[128];
    sprintf(buffer, "%i", points_.n_cols());
    fx_set_param(l_bfgs_node, "num_of_points", buffer);
    data::Save("result.csv", final_init);

    l_bfgs.Init(&optfun, l_bfgs_node);
    l_bfgs.set_coordinates(final_init);
    //l_bfgs.set_sigma(1000);
    l_bfgs.ComputeLocalOptimumBFGS();
    Matrix end_results;
    l_bfgs.GetResults(&end_results);
    data::Save("result.csv", end_results);
  }
  
 private:
  FORBID_ACCIDENTAL_COPIES(MultiscaleMVU<OptimizedFunction>);
  datanode *module_;
  AllCentroidkNN  scaler_;
  index_t start_scale_;
  index_t end_scale_;
  index_t step_scale_;
  Matrix points_;
  index_t num_of_points_;
  index_t dimension_;
  index_t new_dimension_;
};

#endif
