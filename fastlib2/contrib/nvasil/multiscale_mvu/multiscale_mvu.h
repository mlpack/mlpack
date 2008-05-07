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
    start_scale_=fx_param_int(module_, "start_scale", scaler_.tree_min_depth());
    end_scale_=fx_param_int(module_, "end_scale", scaler_.tree_max_depth());
    step_scale_=fx_param_int(module_, "scale_step", 1);
    dimension_=points.n_rows();
    num_of_points_=points.n_cols();
  }
  void Destruct();
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
    // First you finish for all inermidiate steps
    for(index_t level=start_scale_; level<=end_scale_; level+=step_scale_) {
      scaler_.RetrieveCentroids(level, &centroid_ids, &result); 
      // These are the intermediate data for optimization
      // the centroids and the resulting optimized coordinates
      Matrix interim_data;
      interim_data.Init(dimension_, centroid_ids.size());
      // These coordinates will be used as a starting point for optimization
      Matrix init_data;
      init_data.Init(new_dimension_, centroid_ids.size());
      for(index_t j=0; j<centroid_ids.size(); j++) {
        init_data.CopyColumnFromMat(j, centroid_ids[j], result);
        interim_data.CopyColumnFromMat(j, centroid_ids[j], centroids);
      }
      OptimizedFunction optfun;
      optfun.Init(optfun_node, interim_data);
      l_bfgs_.Init(l_bfgs_node, optfun);
      l_bfgs_.set_coordinates(init_data);
      l_bfgs_.ComputeLocalOptimumBFGS();
      // Now put the results back to interim data
      interim_data.Destruct();
      l_bfgs_.GetResults(&interim_data);
      for(index_t j=0; j<centroid_ids.size(); j++) {
        result.CopyColumnFromMat(centroid_ids[j], j, init_data);
      }
      l_bfgs_.Destruct();
      centroid_ids.Renew();
      init_data.Destruct();
      interim_data.Destruct();
    }
    // After you finish the intermediate steps we have to run it for 
    // the points
    Matrix final_result;
    final_result.Init(new_dimension_, num_of_points_);
    scaler_.FromCentroidsToPointsRecurse(result, 
      end_scale_, &final_result);
  }
  
 private:
  FORBID_ACCIDENTAL_COPIES(MultiscaleMVU<OptimizedFunction>);
  datanode *module_;
  LBfgs<OptimizedFunction> l_bfgs_;
  AllCentroidkNN  scaler_;
  index_t start_scale_;
  index_t end_scale_;
  index_t step_scale_;
  index_t num_of_points_;
  index_t dimension_;
  index_t new_dimension_;
};

#endif
