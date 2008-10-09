/*
 * =====================================================================================
 *
 *       Filename:  main_classifier.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/08/2008 05:54:43 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <string>
#include "mvu_classification.h"
#include "../l_bfgs/l_bfgs.h"

int main(int argc, char *argv[]) {
  fx_module *fx_root=fx_init(argc, argv, NULL);  
  std::string labeled_points_file = fx_param_str_req(fx_root, "labeled_points_file");
  std::string labels_file = fx_param_str_req(fx_root, "labels_file");
  std::string unlabeled_points_file=fx_param_str_req(fx_root, "unlabeled_points_file");
  index_t num_of_classes = fx_param_int_req(fx_root, "num_of_classes");
  MaxFurthestNeighborsSemiSupervised opt_fun;
  LBfgs<MaxFurthestNeighborsSemiSupervised> engine;
  Matrix labeled_data_points;
  data::Load(labeled_points_file.c_str(), &labeled_data_points);
  Matrix unlabeled_data_points;
  data::Load(unlabeled_points_file.c_str(), &unlabeled_data_points);
  Matrix labels;
  data::Load(labels_file.c_str(), &labels);
  NOTIFY("Doing a sanity check on the labels");
  for(index_t i=0; i<labels.n_cols(); i++) {
    if (labels.get(0, i)<0 || labels.get(0, i)>=num_of_classes) {
      FATAL("Sanity check failed label %i:%i is out of the class range [0 %i]",
            i, index_t(labels.get(0, i)), num_of_classes);
    }
  }

  fx_module *opt_fun_module = fx_submodule(fx_root, "opt_fun");
  fx_module *l_bfgs_module = fx_submodule(fx_root, "l_bfgs");
  opt_fun.Init(opt_fun_module, labeled_data_points, unlabeled_data_points);
  engine.Init(&opt_fun, l_bfgs_module);
  engine.ComputeLocalOptimumBFGS();
  
  labeled_data_points.CopyColumnFromMat(0, 
                                        0, 
                                        labeled_data_points.n_rows(), 
                                        *engine.coordinates());  

  unlabeled_data_points.CopyColumnFromMat(labeled_data_points.n_cols(), 
                                          0, 
                                          unlabeled_data_points.n_rows(), 
                                          *engine.coordinates());  
   
  AllkNN allknn;
  fx_module *allknn_module = fx_submodule(fx_root, "allknn");
  index_t knns=fx_param_int(allknn_module, "knns", 5);
  allknn.Init(unlabeled_data_points, labeled_data_points, allknn_module);
  ArrayList<index_t> neighbors;
  ArrayList<double> distances;
  allknn.ComputeNeighbors(&neighbors, &distances);
  // now do the knn classification
  Matrix classification_results;
  classification_results.Init(1, unlabeled_data_points.n_cols());
  for(index_t i=0; i<unlabeled_data_points.n_cols(); i++) {
    index_t score[num_of_classes];
    memset(score, 0, num_of_classes*sizeof(index_t));
    for(index_t j=0; j<knns; j++) {
      index_t n=neighbors[i*knns+j];
      score[index_t(labels.get(0,n))]+=1;
    }
    ptrdiff_t winner = std::max_element(score, score+num_of_classes)-score;
    classification_results.set(0, i, winner);
  }
  data::Save("classification_results", classification_results);
  index_t total_hits=0;
  if (fx_param_exists(fx_root, "validation_file")) {
    // find the classification score
    std::string validation_file =fx_param_str_req(fx_root, "validation_file");
    Matrix validation_labels;
    data::Load(validation_file.c_str(), &validation_labels);
    for(index_t i=0; i<classification_results.n_cols(); i++) {
      if (classification_results.get(0, i) == validation_labels.get(0, i)) {
        total_hits+=1;
      }
    }
  }
  double total_score=100.0*total_hits/classification_results.n_cols();
  fx_result_double(fx_root, "classification_score", total_score);
  NOTIFY("Classification Results %lg%%", total_score);
  fx_done(fx_root);
}
