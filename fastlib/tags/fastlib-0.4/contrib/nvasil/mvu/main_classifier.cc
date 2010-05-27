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

double  ComputeClassificationScore(fx_module *module, 
                                   Matrix &labeled_data_points,
                                   Matrix &labels,
                                   Matrix &unlabeled_data_points, 
                                   Matrix *classification_results);
 
int main(int argc, char *argv[]) {
  fx_module *fx_root=fx_init(argc, argv, NULL);  
  std::string labeled_points_file = fx_param_str_req(fx_root, "labeled_points_file");
  std::string labels_file = fx_param_str_req(fx_root, "labels_file");
  std::string unlabeled_points_file=fx_param_str(fx_root, "unlabeled_points_file", "");
  index_t num_of_classes = fx_param_int_req(fx_root, "num_of_classes");
  std::string mode = fx_param_str(fx_root, "mode", "svm");
  Matrix labeled_data_points;
  data::Load(labeled_points_file.c_str(), &labeled_data_points);
  Matrix unlabeled_data_points;
  if (unlabeled_points_file.empty()) {
    unlabeled_data_points.Init(labeled_data_points.n_rows(), 0);
  } else {
    data::Load(unlabeled_points_file.c_str(), &unlabeled_data_points);
  }
  Matrix labels;
  data::Load(labels_file.c_str(), &labels);
  if (labels.n_cols()==1) {
    Matrix labels1;
    la::TransposeInit(labels, &labels1);
    labels.Destruct();
    labels.Own(&labels1);
  }
  NOTIFY("Doing a sanity check on the labels");
  for(index_t i=0; i<labels.n_cols(); i++) {
    if (labels.get(0, i)<0 || labels.get(0, i)>=num_of_classes) {
      FATAL("Sanity check failed label %i:%i is out of the class range [0 %i]",
            i, index_t(labels.get(0, i)), num_of_classes);
    }
  }
  NOTIFY("sanity check passed");
  
  if (fx_param_exists(fx_root, "validation_file")) {
    // Do a validation test with nearest neighbors only
    NOTIFY("Validating the classification score with simple allknn");
    Matrix classification_results;
    double total_score = ComputeClassificationScore(fx_root, 
                                                    labeled_data_points,
                                                    labels,
                                                    unlabeled_data_points, 
                                                    &classification_results);
    fx_result_double(fx_root, "classification_score", total_score);
    NOTIFY("Simple Classification Results %lg%%", total_score);
  }
  fx_module *opt_fun_module = fx_submodule(fx_root, "opt_fun");
  fx_module *l_bfgs_module = fx_submodule(fx_root, "l_bfgs");
  fx_set_param_double(l_bfgs_module, "use_default_termination", false);
  Matrix result;
  ArrayList<index_t> anchors;
  anchors.Init();
  if (mode == "mvu") {
    MaxFurthestNeighborsSemiSupervised opt_fun;
    LBfgs<MaxFurthestNeighborsSemiSupervised> engine;
    opt_fun.Init(opt_fun_module, labeled_data_points, unlabeled_data_points);
    engine.Init(&opt_fun, l_bfgs_module);
    engine.ComputeLocalOptimumBFGS();
    result.Copy(*engine.coordinates());
  } else {
    if (mode == "svm") {
      MaxFurthestNeighborsSvmSemiSupervised opt_fun;
      LBfgs<MaxFurthestNeighborsSvmSemiSupervised> engine;
      opt_fun.Init(opt_fun_module, 
                   labeled_data_points, 
                   unlabeled_data_points, 
                   labels);
      engine.Init(&opt_fun, l_bfgs_module);
      engine.ComputeLocalOptimumBFGS();
      result.Copy(*engine.coordinates());
      opt_fun.anchors(&anchors);
    } else {
      FATAL("This mode (%s) is not supported", mode.c_str());
    }
  }
 
  index_t new_dimension =fx_param_int_req(fx_root, "/opt_fun/new_dimension");
  index_t num1=labeled_data_points.n_cols();
  labeled_data_points.Destruct(); 
  labeled_data_points.Copy(result.GetColumnPtr(0), new_dimension, num1);  
  data::Save("unfolded.csv", result);
  index_t num2=unlabeled_data_points.n_cols();
  if (!unlabeled_points_file.empty()) {
    unlabeled_data_points.Destruct();
    unlabeled_data_points.Copy(result.GetColumnPtr(num1), 
        new_dimension, num2); 
  }  
  double total_score;
  if (fx_param_exists(fx_root, "validation_file")) {
    NOTIFY("Computing the unfolded optimization score");
    if (mode=="mvu") {
      Matrix classification_results;
      total_score = ComputeClassificationScore(fx_root, 
                                               labeled_data_points,
                                               labels,
                                               unlabeled_data_points, 
                                               &classification_results);
      data::Save("classification_results", classification_results);
    } else {
      if (mode=="svm") {
        // find the classification score
        std::string validation_file =fx_param_str_req(fx_root, "validation_file");
        Matrix validation_labels;
        data::Load(validation_file.c_str(), &validation_labels);
        if (validation_labels.n_cols()==1) {
          Matrix labels1;
          la::TransposeInit(validation_labels, &labels1);
          validation_labels.Destruct();
          validation_labels.Own(&labels1);
        }

        total_score=0.0;
        for(index_t i=0; i<validation_labels.n_cols(); i++) {
          for(index_t j=0; j<1; j++) {
            double anchor_label=labels.get(0, anchors[j]);
            double *p1=labeled_data_points.GetColumnPtr(anchors[j]);
            double *p2=unlabeled_data_points.GetColumnPtr(i);
            double dot_product=la::Dot(new_dimension, p1, p2);
            if (dot_product > 0.0  and anchor_label==validation_labels.get(0, i)) {
              total_score+=1;
            } else {
              if (dot_product <0.0 and anchor_label!=validation_labels.get(0, i)) {
                total_score+=1;
              }
            }
          }
        }
        total_score = 100.0 * total_score/validation_labels.n_cols();
      } else {
        FATAL("This mode (%s) is not supported", mode.c_str());
      }
    }
    fx_result_double(fx_root, "unfolded_classification_score", total_score);
    NOTIFY("Unfolded Classification Results %lg%%", total_score);
  }
  fx_done(fx_root);
}

double  ComputeClassificationScore(fx_module *module, 
                                   Matrix &labeled_data_points,
                                   Matrix &labels,
                                   Matrix &unlabeled_data_points, 
                                   Matrix *classification_results) {
  AllkNN allknn;
  fx_module *allknn_module = fx_submodule(module, "allknn");
  index_t knns=fx_param_int(allknn_module, "knns", 5);
  index_t num_of_classes=fx_param_int_req(module, "num_of_classes");
  allknn.Init(unlabeled_data_points, labeled_data_points, allknn_module);
  ArrayList<index_t> neighbors;
  ArrayList<double> distances;
  allknn.ComputeNeighbors(&neighbors, &distances);
  // now do the knn classification
  classification_results->Init(1, unlabeled_data_points.n_cols());
  for(index_t i=0; i<unlabeled_data_points.n_cols(); i++) {
    index_t score[num_of_classes];
    memset(score, 0, num_of_classes*sizeof(index_t));
    for(index_t j=0; j<knns; j++) {
      index_t n=neighbors[i*knns+j];
      score[index_t(labels.get(0,n))]+=1;
    }
    ptrdiff_t winner = std::max_element(score, score+num_of_classes)-score;
    classification_results->set(0, i, winner);
  }
  index_t total_hits=0;
  if (fx_param_exists(module, "validation_file")) {
    // find the classification score
    std::string validation_file =fx_param_str_req(module, "validation_file");
    Matrix validation_labels;
    data::Load(validation_file.c_str(), &validation_labels);
    if (validation_labels.n_cols()==1) {
      Matrix labels1;
      la::TransposeInit(validation_labels, &labels1);
      validation_labels.Destruct();
      validation_labels.Own(&labels1);
    }

    for(index_t i=0; i<classification_results->n_cols(); i++) {
      if (classification_results->get(0, i) == validation_labels.get(0, i)) {
        total_hits+=1;
      }
    }
  }
  double total_score=100.0*total_hits/classification_results->n_cols();
  return total_score;
}
 
