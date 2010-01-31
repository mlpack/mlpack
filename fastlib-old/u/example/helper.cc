/**
 * @file helper.cc
 *
 * Implementation for the k-nearest-neighbors classifier.
 */

#include "helper.h"

void KnnClassifier::InitTrain(const Dataset& dataset, int n_classes,
    datanode *module) {
  DEBUG_ASSERT_MSG(n_classes == 2, "This is only a binary classifier");
  matrix_.Copy(dataset.matrix());
  k_ = fx_param_int(module, "k", 5);     // sets default parameter for k
  DEBUG_ASSERT_MSG(k_ < matrix_.n_cols(), "knn/k must be smaller than dataset!");
  DEBUG_ASSERT_MSG(k_ % 2 == 1, "knn/k must not be even!");
}

int KnnClassifier::Classify(const Vector& test_datum) {
  ArrayList<double> distances; // Candidate distances, farthest first
  ArrayList<int> classes;      // The classes corresponding to each
  
  distances.Init(k_);
  classes.Init(k_);
  
  for (int k_i = 0; k_i < k_; k_i++) {
    distances[k_i] = DBL_MAX;
    classes[k_i] = 2; // initialize to an invalid class, to check later
  }
  
  for (index_t point = 0; point < matrix_.n_cols(); point++) {
    Vector training_datum_with_label;
    Vector training_datum;
    double dist_squared;
    
    // Get the proper column from the dataset, but we have to remove the
    // test label.
    matrix_.MakeColumnVector(point, &training_datum_with_label);
    training_datum_with_label.MakeSubvector(0, matrix_.n_rows() - 1,
        &training_datum);
    dist_squared = la::DistanceSqEuclidean(test_datum, training_datum);
    DEBUG_ASSERT_MSG(!isnan(dist_squared), "Cannot handle missing data");
    
    if (unlikely(dist_squared < distances[0])) {
      int k_i = 0;
      
      // Bump the nearest neighbors found
      while (k_i + 1 < k_ && dist_squared < distances[k_i + 1]) {
        distances[k_i] = distances[k_i + 1];
        classes[k_i] = classes[k_i + 1];
        k_i++;
      }
      
      distances[k_i] = dist_squared;
      classes[k_i] = int(training_datum_with_label[matrix_.n_rows() - 1]);
    }
  }
  
  int count_positive = 0;
  int count_negative = 0;
  
  for (int k_i = 0; k_i < k_; k_i++) {
    DEBUG_ASSERT(classes[k_i] == 0 || classes[k_i] == 1); // Debug check
    if (classes[k_i] == 0) {
      count_negative++;
    } else {
      count_positive++;
    }
  }
  
  return count_positive > count_negative ? 1 : 0;
}
