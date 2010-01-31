/**
 * @file helper.h
 *
 * A simple K-nearest-neighbors classifier.
 */

#ifndef U_EXAMPLE_HELPER
#define U_EXAMPLE_HELPER

#include "fastlib/fastlib.h"

/**
 * Simple binary K-nearest-neighbors classifier.
 */
class KnnClassifier {
 private:
   /** Our model *is* the data. */
   Matrix matrix_;
   /** Number of nearest neighbors to look at. */
   int k_;
   
 public:
   /**
    * Trains and initializes on a data set.
    *
    * Assumes that the last column is the class, and the other columns
    * are just the data.
    */
   void InitTrain(const Dataset& dataset, int n_classes, datanode *module);
   
   /**
    * Classifies a vector as an integer label.
    */
   int Classify(const Vector& test_datum);
};

#endif
