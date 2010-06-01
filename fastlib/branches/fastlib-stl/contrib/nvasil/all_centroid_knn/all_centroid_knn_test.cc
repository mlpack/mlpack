/*
 * =====================================================================================
 *
 *       Filename:  all_centroid_knn_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/05/2008 10:21:52 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "all_centroid_knn.h"

class TestAllCentroidkNN {
 public:
  void Init() {
    engine_ = new AllCentroidkNN();
  }
  void Destruct() {
    delete engine_;
  }
  void Test1() {
    Init();
    Matrix points;
    data::Load("/net/hg200/nvasil/dataset/swiss_roll/swiss_roll_10000.csv" , &points);
    datanode *module=fx_submodule(NULL, "/", "temp");
    engine_->Init(points, NULL);
    NOTIFY("Number of centroids %i", engine_->centroid_counter_);
    Matrix centroids;
    NOTIFY("Computing centroids...\n");
    engine_->ComputeCentroids(&centroids);
    NOTIFY("Centroids computed...\n");
    ArrayList<index_t> centroid_ids;
    Matrix features;
    features.Init(points.n_rows(), points.n_cols());
    NOTIFY("Retrieving centroids ...");
    engine_->RetrieveCentroids(8, &centroid_ids, &features);
    NOTIFY("Centroids retrieved...");
    ArrayList<index_t> resulting_neighbors;
    NOTIFY("Computing all k centroid distances");
    ArrayList<double> distances;
    engine_->AllkCentroids(centroids, centroid_ids, &resulting_neighbors, 
      &distances); 
    NOTIFY("k-centroid distances computed");
    Destruct();
  }
  void TestAll() {
    Test1();
  }
 private:
  AllCentroidkNN *engine_;  
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  TestAllCentroidkNN test;
  test.TestAll();
  fx_done();
}

