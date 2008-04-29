#include <fastlib/fastlib.h>
#include "allknn.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  const char *ref_data = fx_param_str(NULL, "R", "my_ref_data.data");
  const char *q_data = fx_param_str(NULL, "Q", "my_qry_data.data");

  Matrix r_set, q_set;
  data::Load(ref_data, &r_set);
  data::Load(q_data, &q_set);

  AllKNN allknn;
  ArrayList<double> neighbor_distances;
  ArrayList<index_t> neighbor_indices;

  datanode *allknn_module = fx_submodule(NULL, "allknn", "allknn");
  index_t knn = fx_param_int(allknn_module, "knns",1);
  allknn.Init(q_set, r_set, allknn_module);

  fx_timer_start(allknn_module, "computing_neighbors");
  allknn.ComputeNeighbors(&neighbor_indices, &neighbor_distances);
  fx_timer_stop(allknn_module, "computing_neighbors");

  DEBUG_ASSERT(q_set.n_cols() * knn == neighbor_indices.size());
  for (index_t i = 0; i < q_set.n_cols(); i++) {
    NOTIFY("%"LI"d :", i);
    for(index_t j = 0; j < knn; j++) {
      NOTIFY("\t%"LI"d : %lf", 
	     neighbor_indices[knn*i+j], neighbor_distances[i*knn
							   +knn-1
							   -j]);
    }
  }

  fx_silence();
  fx_done();

  return 0;
}
