#include <fastlib/fastlib.h>
#include "allknn.h"
#include <time.h>

int main(int argc, char *argv[]) {

  fx_init(argc, argv, NULL);

  clock_t start, end, build, build1, find = 0, find1 = 0;
  const char *ref_data = fx_param_str(NULL, "R", "my_ref_data.data");
  const char *q_data = fx_param_str(NULL, "Q", "my_qry_data.data");

  Matrix r_set, q_set;
  data::Load(ref_data, &r_set);
  data::Load(q_data, &q_set);

  GenMatrix<float> queries, references;
  //GenMatrix<double> queries, references;
  queries.Init(q_set.n_rows(), q_set.n_cols());
  references.Init(r_set.n_rows(), r_set.n_cols());

  for (index_t i = 0; i < q_set.n_cols(); i++) {
    for(index_t j = 0; j < q_set.n_rows(); j++) {
      queries.set(j, i, (float) q_set.get(j, i));
    }
  }

  for (index_t i = 0; i < r_set.n_cols(); i++) {
    for(index_t j = 0; j < r_set.n_rows(); j++) {
      references.set(j, i, (float) r_set.get(j, i));
    }
  }


  //NOTIFY("%"LI"d , %"LI"d", q_set.n_rows(), q_set.n_cols());
  AllKNN<float> allknn;
  ArrayList<float> neighbor_distances, new_neighbor_distances;
  ArrayList<index_t> neighbor_indices, new_neighbor_indices;

  datanode *allknn_module = fx_submodule(NULL, "allknn");
  index_t knn = fx_param_int(allknn_module, "knns",1);
  
  start = clock();
  allknn.Init(queries, references, allknn_module);
  end = clock();
  
  build = end - start;
  /*
  fx_timer_start(allknn_module, "computing_neighbors");
  start = clock();
  allknn.ComputeNeighbors(&neighbor_indices, &neighbor_distances);
  end = clock();
  fx_timer_stop(allknn_module, "computing_neighbors");
  
  find = end - start;
  */
  NOTIFY("Phase 1 complete");

  
  start = clock();
  allknn.MakeCCoverTrees();
  end = clock();
  build1 = end - start;
  
  /*
  NOTIFY("Phase II complete");
  start = clock();
  allknn.ComputeNeighborsNew(&new_neighbor_indices, &new_neighbor_distances);
  end = clock();
  find1 = end - start;
  */
  NOTIFY("done");
  /*
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
  */
  float b = (float)build / (float)CLOCKS_PER_SEC;
  float f = (float)find / (float)CLOCKS_PER_SEC;
  printf("build = %f, find = %f\n", b, f);
  
  float b1 = (float)build1 / (float)CLOCKS_PER_SEC;
  float f1 = (float)find1 / (float)CLOCKS_PER_SEC;
  printf("build = %f, find = %f\n", b1, f1);
  

  fx_param_bool(NULL, "fx/silent", 1);
  fx_done(NULL);

  return 0;
}


/*
mnist_1_small X 2 = 6.1 + 16.74
mnist_1_small X 2 5knn = 6.14 + 43.61
mnist_1_small mnist_1_large = 92.29 + 238.86995
*/
