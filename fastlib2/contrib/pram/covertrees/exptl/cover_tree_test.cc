#include <fastlib/fastlib.h>
#include "allknn.h"
#include <time.h>

void run_allknn(char *ref_data, char *q_data);

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  //const char **ref_data;
  //ref_data = (const char **)malloc(4 * sizeof(const char *));
  const char *ref_data_0 = fx_param_str(NULL, "R1", "my_ref_data.data");
  const char *ref_data_1 = fx_param_str(NULL, "R2", "my_ref_data.data");
  const char *ref_data_2 = fx_param_str(NULL, "R3", "my_ref_data.data");
  const char *ref_data_3 = fx_param_str(NULL, "R4", "my_ref_data.data");

  //const char **q_data;
  //q_data = (const char **) malloc (4 * sizeof(const char *));
  const char *q_data_0 = fx_param_str(NULL, "Q1", "my_qry_data.data");
  const char *q_data_1 = fx_param_str(NULL, "Q2", "my_qry_data.data");
  const char *q_data_2 = fx_param_str(NULL, "Q3", "my_qry_data.data");
  const char *q_data_3 = fx_param_str(NULL, "Q4", "my_qry_data.data");

  clock_t start, end, build, find;
  float b, f;
  GenMatrix<float> queries, references;
  Matrix r_set, q_set;
  index_t flag = 1;
  index_t knn;

  NOTIFY("-------------------------DATASET %s------------------------", ref_data_0);
  data::Load(ref_data_0, &r_set);
  data::Load(q_data_0, &q_set);

  //GenMatrix<double> queries, references;
  queries.Init(q_set.n_rows(), q_set.n_cols());
  references.Init(r_set.n_rows(), r_set.n_cols());

  fx_timer_start(NULL, "conversion_to_float");
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
  fx_timer_stop(NULL, "conversion_to_float");

  //NOTIFY("%"LI"d , %"LI"d", q_set.n_rows(), q_set.n_cols());
  NOTIFY("D = %"LI"d", q_set.n_rows());
  NOTIFY("|R| = %"LI"d, |Q| = %"LI"d", r_set.n_cols(), q_set.n_cols());
  AllKNN<float> allknn_normal, allknn_cluster;
  ArrayList<float> neighbor_distances_n, neighbor_distances_n_c, 
    neighbor_distances_c, neighbor_distances_c_c;
  ArrayList<index_t> neighbor_indices_n, neighbor_indices_n_c, 
    neighbor_indices_c, neighbor_indices_c_c;

  datanode *allknn_module_n = fx_submodule(NULL, "allknn_n", "allknn_n");
  knn = fx_param_int(allknn_module_n, "knns",1);
  
  start = clock();
  fx_timer_start(allknn_module_n, "normal_cover_tree");
  allknn_normal.Init(queries, references, allknn_module_n);
  fx_timer_stop(allknn_module_n, "normal_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Normal Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_n, "computing_neighbors");
  start = clock();
  allknn_normal.ComputeNeighbors(&neighbor_indices_n, &neighbor_distances_n);
  end = clock();
  fx_timer_stop(allknn_module_n, "computing_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Normal Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_n, "centroid_cover_tree");
  allknn_normal.MakeCCoverTrees();
  fx_timer_stop(allknn_module_n, "centroid_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_n, "computing_centroid_neighbors");
  allknn_normal.ComputeNeighborsNew(&neighbor_indices_n_c,
				    &neighbor_distances_n_c);
  fx_timer_stop(allknn_module_n, "computing_centroid_neighbors");
  end = clock();

  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;

  NOTIFY("Centroid Tree Search Complete in %lf seconds", f);

  NOTIFY("Phase I over ...");


  datanode *allknn_module_c = fx_submodule(NULL, "allknn_c", "allknn_c");
  knn = fx_param_int(allknn_module_c, "knns",1);

  start = clock();
  fx_timer_start(allknn_module_c, "cluster_cover_tree");
  allknn_cluster.Init(queries, references, allknn_module_c, &flag);
  fx_timer_stop(allknn_module_c, "cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_c, "computing_cluster_neighbors");
  start = clock();
  allknn_cluster.ComputeNeighbors(&neighbor_indices_c, &neighbor_distances_c);
  end = clock();
  fx_timer_stop(allknn_module_c, "computing_cluster_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Cluster Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_c, "centroid_cluster_cover_tree");
  allknn_cluster.MakeCCoverTrees(&flag);
  fx_timer_stop(allknn_module_c, "centroid_cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("CLuster Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_c, "computing_cluster_centroid_neighbors");
  allknn_cluster.ComputeNeighborsNew(&neighbor_indices_c_c,
				     &neighbor_distances_c_c);
  fx_timer_stop(allknn_module_c, "computing_cluster_centroid_neighbors");
  end = clock();
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Centroid Tree Search Complete in %lf seconds", f);
  
  NOTIFY("Phase II over ...");
  
  q_set.Destruct();
  r_set.Destruct();
  queries.Destruct();
  references.Destruct();

  NOTIFY("----------------------DATASET %s-----------------------", ref_data_1);

  data::Load(ref_data_1, &r_set);
  data::Load(q_data_1, &q_set);

  //GenMatrix<double> queries, references;
  queries.Init(q_set.n_rows(), q_set.n_cols());
  references.Init(r_set.n_rows(), r_set.n_cols());

  fx_timer_start(NULL, "conversion_to_float");
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
  fx_timer_stop(NULL, "conversion_to_float");

  //NOTIFY("%"LI"d , %"LI"d", q_set.n_rows(), q_set.n_cols());
  NOTIFY("D = %"LI"d", q_set.n_rows());
  NOTIFY("|R| = %"LI"d, |Q| = %"LI"d", r_set.n_cols(), q_set.n_cols());
  AllKNN<float> allknn_normal1, allknn_cluster1;
  ArrayList<float> neighbor_distances_n1, neighbor_distances_n1_c, 
    neighbor_distances_c1, neighbor_distances_c1_c;
  ArrayList<index_t> neighbor_indices_n1, neighbor_indices_n1_c, 
    neighbor_indices_c1, neighbor_indices_c1_c;

  datanode *allknn_module_n1 = fx_submodule(NULL, "allknn_n1", "allknn_n1");
  knn = fx_param_int(allknn_module_n1, "knns",1);
  
  start = clock();
  fx_timer_start(allknn_module_n1, "normal_cover_tree");
  allknn_normal1.Init(queries, references, allknn_module_n1);
  fx_timer_stop(allknn_module_n1, "normal_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Normal Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_n1, "computing_neighbors");
  start = clock();
  allknn_normal1.ComputeNeighbors(&neighbor_indices_n1, &neighbor_distances_n1);
  end = clock();
  fx_timer_stop(allknn_module_n1, "computing_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Normal Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_n1, "centroid_cover_tree");
  allknn_normal1.MakeCCoverTrees();
  fx_timer_stop(allknn_module_n1, "centroid_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_n1, "computing_centroid_neighbors");
  allknn_normal1.ComputeNeighborsNew(&neighbor_indices_n1_c,
				    &neighbor_distances_n1_c);
  fx_timer_stop(allknn_module_n1, "computing_centroid_neighbors");
  end = clock();

  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;

  NOTIFY("Centroid Tree Search Complete in %lf seconds", f);

  NOTIFY("Phase I over ...");


  datanode *allknn_module_c1 = fx_submodule(NULL, "allknn_c1", "allknn_c1");
  knn = fx_param_int(allknn_module_c1, "knns",1);
  flag = 1;

  start = clock();
  fx_timer_start(allknn_module_c1, "cluster_cover_tree");
  allknn_cluster1.Init(queries, references, allknn_module_c1, &flag);
  fx_timer_stop(allknn_module_c1, "cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_c1, "computing_cluster_neighbors");
  start = clock();
  allknn_cluster1.ComputeNeighbors(&neighbor_indices_c1, &neighbor_distances_c1);
  end = clock();
  fx_timer_stop(allknn_module_c1, "computing_cluster_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Cluster Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_c1, "centroid_cluster_cover_tree");
  allknn_cluster1.MakeCCoverTrees(&flag);
  fx_timer_stop(allknn_module_c1, "centroid_cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("CLuster Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_c1, "computing_cluster_centroid_neighbors");
  allknn_cluster1.ComputeNeighborsNew(&neighbor_indices_c1_c,
				     &neighbor_distances_c1_c);
  fx_timer_stop(allknn_module_c1, "computing_cluster_centroid_neighbors");
  end = clock();
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Centroid Tree Search Complete in %lf seconds", f);
  
  NOTIFY("Phase II over ...");

  q_set.Destruct();
  r_set.Destruct();
  queries.Destruct();
  references.Destruct();

  NOTIFY("----------------------DATASET %s-----------------------", ref_data_2);

  data::Load(ref_data_2, &r_set);
  data::Load(q_data_2, &q_set);

  //GenMatrix<double> queries, references;
  queries.Init(q_set.n_rows(), q_set.n_cols());
  references.Init(r_set.n_rows(), r_set.n_cols());

  fx_timer_start(NULL, "conversion_to_float");
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
  fx_timer_stop(NULL, "conversion_to_float");

  //NOTIFY("%"LI"d , %"LI"d", q_set.n_rows(), q_set.n_cols());
  NOTIFY("D = %"LI"d", q_set.n_rows());
  NOTIFY("|R| = %"LI"d, |Q| = %"LI"d", r_set.n_cols(), q_set.n_cols());
  AllKNN<float> allknn_normal2, allknn_cluster2;
  ArrayList<float> neighbor_distances_n2, neighbor_distances_n2_c, 
    neighbor_distances_c2, neighbor_distances_c2_c;
  ArrayList<index_t> neighbor_indices_n2, neighbor_indices_n2_c, 
    neighbor_indices_c2, neighbor_indices_c2_c;

  datanode *allknn_module_n2 = fx_submodule(NULL, "allknn_n2", "allknn_n2");
  knn = fx_param_int(allknn_module_n2, "knns",1);
  
  start = clock();
  fx_timer_start(allknn_module_n2, "normal_cover_tree");
  allknn_normal2.Init(queries, references, allknn_module_n2);
  fx_timer_stop(allknn_module_n2, "normal_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Normal Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_n2, "computing_neighbors");
  start = clock();
  allknn_normal2.ComputeNeighbors(&neighbor_indices_n2, &neighbor_distances_n2);
  end = clock();
  fx_timer_stop(allknn_module_n2, "computing_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Normal Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_n2, "centroid_cover_tree");
  allknn_normal2.MakeCCoverTrees();
  fx_timer_stop(allknn_module_n2, "centroid_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_n2, "computing_centroid_neighbors");
  allknn_normal2.ComputeNeighborsNew(&neighbor_indices_n2_c,
				    &neighbor_distances_n2_c);
  fx_timer_stop(allknn_module_n2, "computing_centroid_neighbors");
  end = clock();

  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;

  NOTIFY("Centroid Tree Search Complete in %lf seconds", f);

  NOTIFY("Phase I over ...");


  datanode *allknn_module_c2 = fx_submodule(NULL, "allknn_c2", "allknn_c2");
  knn = fx_param_int(allknn_module_c2, "knns",1);
  flag = 1;

  start = clock();
  fx_timer_start(allknn_module_c2, "cluster_cover_tree");
  allknn_cluster2.Init(queries, references, allknn_module_c2, &flag);
  fx_timer_stop(allknn_module_c2, "cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_c2, "computing_cluster_neighbors");
  start = clock();
  allknn_cluster2.ComputeNeighbors(&neighbor_indices_c2, &neighbor_distances_c2);
  end = clock();
  fx_timer_stop(allknn_module_c2, "computing_cluster_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Cluster Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_c2, "centroid_cluster_cover_tree");
  allknn_cluster2.MakeCCoverTrees(&flag);
  fx_timer_stop(allknn_module_c2, "centroid_cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("CLuster Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_c2, "computing_cluster_centroid_neighbors");
  allknn_cluster2.ComputeNeighborsNew(&neighbor_indices_c2_c,
				     &neighbor_distances_c2_c);
  fx_timer_stop(allknn_module_c2, "computing_cluster_centroid_neighbors");
  end = clock();
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Centroid Tree Search Complete in %lf seconds", f);
  
  NOTIFY("Phase II over ...");


  q_set.Destruct();
  r_set.Destruct();
  queries.Destruct();
  references.Destruct();

  NOTIFY("----------------------DATASET %s-----------------------", ref_data_3);

  data::Load(ref_data_3, &r_set);
  data::Load(q_data_3, &q_set);

  //GenMatrix<double> queries, references;
  queries.Init(q_set.n_rows(), q_set.n_cols());
  references.Init(r_set.n_rows(), r_set.n_cols());

  fx_timer_start(NULL, "conversion_to_float");
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
  fx_timer_stop(NULL, "conversion_to_float");

  //NOTIFY("%"LI"d , %"LI"d", q_set.n_rows(), q_set.n_cols());
  NOTIFY("D = %"LI"d", q_set.n_rows());
  NOTIFY("|R| = %"LI"d, |Q| = %"LI"d", r_set.n_cols(), q_set.n_cols());
  AllKNN<float> allknn_normal3, allknn_cluster3;
  ArrayList<float> neighbor_distances_n3, neighbor_distances_n3_c, 
    neighbor_distances_c3, neighbor_distances_c3_c;
  ArrayList<index_t> neighbor_indices_n3, neighbor_indices_n3_c, 
    neighbor_indices_c3, neighbor_indices_c3_c;

  datanode *allknn_module_n3 = fx_submodule(NULL, "allknn_n3", "allknn_n3");
  knn = fx_param_int(allknn_module_n3, "knns",1);
  
  start = clock();
  fx_timer_start(allknn_module_n3, "normal_cover_tree");
  allknn_normal3.Init(queries, references, allknn_module_n3);
  fx_timer_stop(allknn_module_n3, "normal_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Normal Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_n3, "computing_neighbors");
  start = clock();
  allknn_normal3.ComputeNeighbors(&neighbor_indices_n3, &neighbor_distances_n3);
  end = clock();
  fx_timer_stop(allknn_module_n3, "computing_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Normal Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_n3, "centroid_cover_tree");
  allknn_normal3.MakeCCoverTrees();
  fx_timer_stop(allknn_module_n3, "centroid_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_n3, "computing_centroid_neighbors");
  allknn_normal3.ComputeNeighborsNew(&neighbor_indices_n3_c,
				    &neighbor_distances_n3_c);
  fx_timer_stop(allknn_module_n3, "computing_centroid_neighbors");
  end = clock();

  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;

  NOTIFY("Centroid Tree Search Complete in %lf seconds", f);

  NOTIFY("Phase I over ...");


  datanode *allknn_module_c3 = fx_submodule(NULL, "allknn_c3", "allknn_c3");
  knn = fx_param_int(allknn_module_c3, "knns",1);
  flag = 1;

  start = clock();
  fx_timer_start(allknn_module_c3, "cluster_cover_tree");
  allknn_cluster3.Init(queries, references, allknn_module_c3, &flag);
  fx_timer_stop(allknn_module_c3, "cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Tree built in %lf seconds", b);
  
  fx_timer_start(allknn_module_c3, "computing_cluster_neighbors");
  start = clock();
  allknn_cluster3.ComputeNeighbors(&neighbor_indices_c3, &neighbor_distances_c3);
  end = clock();
  fx_timer_stop(allknn_module_c3, "computing_cluster_neighbors");
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  NOTIFY("Cluster Tree Search Complete in %lf seconds", f);

   
  start = clock();
  fx_timer_start(allknn_module_c3, "centroid_cluster_cover_tree");
  allknn_cluster3.MakeCCoverTrees(&flag);
  fx_timer_stop(allknn_module_c3, "centroid_cluster_cover_tree");
  end = clock();
  
  build = end - start;
  b = (float)build / (float)CLOCKS_PER_SEC;
 
  NOTIFY("CLuster Centroid Tree Built in %lf seconds", b);

  start = clock();  
  fx_timer_start(allknn_module_c3, "computing_cluster_centroid_neighbors");
  allknn_cluster3.ComputeNeighborsNew(&neighbor_indices_c3_c,
				     &neighbor_distances_c3_c);
  fx_timer_stop(allknn_module_c3, "computing_cluster_centroid_neighbors");
  end = clock();
  
  find = end - start;
  f = (float)find / (float)CLOCKS_PER_SEC;
  
  NOTIFY("Cluster Centroid Tree Search Complete in %lf seconds", f);
  
  NOTIFY("Phase II over ...");


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

  //fx_silence();
  fx_done();

  return 0;
}


/*
mnist_1_small X 2 = 6.1 + 16.74
mnist_1_small X 2 5knn = 6.14 + 43.61
mnist_1_small mnist_1_large = 92.29 + 238.86995
*/
