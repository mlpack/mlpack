#include "fastlib/fastlib.h"
#include "do_mcf.h"
#include "la_augment.h"

template<typename T>
void FindFarthestPoint(int n_points,
		       const ArrayList<GenMatrix<T> > &datasets,
		       const ArrayList<GenVector<T> > &cluster_centers,
		       const GenVector<int> &cluster_memberships,
		       int* p_argmax_dataset_index,
		       int* p_argmax_point_index_in_dataset) {
  
  int &argmax_dataset_index = *p_argmax_dataset_index;
  int &argmax_point_index_in_dataset = *p_argmax_point_index_in_dataset;
  

  int n_datasets = datasets.size();
  int n_dims = datasets[0].n_rows();

  double max_dist = -std::numeric_limits<double>::max();
  argmax_dataset_index = -1;
  argmax_point_index_in_dataset = -1;

  int i = 0;
  for(int m = 0; m < n_datasets; m++) {
    int n_points_in_dataset = datasets[m].n_cols();
    for(int j = 0; j < n_points_in_dataset; j++) {
      double dist =
	la::DistanceSqEuclidean(n_dims,
				datasets[m].GetColumnPtr(j),
				cluster_centers[cluster_memberships[i]].ptr());
      if(dist > max_dist) {
	max_dist = dist;
	argmax_dataset_index = m;
	argmax_point_index_in_dataset = j;
      }
      i++;
    }
  }
}



bool CheckFileWritten(const char* problem_filename, int iteration_num) {
  FILE* file = fopen(problem_filename, "r");
  int read_iteration_num;
  fscanf(file, "c %d", &read_iteration_num);
  fclose(file);
  return (read_iteration_num == iteration_num);
}
  

void WriteProblem(const char* filename, const Matrix &distances_sq,
		  int min_points_per_cluster, int iteration_num) {
  int n_clusters = distances_sq.n_rows();
  int n_points = distances_sq.n_cols();
  
  //const char* filename = "clustering_problem";
  FILE* file = fopen(filename, "w");
 

  int n_nodes = n_points + n_clusters + 1;
  int n_arcs = (n_points * n_clusters) + n_clusters;

  int cluster_ids[n_clusters];
  for(int j = 0; j < n_clusters; j++) {
    cluster_ids[j] = n_points + j + 1;
  }

  int sink_id = n_points + n_clusters + 1;
 
  fprintf(file, "c %d\n", iteration_num);
  fprintf(file, "p min %d %d\n", n_nodes, n_arcs);
  for(int i = 0; i < n_points; i++) {
    fprintf(file, "n %d 1\n", i + 1);
  }
  for(int j = 0; j < n_clusters; j++) {
    fprintf(file, "n %d %d\n", cluster_ids[j], -min_points_per_cluster);
  }
  fprintf(file, "n %d %d\n",
	  sink_id, -n_points + (n_clusters * min_points_per_cluster));
  for(int i = 0; i < n_points; i++) {
    for(int j = 0; j < n_clusters; j++) {
      fprintf(file, "a %d %d 0 1 %f\n",
	      i + 1, cluster_ids[j],
	      distances_sq.get(j, i));
    }
  }
  for(int j = 0; j < n_clusters; j++) {
    fprintf(file, "a %d %d 0 free 0\n",
	    cluster_ids[j], sink_id);
  }

  fclose(file);
}

// p_cluster_memberships must point to an Init'd Vector
void ReadSolution(const char* filename, GenVector<int>* p_cluster_memberships,
		  int* p_n_changes) {
  GenVector<int> &cluster_memberships = *p_cluster_memberships;
  int &n_changes = *p_n_changes;
  
  int n_points = cluster_memberships.length();

  FILE* file = fopen(filename, "r");
  
  char* buffer = (char*) malloc(80 * sizeof(char));
  for(int i = 0; i < 5; i++) {
    fgets(buffer, 80, file);
  }
  free(buffer);

  int source_id;
  int dest_id;
  int flow;
  int n_points_plus_1 = n_points + 1;
  n_changes = 0;
  for(int i = 0; i < n_points; i++) {
    fscanf(file, "f %d %d %d\n", &source_id, &dest_id, &flow);
    int point_num = source_id - 1;
    int cluster_num = dest_id - n_points_plus_1;
    if(cluster_memberships[point_num] != cluster_num) {
      n_changes++;
      cluster_memberships[point_num] = cluster_num;
    }
    cluster_memberships[source_id - 1] = dest_id - n_points_plus_1;
  }
  fclose(file);

/*   for(int i = 0; i < n_points; i++) { */
/*     printf("%d ", cluster_memberships[i]); */
/*   } */
/*   printf("\n"); */
}

template<typename T>
void ConstrainedKMeans(const GenMatrix<T> &dataset,
		       int n_clusters,
		       int max_iterations,
		       int min_points_per_cluster,
		       const char* problem_filename,
		       const char* solution_filename,
		       GenVector<int>* p_cluster_memberships,
		       int cluster_counts[]) {
  ArrayList<GenMatrix<T> > datasets;
  datasets.Init(1);
  datasets[0].Alias(dataset);
  ConstrainedKMeans(datasets,
		    n_clusters,
		    max_iterations,
		    min_points_per_cluster,
		    problem_filename,
		    solution_filename,
		    p_cluster_memberships,
		    cluster_counts);
}



template<typename T>
void ConstrainedKMeans(const ArrayList<GenMatrix<T> > &datasets,
		       int n_clusters,
		       int max_iterations,
		       int min_points_per_cluster,
		       const char* problem_filename,
		       const char* solution_filename,
		       GenVector<int>* p_cluster_memberships,
		       int cluster_counts[]) {
  GenVector<int> &cluster_memberships= *p_cluster_memberships;

  int n_dims = datasets[0].n_rows();
    
  ArrayList<GenVector<T> > cluster_centers;
  cluster_centers.Init(n_clusters);
  for(int k = 0; k < n_clusters; k++) {
    cluster_centers[k].Init(n_dims);
  }
    
  int n_datasets = datasets.size();
  int n_points = 0;
  for(int m = 0; m < n_datasets; m++) {
    n_points += datasets[m].n_cols();
  }
  cluster_memberships.Init(n_points);

  /* Gonzales algorithm to initialize clusters */
  int current_n_clusters = 1;
  cluster_centers[0].CopyValues(datasets[0].GetColumnPtr(0));
  for(int i = 0; i < n_points; i++) {
    // assign all points the single existing cluster
    cluster_memberships[i] = 0;
  }
  
  while(current_n_clusters < n_clusters) {
    int farthest_dataset;
    int farthest_point;
    FindFarthestPoint(n_points, datasets,
		      cluster_centers, cluster_memberships,
		      &farthest_dataset,
		      &farthest_point);
    cluster_centers[current_n_clusters].
      CopyValues(datasets[farthest_dataset].GetColumnPtr(farthest_point));
    current_n_clusters++;

    // update cluster memberships using cluster centers
    int i = 0;
    for(int m = 0; m < n_datasets; m++) {
      const GenMatrix<T> &data = datasets[m];
      int n_points_in_data = data.n_cols();
      for(int j = 0; j < n_points_in_data; j++) {
	GenVector<T> point;
	data.MakeColumnVector(j, &point);
	double min_dist_sq = std::numeric_limits<double>::max();
	int nearest_cluster_index = -1;
	for(int k = 0; k < current_n_clusters; k++) {
	  double dist_sq =
	    la::DistanceSqEuclidean(point, cluster_centers[k]);
	  if(dist_sq < min_dist_sq) {
	    min_dist_sq = dist_sq;
	    nearest_cluster_index = k;
	  }
	}
	cluster_memberships[i] = nearest_cluster_index;
	i++;
      }
    }
  }

  
/*   for(int i = 0; i < n_clusters; i++) { */
/*     printf("cluster center %d\n", i); */
/*     cluster_centers[i].PrintDebug(""); */
/*   } */

  // the clusters are now initialized using the Gonzales algorithm

  int iteration_num = 1;
  int n_changes = 0;
  bool converged = false;
  while(!converged) {
    // update cluster centers using cluster memberships
    for(int k = 0; k < n_clusters; k++) {
      cluster_centers[k].SetZero();
      cluster_counts[k] = 0;
    }
    int i = 0;
    for(int m = 0; m < n_datasets; m++) {
      const GenMatrix<T> &data = datasets[m];
      int n_points_in_data = data.n_cols();
      for(int j = 0; j < n_points_in_data; j++) {
	GenVector<T> point;
	data.MakeColumnVector(j, &point);
	int cluster_index = cluster_memberships[i];
	la::AddTo(point, &(cluster_centers[cluster_index]));
	cluster_counts[cluster_index]++;
	i++;
      }
    }
    for(int k = 0; k < n_clusters; k++) {
      la::Scale(((double)1) / ((double)(cluster_counts[k])),
		&(cluster_centers[k]));
    }
   
    // update cluster memberships using cluster centers
    Matrix distances_sq;
    distances_sq.Init(n_clusters, n_points);
    i = 0;
    for(int m = 0; m < n_datasets; m++) {
      const GenMatrix<T> &data = datasets[m];
      int n_points_in_data = data.n_cols();
      for(int j = 0; j < n_points_in_data; j++) {
	GenVector<T> point;
	data.MakeColumnVector(j, &point);
	for(int k = 0; k < n_clusters; k++) {
	  distances_sq.set(k, i, 
			   la::DistanceSqEuclidean(point,
						   cluster_centers[k]));
	}
	i++;
      }
    }
    
    WriteProblem(problem_filename, distances_sq, min_points_per_cluster, iteration_num);
    
    /* // not necessary!
    if(!CheckFileWritten(problem_filename, iteration_num)) {
      printf("Not calling WriteProblem() becuase file not yet written!\n");
      exit(1);
    }
    */ 

    DoMCF(problem_filename, solution_filename);
    ReadSolution(solution_filename, &cluster_memberships, &n_changes);

    printf("n_changes = %d\n", n_changes);

    iteration_num++;
    if((n_changes == 0) || (iteration_num > max_iterations)) {
      converged = true;
    }
  }
  printf("clustering optimizer converged in %d iterations\n", iteration_num);

  for(int i = 0; i < n_clusters; i++) {
    cluster_counts[i] = 0;
  }
  for(int i = 0; i < n_points; i++) {
    cluster_counts[cluster_memberships[i]]++;
  }

}
