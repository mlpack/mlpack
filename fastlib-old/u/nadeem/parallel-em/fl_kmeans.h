#ifndef FL_KMEANS_H
#define FL_KMEANS_H

#include "fastlib/fastlib.h"

#define USE_FULL_COV_MAT 0
#define COV_MAT_MIN_DIAG 0.001
#define NUM_KMEANS_REPEATS 10

bool gonzalez_cluster(Matrix const &data, int num_clusters, ArrayList<Vector> 
		      *centroids_);

double kmeans(Matrix const &data, int num_clusters, 
		   ArrayList<int>* labels, ArrayList<Vector>* centroids, 
		   int max_iter=500, double error_thresh=1e-04, bool seed_gonzalez=true);

int kmeans_classify(Vector const &data_t, ArrayList<Vector> 
			   const &centroids, double* error);

int kmeans_CV(Dataset const &data, int folds, int min_k, int max_k);

void compute_kmeans_cov_mat(Matrix const &data, 
			       ArrayList<Vector> const &centroids, 
			       ArrayList<int> const &data_labels, 
			       ArrayList<Matrix> *cov_mats);
#endif
