#include "fl_kmeans.h"
#include <sys/time.h>

bool gonzalez_cluster(Matrix const &data, int num_clusters, ArrayList<Vector> 
		      *centroids_)
{
  //  fprintf(stderr, "Doing Gonzalez...");
  ArrayList<Vector> &centroids = *centroids_;
  int num_points = data.n_cols();
  double min_dist = DBL_MAX;
  double max_dist = -DBL_MAX;
  int index;
  Vector temp_vec;

  centroids.Init(num_clusters);

  index = rand() % num_points;
  //  fprintf(stderr, "Gonzalez Index=%d\n",index);

  data.MakeColumnVector(index, &(temp_vec));
  centroids[0].Copy(temp_vec);

  for (int i=1; i < num_clusters; i++) {

    max_dist = 0;
    for (int j=0; j < num_points; j++) {
      
      Vector data_v;
      data.MakeColumnVector(j, &data_v);
      min_dist = DBL_MAX;
      
      for (int k=0; k < i; k++) {
	double dist = la::DistanceSqEuclidean(data_v, centroids[k]);
	if (dist < min_dist)
	  min_dist = dist;
      }

      if (min_dist > max_dist) {
	max_dist = min_dist;
	index = j;
      }
    }

    Vector v;
    data.MakeColumnVector(index, &v);
    centroids[i].Copy(v);
  }

  //  fprintf(stderr, "done\n");

  return true;
}

/*----------------------------------------------------------------------------
INPUT : data, num_clusters, max_iter, error_thresh
RETURN: centroids_ - centroid of each the cluster,
        labels_ - label of each data point, i.e. the cluster it belongs to  
This method assumes that the individual elements in the ArrayList for labels 
and centroids are not initialized. It will allocate fresh memory for those. 
The "labels *" and "centroids *" should point to valid objects.
-----------------------------------------------------------------------------*/
double kmeans(Matrix const &data, int num_clusters, 
	    ArrayList<int> *labels_, ArrayList<Vector> *centroids_, 
	    int max_iter, double error_thresh, bool seed_gonzalez)
{
  if (data.n_cols() < num_clusters) 
    return -1;

  ArrayList<int> counts; // number of points in each cluster
  ArrayList<Vector> tmp_centroids; // intermediate centroids
  int num_points, num_dims;
  int i, j, num_iter=0;
  double error, old_error;

  //Assign pointers to references to avoid repeated dereferencing.
  ArrayList<int> &labels = *labels_;
  ArrayList<Vector> &centroids = *centroids_;

  num_points = data.n_cols();
  num_dims = data.n_rows();
  

  if (seed_gonzalez)
    gonzalez_cluster(data, num_clusters, centroids_);

  else {
    //Instead of calling Gonzalez, let's try to do random seeding 
    // and see what happens
    centroids.Init(num_clusters);
    for (int i=0; i < num_clusters; i++) {
      int index = rand() % num_points;
      Vector temp_vec;
      data.MakeColumnVector(index, &(temp_vec));
      centroids[i].Copy(temp_vec);
    }
    // End of random seeding code
  }

  //  fprintf(stderr,"Running K-Means...");
  

  //  centroids.Init(num_clusters);
  tmp_centroids.Init(num_clusters);

  counts.Init(num_clusters);
  labels.Init(num_points);

  //  srand ( time(NULL) );
  
  //Initialize the clusters to k points from the dataset
  for (j=0; j < num_clusters; j++)
    tmp_centroids[j].Init(num_dims);

  error = DBL_MAX;

  do {

    old_error = error; error = 0;

    for (i=0; i < num_clusters; i++) {
      tmp_centroids[i].SetZero();
          counts[i] = 0;
    }

    for (i=0; i < num_points; i++) {

      // Find the cluster closest to this point and update its label
      double min_distance = DBL_MAX;
      Vector data_i_Vec;
      data.MakeColumnVector(i, &data_i_Vec);
      
      for (j=0; j < num_clusters; j++ ) {
	double distance = la::DistanceSqEuclidean(data_i_Vec, centroids[j]);
	if (distance < min_distance) {
	  labels[i] = j;
	  min_distance = distance;
	}
      }

      // Accumulate the stats for the new centroid of the target cluster
      la::AddTo(data_i_Vec, &(tmp_centroids[labels[i]]));
      counts[labels[i]]++;
      error += min_distance;
    }

    // Now recompute all the centroids
    for (int j=0; j < num_clusters; j++) {
      if (counts[j] > 0)
	la::ScaleOverwrite((1/(double)counts[j]), 
			   tmp_centroids[j], &(centroids[j]));
    }

    num_iter++;

    //Keep repeating until we converge or run out of steam
  } while ((fabs(error - old_error) > error_thresh)
	   && (num_iter < max_iter));

  //  fprintf(stderr, "Done\n");

  return error;  
}

int kmeans_classify(Vector const &data_Vec, ArrayList<Vector> const &centroids,
		    double* error)
{
  int label=0;
  double min_distance = DBL_MAX;
  
  // Find the cluster closest to this point and update its label
  for (int j=0; j < centroids.size(); j++ ) {
    double distance = la::DistanceSqEuclidean(data_Vec, centroids[j]);
    if (distance < min_distance) {
      label = j;
      min_distance = distance;
    }
  }

  *error = min_distance;
  return label;
}

void compute_kmeans_cov_mat(Matrix const &data, 
			    ArrayList<Vector> const &centroids, 
			    ArrayList<int> const &data_labels, 
			    ArrayList<Matrix> *cov_mats)
{
  ArrayList<Matrix> &g_sigma = *cov_mats;
  int num_clusters = centroids.size();
  int num_dim = data.n_rows();
  int num_data = data.n_cols();
  Vector num_points;

  g_sigma.Init(num_clusters);
  num_points.Init(num_clusters);
  num_points.SetZero();

  for (int i=0; i < num_clusters; i++) {
    g_sigma[i].Init(num_dim, num_dim);
    g_sigma[i].SetZero();
  }

  for (int j=0; j < num_data; j++) {

    int i= data_labels[j];
    num_points[i]++;

    Matrix temp_cov;
    Vector data_j_Vec, sub_Vec;
    data.MakeColumnVector(j, &data_j_Vec);

    la::SubInit(centroids[i], data_j_Vec, &sub_Vec);
	
    // Now create a diagonal covariance matrix
    temp_cov.Init(sub_Vec.length(), sub_Vec.length());
    temp_cov.SetZero();
    for (int v=0; v<sub_Vec.length(); v++)
#if USE_FULL_COV_MAT 
      for (int w=0; w<sub_Vec.length(); w++)
      	temp_cov.set(v, w, sub_Vec[v] * sub_Vec[w]);
#else
      temp_cov.set(v, v, math::Sqr(sub_Vec[v]));
#endif
    
    la::AddTo(temp_cov, &(g_sigma[i]));
  }

  for (int i=0; i < num_clusters; i++) {
    if (num_points[i] > 0)
      la::Scale(1.0/(double)num_points[i], &(g_sigma[i]));  

    // Now add a small "regularisation" value to the diagonal elements
    for (int d=0; d < num_dim; d++)
      if (g_sigma[i].get(d, d) < COV_MAT_MIN_DIAG)
	g_sigma[i].set(d, d, COV_MAT_MIN_DIAG);   
  }
}

double compute_log_prob(Vector const &data_k_Vec, 
			ArrayList<Vector> const &centroids ,
			ArrayList <double> const &log_cov_dets, 
			ArrayList <Matrix> const &inv_cov_mats,
			Vector const &log_cltr_wt)
{
  double log_gauss_const = (data_k_Vec.length()) * log(2 * math::PI);
  Vector log_probs;
  double prob;
  int num_clusters = centroids.size();
  double max_val=-DBL_MAX;

  log_probs.Init(num_clusters);

  for (int p=0; p < num_clusters; p++) {
    // Subtract the mean vector from the current data vector
    Vector diffVector;
    la::SubInit(centroids[p], data_k_Vec, &diffVector);
    // Multiply the adjusted vector with the inverse of covariance matrix
    Vector resultVec;
    la::MulInit(diffVector, inv_cov_mats[p], &resultVec);   
    // Now compute the probability from multivariate Gaussian
    log_probs[p] = log_cltr_wt[p] + 
      (-0.5 * ((log_gauss_const + log_cov_dets[p]) + 
	       la::Dot(resultVec, diffVector)));

    if (max_val < log_probs[p])
      max_val = log_probs[p];
  }

  prob = 0;
  for (int p=0; p < num_clusters; p++)
    prob += exp(log_probs[p] - max_val); 

  return log(prob) + max_val;

}

int kmeans_CV(Dataset const &data, int folds, int min_k, int max_k)
{
  int best_k = min_k;
  double best_llh = -DBL_MAX;
  //  double new_llh = 0;
  double num_test_points;
  Matrix all_llh;

  //Seed the random number generator
  timeval t1;
  gettimeofday(&t1,NULL);
  srand(t1.tv_usec);

  all_llh.Init(max_k-min_k+1, NUM_KMEANS_REPEATS);
  all_llh.SetZero();

  for (int r=0; r < NUM_KMEANS_REPEATS; r++) {

    fprintf(stderr, "REPEAT %d:\n",r);

    //Make a new permutation for each repeat
    ArrayList<int> permutation;
    math::MakeRandomPermutation(data.n_points(), &permutation);

    for (int i=min_k; i<=max_k; i++) { // For each no. of clusters
      for (int j=0; j < folds; j++) {  // k-fold 
	
	Dataset train, test;
	ArrayList<int> labels;
	ArrayList<Vector> centroids;

	// Split the dataset and run kmeans on the training subset
	data.SplitTrainTest(folds, j, permutation, &train, &test);
	num_test_points = test.n_points();
	
	if (r < NUM_KMEANS_REPEATS/2)
	  kmeans(train.matrix(), i, &labels, &centroids);
	else
	  kmeans(train.matrix(), i, &labels, &centroids, false);
	
	// We will use "pseudo log-likelihood" to do cross-validation
	// Compute Cluster Weights
	Vector log_cltr_wt;
	log_cltr_wt.Init(i);
	log_cltr_wt.SetZero();
	
	for (int w=0; w < labels.size(); w++)
	  log_cltr_wt[labels[w]]++;

	for (int w=0; w < i; w++) 
	  log_cltr_wt[w] = log(log_cltr_wt[w]) - log(labels.size());
	
	// Compute the covariance matrix
	ArrayList <Matrix> cov_mats, inv_cov_mats;
	ArrayList <double> log_cov_dets;
	inv_cov_mats.Init(i);
	log_cov_dets.Init(i);
	
	compute_kmeans_cov_mat(train.matrix(), centroids,labels, &cov_mats);
	
	for (int a=0; a<i; a++) {
	  la::InverseInit(cov_mats[a], &(inv_cov_mats[a]));
	  log_cov_dets[a] = la::DeterminantLog(cov_mats[a],0);
	}
	
	// Now compute the log-likelihood
	double log_prob;
	double curr_fold_llh = 0;
	for (int k=0; k < num_test_points; k++) { //over all the test points
	  Vector data_k_Vec;
	  test.matrix().MakeColumnVector(k, &data_k_Vec);
	  
	  log_prob = compute_log_prob(data_k_Vec, centroids, log_cov_dets, 
				      inv_cov_mats, log_cltr_wt);
	  
	  curr_fold_llh += log_prob;  
	}  // over all the test points
	
	all_llh.set(i-min_k,r, all_llh.get(i-min_k,r)+curr_fold_llh);
      }

      fprintf(stderr, "%d clusters: log_llh=%f\n",i, all_llh.get(i-min_k,r));
    }
  }

  best_llh = -DBL_MAX;
  for (int a=min_k; a <= max_k; a++)
    for (int b=0; b < NUM_KMEANS_REPEATS; b++) {
      if (best_llh < all_llh.get(a-min_k, b) && all_llh.get(a-min_k, b) < 0 && all_llh.get(a-min_k, b) == all_llh.get(a-min_k, b)) {
	best_llh = all_llh.get(a-min_k, b);
	best_k = a;
      }	  
    }
      
  // Now compute the optimal
  fprintf(stderr, "Optimal number of clusters = %d\n", best_k);
  return best_k;
}

