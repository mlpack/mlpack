#include "fastlib/fastlib.h"
#include "dtw.h"



void Normalize(Vector* p_ts) {
  Vector &ts = *p_ts;

  index_t n_times = ts.length();

  // compute mean over time
  double mean = 0;
  for(index_t t = 0; t < n_times; t++) {
    mean += ts[t];
  }
  mean /= ((double) n_times);
  
  // center using temporal mean
  for(index_t t = 0; t < n_times; t++) {
    ts[t] -= mean;
  }
  
  // compute variance
  double variance = 0;
  for(index_t t = 0; t < n_times; t++) {
    variance += ts[t] * ts[t];
  }
  variance /= ((double)(n_times - 1));
  double inv_std_dev = 1 / sqrt(variance);
  
  // scale by 1 / (standard deviation)
  la::Scale(inv_std_dev, &ts);
}
  
  




int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  //const char* data_filename = "/Volumes/Tera/CABI_data/AFNI/analysis/1_8_2010/hi_res_ts.dat";
  const char* training_data_filename = "training_hi_res_ts.dat";
  const char* test_data_filename = "test_hi_res_ts.dat";

  Matrix training_data_with_labels;
  data::Load(training_data_filename, &training_data_with_labels);

  Matrix test_data_with_labels;
  data::Load(test_data_filename, &test_data_with_labels);

  index_t n_training_points = training_data_with_labels.n_cols();
  index_t n_test_points = test_data_with_labels.n_cols();

  printf("%d %d\n", n_training_points, n_test_points);

  // for each feature, we want to construct a warping cost matrix between all pairs of points
  
  int n_features = 56;

  double temp1 = ((double)(training_data_with_labels.n_rows() - 1.0)) / ((double)n_features);
  double temp2 = ((double)(test_data_with_labels.n_rows() - 1.0)) / ((double)n_features);
  if(temp1 != round(temp1)) {
    FATAL("problem with dimensions of training data\n");
  }
  else if(temp2 != round(temp2)) {
    FATAL("problem with dimensions of test data\n");
  }
  else if(((int)temp1) != ((int)temp2)) {
    FATAL("training data dimensions do not match test data dimensions\n");
  }
  index_t n_times = (int) temp1;

  printf("n_times = %d\n", n_times);

  for(int k = 0; k < n_features; k++) {
    for(index_t i = 0; i < n_training_points; i++) {
      Vector ts_i;
      training_data_with_labels.MakeColumnSubvector(i,
						    (k * n_times) + 1,
						    n_times, &ts_i);
      Normalize(&ts_i);
    }
    
    for(index_t j = 0; j < n_test_points; j++) {
      Vector ts_j;
      test_data_with_labels.MakeColumnSubvector(j,
						(k * n_times) + 1,
						n_times, &ts_j);
      Normalize(&ts_j);
    }
  }
    
  printf("done normalizing data\n");



  ArrayList<Matrix> warping_cost_matrices;
  warping_cost_matrices.Init(n_features, n_features);
  for(int k = 0; k < n_features; k++) {
    warping_cost_matrices[k].Init(n_training_points, n_test_points);
  }
  
  for(int k = 0; k < n_features; k++) {
    printf("k = %d\n", k);
    for(int j = 0; j < n_test_points; j++) {
      Vector ts_j;
      test_data_with_labels.MakeColumnSubvector(j,
						(k * n_times) + 1,
						n_times, &ts_j);
      for(int i = 0; i < n_training_points; i++) {
	Vector ts_i;
	training_data_with_labels.MakeColumnSubvector(i,
						      (k * n_times) + 1,
						      n_times, &ts_i);
	ArrayList< GenVector<int> > optimal_path;
	double score = ComputeDTWAlignmentScore(-1, ts_i, ts_j, &optimal_path);
	warping_cost_matrices[k].set(i, j, score);
      }
    }
  }



  Vector scores;
  scores.Init(n_training_points);


  GenVector<int> predicted_labels;
  predicted_labels.Init(n_test_points);


  for(int j = 0; j < n_test_points; j++) {
    scores.SetZero();
    for(int k = 0; k < n_features; k++) {
      for(int i = 0; i < n_training_points; i++) {
	scores[i] += log(warping_cost_matrices[k].get(i, j));
      }
    }
    printf("test point %d\n", j);
    scores.PrintDebug("scores");
    
    index_t argmin = 0;
    double min = scores[0];
    for(int i = 1; i < n_training_points; i++) {
      if(scores[i] < min) {
	min = scores[i];
	argmin = i;
      }
    }

    predicted_labels[j] = (int) (training_data_with_labels.get(0, argmin));
  }

  int n_correct = 0;
  for(int j = 0; j < n_test_points; j++) {
    printf("\npredicted_labels[%d] = %d", j, predicted_labels[j]);
    if(predicted_labels[j] == ((int) (test_data_with_labels.get(0, j)))) {
      n_correct++;
      printf("\tcorrect");
    }
  }
  printf("\n");

  printf("Accuracy: %d/%d = %f\n", n_correct, n_test_points,
	 ((double)n_correct) / ((double)n_test_points));


  fx_done(fx_root);
}
