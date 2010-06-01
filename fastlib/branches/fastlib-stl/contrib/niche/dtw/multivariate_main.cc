#include "fastlib/fastlib.h"
#include "dtw.h"


void GetPointAsMatrix(Matrix data_with_labels,
		      index_t ind, index_t n_times, index_t n_features,
		      index_t prod_n_times_n_features, Matrix* p_point_mat) {
  Vector point_vec;
  Matrix point_skinny_mat;
    
  data_with_labels.MakeColumnSubvector(ind, 1, prod_n_times_n_features,
				       &point_vec);
  point_skinny_mat.AliasColVector(point_vec);
  Matrix q;
  point_skinny_mat.MakeReshaped(n_times, n_features,
				&q);
  p_point_mat -> Copy(q);
}
  

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
  if(variance > 0) {
    double inv_std_dev = 1 / sqrt(variance);
    // scale by 1 / (standard deviation)
    la::Scale(inv_std_dev, &ts);
  }
  else {
    printf("skipping variance normalization\n");
  }
}
  
  




int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  //const char* training_data_filename = 
  //  "/Volumes/Tera/CABI_data/AFNI/analysis/1_8_2010/training_hi_res_ts.dat";
  //const char* test_data_filename = 
  //  "/Volumes/Tera/CABI_data/AFNI/analysis/1_8_2010/test_hi_res_ts.dat";
  //const char* training_data_filename = "training_noisy_generated_hi_res_ts.dat";
  //const char* test_data_filename = "test_noisy_generated_hi_res_ts.dat";

  const char* directory = "/Volumes/Tera/CABI_data/AFNI/analysis/1_8_2010";
  const char* file_descriptor = fx_param_str_req(NULL, "desc");
  
  char training_data_filename[200];
  sprintf(training_data_filename, "%s/training_hi_res_ts%s.dat", directory, file_descriptor);

  char test_data_filename[200];
  sprintf(test_data_filename, "%s/test_hi_res_ts%s.dat", directory, file_descriptor);

  
  
  const int n_features = fx_param_int_req(NULL, "n_features");
  
  const bool locked_features = fx_param_bool_req(NULL, "locked");
  //int n_features = 56;
  //int n_features = 82;
  //bool locked_features = false;

  
  Matrix training_data_with_labels;
  data::Load(training_data_filename, &training_data_with_labels);

  Matrix test_data_with_labels;
  data::Load(test_data_filename, &test_data_with_labels);

  index_t n_training_points = training_data_with_labels.n_cols();
  index_t n_test_points = test_data_with_labels.n_cols();

  printf("%d %d\n", n_training_points, n_test_points);

  // for each feature, we want to construct a warping cost matrix between all pairs of points
  

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

  /* transpose data for DTW with locked features
     note: do not transpose data for DTW with unlocked features
  */

  


  Vector scores;
  scores.Init(n_training_points);

  GenVector<int> predicted_labels;
  predicted_labels.Init(n_test_points);

  index_t prod_n_times_n_features = n_times * n_features;

  for(int j = 0; j < n_test_points; j++) {
    scores.SetZero();

    Matrix test_point_mat;
    GetPointAsMatrix(test_data_with_labels, j, n_times, n_features,
		     prod_n_times_n_features,
		     &test_point_mat);
    Matrix test_point_mat_to_use;
    if(locked_features) {
      la::TransposeInit(test_point_mat, &test_point_mat_to_use);
    }
    else {
      test_point_mat_to_use.Alias(test_point_mat);
    }

    for(int i = 0; i < n_training_points; i++) {
      Matrix training_point_mat;
      GetPointAsMatrix(training_data_with_labels, i, n_times, n_features,
		       prod_n_times_n_features,
		       &training_point_mat);
      Matrix training_point_mat_to_use;
      if(locked_features) {
	la::TransposeInit(training_point_mat, &training_point_mat_to_use);
      }
      else {
	training_point_mat_to_use.Alias(training_point_mat);
      }

      //test_point_mat.PrintDebug("test point");
      //training_point_mat.PrintDebug("training_point");
      scores[i] =
	ComputeDTWAlignmentScore(-1,
				 test_point_mat_to_use,
				 training_point_mat_to_use,
				 locked_features);
      //printf("scores[%d] = %f\n", i, scores[i]);
      //exit(1);
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
