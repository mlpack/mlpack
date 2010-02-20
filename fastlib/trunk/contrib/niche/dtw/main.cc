#include "fastlib/fastlib.h"
#include "dtw.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  Matrix training_data_with_labels;
  data::Load("/scratch/niche/ucr_time_series_datasets/CBF/CBF_TRAIN",
	     &training_data_with_labels);

  Matrix test_data_with_labels;
  data::Load("/scratch/niche/ucr_time_series_datasets/CBF/CBF_TEST",
	     &test_data_with_labels);

  index_t n_training_points = training_data_with_labels.n_cols();
  index_t n_test_points = test_data_with_labels.n_cols();

  GenVector<int> predicted_labels;
  predicted_labels.Init(n_test_points);
 

  for(int i = 0; i < n_test_points; i++) {
    Vector cur_test_ts;
    test_data_with_labels.MakeColumnSubvector(i, 1, 60, &cur_test_ts);
    
    double min_score = std::numeric_limits<double>::max();
    int best_training_ind = -1;
    
    for(int j = 0; j < n_training_points; j++) {
      Vector cur_training_ts;
      training_data_with_labels.MakeColumnSubvector(j, 1, 60, &cur_training_ts);
      
      ArrayList< GenVector<int> > best_path;
      double score =
	ComputeDTWAlignmentScore(-1, cur_training_ts, cur_test_ts, &best_path);

      if(unlikely(score < min_score)) {
	min_score = score;
	best_training_ind = j;
      }
    }
    predicted_labels[i] = (int) training_data_with_labels.get(0, best_training_ind);
  }

  int n_correct = 0;
  for(int i = 0; i < n_test_points; i++){
    if(predicted_labels[i] == ((int) (test_data_with_labels.get(0, i)))) {
      n_correct++;
    }
  }

  printf("Accuracy: %d/%d = %f\n", n_correct, n_test_points,
	 ((double)n_correct) / ((double)n_test_points));
	

  
  /*  
  

  

  Vector ts1;
  LoadTimeSeries("ts1.dat", &ts1);
  ts1.PrintDebug("ts1");

  Vector ts2;
  LoadTimeSeries("ts2.dat", &ts2); 
  ts2.PrintDebug("ts2");
  
  ArrayList< GenVector<int> > best_path;

  double score = ComputeDTWAlignmentScore(ts1, ts2, &best_path);
  
  printf("score = %f\n", score);

  */

  fx_done(fx_root);
}
