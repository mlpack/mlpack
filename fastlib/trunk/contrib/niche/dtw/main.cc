#include "fastlib/fastlib.h"
#include "dtw.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  const char* training_data_filename = "/Users/niche/Downloads/ucr_time_series_datasets/FaceFour/FaceFour_TRAIN";
  const char* test_data_filename = "/Users/niche/Downloads/ucr_time_series_datasets/FaceFour/FaceFour_TEST";
  
  //const char* training_data_filename = "/scratch/niche/ucr_time_series_datasets/FaceFour/FaceFour_TRAIN";
  //const char* test_data_filename = "/scratch/niche/ucr_time_series_datasets/FaceFour/FaceFour_TEST";
 
  // 1-NN classification test
  Matrix training_data_with_labels;
  data::Load(training_data_filename, &training_data_with_labels);

  Matrix test_data_with_labels;
  data::Load(test_data_filename, &test_data_with_labels);

  index_t n_training_points = training_data_with_labels.n_cols();
  index_t n_test_points = test_data_with_labels.n_cols();

  index_t n_dims = training_data_with_labels.n_rows() - 1;

  GenVector<int> predicted_labels;
  predicted_labels.Init(n_test_points);

  ArrayList< GenVector<int> > best_optimal_path;
  best_optimal_path.Init(0);

  for(int i = 0; i < n_training_points; i++) {
    Vector cur_training_ts;
    training_data_with_labels.MakeColumnSubvector(i, 1, n_dims, &cur_training_ts);

    char* training_data_filename = (char*) malloc(100 * sizeof(char));
    sprintf(training_data_filename, "results/training_%03d.dat", i);
    FILE* training_data_file = fopen(training_data_filename, "w");
    for(int k = 0; k < cur_training_ts.length(); k++) {
      fprintf(training_data_file, "%f\n", cur_training_ts[k]);
    }
    fclose(training_data_file);
  }
    




  FILE* nn_file = fopen("results/nn.dat", "w");
  
  for(int i = 0; i < n_test_points; i++) {
    Vector cur_test_ts;
    test_data_with_labels.MakeColumnSubvector(i, 1, n_dims, &cur_test_ts);

    char* test_data_filename = (char*) malloc(100 * sizeof(char));
    sprintf(test_data_filename, "results/test_%03d.dat", i);
    FILE* test_data_file = fopen(test_data_filename, "w");
    for(int k = 0; k < cur_test_ts.length(); k++) {
      fprintf(test_data_file, "%f\n", cur_test_ts[k]);
    }
    fclose(test_data_file);
    
    double min_score = std::numeric_limits<double>::max();
    int best_training_ind = -1;
    
    for(int j = 0; j < n_training_points; j++) {
      Vector cur_training_ts;
      training_data_with_labels.MakeColumnSubvector(j, 1, n_dims, &cur_training_ts);
      
      ArrayList< GenVector<int> > optimal_path;
      double score =
	ComputeDTWAlignmentScore(-1, cur_training_ts, cur_test_ts,
				 &optimal_path);

      if(unlikely(score < min_score)) {
	min_score = score;
	best_training_ind = j;
	best_optimal_path.Renew();
	best_optimal_path.InitCopy(optimal_path);
      }
    }

    fprintf(nn_file, "%d %d\n", i, best_training_ind);
    
    char* optimal_path_filename = (char*) malloc(100 * sizeof(char));
    sprintf(optimal_path_filename, "results/optimal_path_%03d.dat", i);
    FILE* optimal_path_file = fopen(optimal_path_filename, "w");
    for(int k = best_optimal_path.size() - 1; k >= 0; k--) {
      fprintf(optimal_path_file,
	      "%d, %d\n",
	      best_optimal_path[k][0], best_optimal_path[k][1]);
    }
    fclose(optimal_path_file);
    

    predicted_labels[i] = (int) training_data_with_labels.get(0, best_training_ind);
  }

  fclose(nn_file);

  FILE* loss_file = fopen("results/loss.dat", "w");
  int n_correct = 0;
  for(int i = 0; i < n_test_points; i++){
    if(predicted_labels[i] == ((int) (test_data_with_labels.get(0, i)))) {
      fprintf(loss_file, "0\n");
      n_correct++;
    }
    else {
      fprintf(loss_file, "1\n");
    }
  }
  fclose(loss_file);

  printf("Accuracy: %d/%d = %f\n", n_correct, n_test_points,
	 ((double)n_correct) / ((double)n_test_points));
	
  
  
    
  
  /*
  // quick single pair alignment

  Vector ts1;
  LoadTimeSeries("ts1.dat", &ts1);
  ts1.PrintDebug("ts1");

  Vector ts2;
  LoadTimeSeries("ts2.dat", &ts2); 
  ts2.PrintDebug("ts2");
  
  ArrayList< GenVector<int> > optimal_path;
  double score = ComputeDTWAlignmentScore(-1, ts1, ts2, &optimal_path);
  printf("score = %f\n", score);

  FILE* file = fopen("optimal_path.dat", "w");
  for(int i = optimal_path.size() - 1; i >= 0; i--) {
    fprintf(file, "%d, %d\n", optimal_path[i][0], optimal_path[i][1]);
  }
  fclose(file);
  */

  fx_done(fx_root);
}
