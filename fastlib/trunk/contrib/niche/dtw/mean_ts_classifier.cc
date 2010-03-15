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

  int n_classes = 10;

  const char* data_filename = "test_hi_res_ts.dat";

  Matrix data_with_labels;
  data::Load(data_filename, &data_with_labels);

  index_t n_points = data_with_labels.n_cols();

  printf("%d\n", n_points);


  int n_features = 56;

  double temp = ((double)(data_with_labels.n_rows() - 1.0)) / ((double)n_features);
  if(temp != round(temp)) {
    FATAL("problem with dimensions of data\n");
  }
  index_t n_times = (int) temp;

  printf("n_times = %d\n", n_times);

  for(int k = 0; k < n_features; k++) {
    for(index_t i = 0; i < n_points; i++) {
      Vector ts_i;
      data_with_labels.MakeColumnSubvector(i,
					   (k * n_times) + 1,
					   n_times, &ts_i);
      Normalize(&ts_i);
    }
  }
  
  printf("done normalizing data\n");
  

  int product_n_features_n_times = n_features * n_times;
  ArrayList<Vector> means;
  means.Init(n_classes);
  for(int c = 0; c < n_classes; c++) {
    means[c].Init(product_n_features_n_times);
  }



  GenVector<int> counts;
  counts.Init(n_classes);


  GenVector<int> predicted_labels;
  predicted_labels.Init(n_points);


  for(int i = 0; i < n_points; i++) { // leave out point i



    // reset means (and counts)
    for(int c = 0; c < n_classes; c++) {
      means[c].SetZero();
    }
    counts.SetZero();

    // compute means (and counts)
    for(int j = 0; j < n_points; j++) {
      if(j == i) {
	continue;
      }
      Vector cur_point;
      data_with_labels.MakeColumnSubvector(j, 1,
					   n_features * n_times,
					   &cur_point);

      int cur_label = (int) (data_with_labels.get(0, j) - 1);
      
      // accumulate mean
      la::AddTo(cur_point, &(means[cur_label]));
      counts[cur_label]++;
    }

    for(int c = 0; c < n_classes; c++) {
      la::Scale(1.0 / ((double)(counts[c])), &(means[c]));
    }

    // nearest centroid classifier
    Vector test_point;
    data_with_labels.MakeColumnSubvector(i, 1,
					 n_features * n_times,
					 &test_point);
    int argmin = 0;
    double min = la::DistanceSqEuclidean(test_point, means[0]);
    for(int c = 1; c < n_classes; c++) {
      double test_dist_sq = la::DistanceSqEuclidean(test_point, means[c]);
      if(test_dist_sq < min) {
	min = test_dist_sq;
	argmin = c;
      }
    }
    predicted_labels[i] = argmin + 1;
  }


  int n_correct = 0;
  for(int i = 0; i < n_points; i++) {
    printf("\npredicted_labels[%d] = %d", i, predicted_labels[i]);
    if(predicted_labels[i] == ((int) (data_with_labels.get(0, i)))) {
      n_correct++;
      printf("\tcorrect");
    }
  }
  printf("\n");

  printf("Accuracy: %d/%d = %f\n", n_correct, n_points,
	 ((double)n_correct) / ((double)n_points));





  fx_done(fx_root);
}
