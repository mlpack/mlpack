#ifndef INSIDE_DTW_IMPL_H
#error "This is not a public header file!"
#endif


void LoadTimeSeries(const char* filename, Vector* p_time_series) {
  Vector &time_series = *p_time_series;

  Matrix time_series_matrix;
  data::Load(filename, &time_series_matrix);

  int n_dims = time_series_matrix.n_cols();

  time_series.Init(n_dims);
  for(int i = 0; i < n_dims; i++) {
    time_series[i]= time_series_matrix.get(0, i);
  }
}

double ComputeDTWAlignmentScore(int b,
				const Matrix &x, const Matrix &y) {
  int n_features = x.n_cols();

  double score = 0;
  for(int k = 0; k < n_features; k++) {
    Vector x_cur_ts;
    Vector y_cur_ts;
    x.MakeColumnVector(k, &x_cur_ts);
    y.MakeColumnVector(k, &y_cur_ts);
    
    ArrayList< GenVector<int> > best_path;
    score += log(ComputeDTWAlignmentScore(b, x_cur_ts, y_cur_ts, &best_path));
  }
  return score;
}

double ComputeDTWAlignmentScore(int b,
				const Vector &x, const Vector &y, 
  				ArrayList< GenVector<int> >* p_best_path) {
  ArrayList< GenVector<int> > &best_path = *p_best_path;

  int n_x = x.length();
  int n_y = y.length();

  if(b == -1) {
    b = n_x + n_y + 1;
  }

  Matrix gamma;
  gamma.Init(n_x + 1, n_y + 1);

  Matrix best_in;
  best_in.Init(n_x + 1, n_y + 1);

  double dbl_max = std::numeric_limits<double>::max();

  for(int i = 1; i <= n_x; i++) {
    gamma.set(i, 0, dbl_max);
  }
  for(int j = 1; j <= n_y; j++) {
    gamma.set(0, j, dbl_max);
  }

  gamma.set(0, 0, 0);

  for(int i = 1; i <= n_x; i++) {
    for(int j = 1; j <= n_y; j++) {
      double diff = x[i - 1] - y[j - 1];
      //double cost = fabs(diff);
      //double cost = fabs(diff * diff * diff * diff);
      double cost = diff * diff;

      double c_both = gamma.get(i - 1, j - 1);
      double c_i = gamma.get(i - 1, j);
      double c_j = gamma.get(i, j - 1);

      // unconstrained DTW
      if((j - b < i) && (i < j + b)) {
	if((c_both <= c_i) && (c_both <= c_j)) {
	  gamma.set(i, j, cost + c_both);
	  best_in.set(i, j, 0); // best path in is through (i-1, j-1)
	}
	else if((c_i <= c_both) && (c_i <= c_j)) {
	  gamma.set(i, j, cost + c_i);
	  best_in.set(i, j, -1); // best path in is through (i-1, j)
	}
	else {
	  gamma.set(i, j, cost + c_j);
	  best_in.set(i, j, 1); // best path in is through (i, j-1)
	}
      }
      else {
	if(i <= (j - b) ) {
	  if(c_both <= c_j) {
	    gamma.set(i, j, cost + c_both);
	    best_in.set(i, j, 0);
	  }
	  else {
	    gamma.set(i, j, cost + c_j);
	    best_in.set(i, j, 1);
	  }
	}
	else if(j <= (i - b)) {
	  if(c_both <= c_i) {
	    gamma.set(i, j, cost + c_both);
	    best_in.set(i, j, 0);
	  }
	  else {
	    gamma.set(i, j, cost + c_i);
	    best_in.set(i, j, -1);
	  }
	}
      }
    }
  }

  //printf("cost of best path = %f\n", gamma.get(n_x, n_y));

  // reconstruct best path through a trace back
  
  best_path.Init(0, n_x + n_y);
  
  int cur_i = n_x;
  int cur_j = n_y;

  int path_length = 0;
  while((cur_i != 0) && (cur_j != 0)) {
    best_path.PushBack(1);
    best_path[path_length].Init(2);

    best_path[path_length][0] = cur_i;
    best_path[path_length][1] = cur_j;

    if(abs(cur_i - cur_j) > b) {
      printf("band exceeded!\n");
    }

    
    //printf("best_path[%d](cur_i, cur_j) = (%d, %d)\n", path_length, cur_i, cur_j);

    path_length++;
    
    double direction_code = best_in.get(cur_i, cur_j);
    if(direction_code < -0.5) {
      cur_i--;
    }
    else if(direction_code > 0.5) {
      cur_j--;
    }
    else {
      cur_i--;
      cur_j--;
    }
  }
  
  
  //printf("best path length = %d\n", best_path.size());
  //printf("path_length = %d\n", path_length);
  
  //printf("n_x + n_y = %d + %d = %d\n", n_x, n_y, n_x + n_y);
  
  //printf("Printing best path:\n");
  //for(int i = best_path.size() - 1; i >= 0; i--) {
  //  printf("(%d, %d)\n", best_path[i][0], best_path[i][1]);
  //}

  int max_diag_dev = -1;
  for(int i = 0; i < path_length; i++) {
    int dev = abs(best_path[i][0] - best_path[i][1]);
    if(unlikely(dev > max_diag_dev)) {
      max_diag_dev= dev;
    }
  }

  //printf("%f\n", gamma.get(n_x, n_y));
  //exit(1);

  //return (double) max_diag_dev;
  //return (double) max_diag_dev / ((double) path_length);
  //return ((double) path_length);
  //return ((double) gamma.get(n_x, n_y)) / ((double) path_length);
  return ((double) gamma.get(n_x, n_y));
}
