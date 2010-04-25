#ifndef LOGPROB_MATRIXMULT_H
#define LOGPROB_MATRIXMULT_H


double LogSumExp(double x, double y) {
  if(x > y) {
    return x + log(1 + exp(y - x));
  }
  else {
    return y + log(1 + exp(x - y));
  }
}


double LogSumExp(const Vector &x) {

  int n_dims = x.length();
  
  double max = -std::numeric_limits<double>::max();
  
  for(int i = 0; i < n_dims; i++) {
    if(unlikely(x[i] > max)) {
      max = x[i];
    }
  }

  double sum = 0;
  for(int i = 0; i < n_dims; i++) {
    sum += exp(x[i] - max);
  }
  
  return max + log(sum);
}
    

double LogSumMapExpVectors(const Vector &a, const Vector &b) {

  int n_dims = a.length();
  
  double max = -std::numeric_limits<double>::max();
  
  double sum;
  for(int i = 0; i < n_dims; i++) {
    sum = a[i] + b[i];
    if(unlikely(sum > max)) {
      max = sum;
    }
  }

  sum = 0;
  for(int i = 0; i < n_dims; i++) {
    sum += exp(a[i] + b[i] - max);
  }
  
  return max + log(sum);
}



void LogMatrixMultiplyOverwrite(const Matrix &A,
				const Matrix &B,
				Matrix* p_C) {
  Matrix& C = *p_C;

  Matrix At;
  la::TransposeInit(A, &At);

  index_t n_At_cols = At.n_cols();
  index_t n_B_cols = B.n_cols();

  for(int i = 0; i < n_At_cols; i++) {
    Vector A_i;
    At.MakeColumnVector(i, &A_i);

    for(int k = 0; k < n_B_cols; k++) {
      Vector B_k;
      B.MakeColumnVector(k, &B_k);

      C.set(i, k, LogSumMapExpVectors(A_i, B_k));
    }
  }
}

void LogMatrixMultiplyATransOverwrite(const Matrix &At,
				      const Vector &x,
				      Vector* p_y) {
  Matrix& y = *p_y;

  index_t n_At_cols = At.n_cols();

  for(int i = 0; i < n_At_cols; i++) {
    Vector A_i;
    At.MakeColumnVector(i, &A_i);

    y[i] = LogSumMapExpVectors(A_i, x);
  }
}


void LogMatrixMultiplyOverwrite(const Vector &x,
				const Matrix &B,
				Vector* p_y) {
  Matrix& y = *p_y;
  
  index_t n_B_cols = B.n_cols();
  
  for(int k = 0; k < n_B_cols; k++) {
    Vector B_k;
    B.MakeColumnVector(k, &B_k);
    
    y[k] = LogSumMapExpVectors(x, B_k);
  }
}



#endif /* LOGPROB_MATRIXMULT_H */
