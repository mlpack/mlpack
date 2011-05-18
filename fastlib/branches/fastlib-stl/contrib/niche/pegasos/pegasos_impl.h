#ifndef INSIDE_PEGASOS_H
#error "This is not a public header file!"
#endif

Pegasos::Pegasos() {

}


void Pegasos::Init(const mat& X, const vec& y, 
		   double lambda, u32 T) {
  Init(X, y, lambda, T, 1);
}


void Pegasos::Init(const mat& X, const vec& y, 
		   double lambda, u32 T, u32 k) {
  X_ = X;
  y_ = y;
  
  n_points_ = X.n_cols;
  
  lambda_ = lambda;
  T_ = T;
  k_ = k;
  
  w_ = vec(X.n_rows);
}


void Pegasos::DoPegasos() {
  if(k_ == 1) {
    DoPegasosTrivialBatch();
  }
  else {
    DoPegasosMiniBatch();
  }
}


void Pegasos::DoPegasosTrivialBatch() {
  w_.zeros();
  
  for(u32 t = 1; t <= T_; t++) {
    u32 ind = rand() % n_points_;
    
    vec x_t = X_.unsafe_col(ind);
    double y_t = y_(ind);
    
    double step_size = 1 / (lambda_ * ((double) t));
    if(y_t * dot(w_, x_t) < 1) {
      w_ = (1 - step_size * lambda_) * w_ + step_size * y_t * x_t;
    }
    else {
      w_ = (1 - step_size * lambda_) * w_;
    }
    
    /*
    // project onto 1 / sqrt(lambda) ball
    double norm_w = norm(w_, 2);
    if(norm_w > 1.0 / sqrt(lambda_)) {
      w_ = w_ * (1.0 / sqrt(lambda_)) / norm_w;
      
    }
    */
  }
}


void Pegasos::DoPegasosMiniBatch() {
  uvec inds(n_points_);
  for(u32 i = 0; i < n_points_; i++) {
    inds(i) = i;
  }
  
  w_.zeros();
  vec subgrad(w_.n_elem);
  for(u32 t = 1; t <= T_; t++) {
    double step_size = 1 / (lambda_ * ((double) t));
    Shuffle(inds);
    
    subgrad.zeros();
    for(u32 i = 0; i < k_; i++) {
      u32 ind = inds(i);
      
      vec x_t = X_.unsafe_col(ind);
      double y_t = y_(ind);
      
      if(y_t * dot(w_, x_t) < 1) {
	subgrad += y_t * x_t;
      }
    }
    w_ = (1 - step_size * lambda_) * w_ + (step_size / ((double)k_)) * subgrad;
    
    /*
    // project onto 1 / sqrt(lambda) ball
    double norm_w = norm(w_, 2);
    if(norm_w > 1.0 / sqrt(lambda_)) {
      w_ = w_ * (1.0 / sqrt(lambda_)) / norm_w;
      
    }
    */
  }
}


vec Pegasos::GetW() {
  return w_;
}


void Pegasos::Shuffle(uvec& numbers) {
  u32 length_minus_1 = numbers.n_elem - 1;
  u32 draw, temp;
  for(u32 i = 0; i < length_minus_1; i++) {
    draw = (rand() % (n_points_ - i)) + i;
    temp = numbers(i);
    numbers(i) = numbers(draw);
    numbers(draw) = temp;
  }
}
