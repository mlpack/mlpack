#ifndef INSIDE_PEGASOS_H
#error "This is not a public header file!"
#endif

Pegasos::Pegasos() {

}


void Pegasos::Init(const mat& X, const vec& y, 
		   double lambda, u32 T) {
  X_ = X;
  y_ = y;
  
  n_points_ = X.n_cols;
  
  lambda_ = lambda;
  T_ = T;
  
  w_ = vec(X.n_rows);
}


void Pegasos::DoPegasos() {
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


vec Pegasos::GetW() {
  return w_;
}
