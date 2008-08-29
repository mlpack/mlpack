
void BigSdpNmfObjectiveMaxVar::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dim_=fx_param_int(module_, "new_dim", 5); 
  rank_=fx_param_int(module_, "rank", 3);
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	eq_lagrange_mult_.Init(values_.size());
  eq_lagrange_mult_.SetAll(0);
	offset_h_ = num_of_rows_*new_dim_;
}

void BigSdpNmfObjectiveMaxVar::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
	eq_lagrange_mult_.Destruct();
}

void BigSdpNmfObjectiveMaxVar::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  // gradient of the variance
  gradient->CopyValues(coordinates);
	la::Scale(-2.0, gradient);
  // Gradient on the equalities and inequalities  
	for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	  la::AddExpert(rank_*new_dim_, -eq_lagrange_mult_[i]+2*sigma_*diff,
				 coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(rank_*new_dim_, -eq_lagrange_mult_[i]+2*sigma_*diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
  }	
}

void BigSdpNmfObjectiveMaxVar::ComputeObjective(Matrix &coordinates, double *objective) {
  *objective=0;	
  for(index_t i=0; i<coordinates.n_cols(); i++) {
	   *objective-=la::Dot(coordinates.n_rows(), 
				 coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
	}
}

void BigSdpNmfObjectiveMaxVar::ComputeFeasibilityError(Matrix &coordinates, double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		*error+=diff*diff;
 	}
}

double BigSdpNmfObjectiveMaxVar::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<values_.size(); i++) {
    // equalities
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		lagrangian+=(sigma_*diff-eq_lagrange_mult_[i])*diff;
	}
	return lagrangian;
}

void BigSdpNmfObjectiveMaxVar::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    // equalities
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		eq_lagrange_mult_[i]-=sigma_*diff;
 	}
}

void BigSdpNmfObjectiveMaxVar::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMaxVar::set_sigma(double sigma) {
  sigma_=sigma;
}

void BigSdpNmfObjectiveMaxVar::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(rank_, (num_of_rows_+num_of_columns_)*new_dim_);
	for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random());
		}
	}
}


bool BigSdpNmfObjectiveMaxVar::IsDiverging(double objective) {
  return false;
} 

///////////////////////////////////////////////////////////////////////////

void BigSdpNmfObjectiveMinVar::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dim_=fx_param_int(module_, "new_dim", 5); 
  rank_=fx_param_int(module_, "rank", 3);
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	eq_lagrange_mult_.Init(values_.size());
  eq_lagrange_mult_.SetAll(0);
	offset_h_ = num_of_rows_*new_dim_;
}

void BigSdpNmfObjectiveMinVar::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
	eq_lagrange_mult_.Destruct();
}

void BigSdpNmfObjectiveMinVar::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  // gradient of the variance
  gradient->CopyValues(coordinates);
	la::Scale(2.0, gradient);
  // Gradient on the equalities and inequalities  
	for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	  la::AddExpert(rank_*new_dim_, -eq_lagrange_mult_[i]+2*sigma_*diff,
				 coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(rank_*new_dim_, -eq_lagrange_mult_[i]+2*sigma_*diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
  }	
}

void BigSdpNmfObjectiveMinVar::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
  for(index_t i=0; i<coordinates.n_cols(); i++) {
	   *objective+=la::Dot(coordinates.n_rows(), 
				 coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
	}
}

void BigSdpNmfObjectiveMinVar::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		*error+=diff*diff;
 	}
}

double BigSdpNmfObjectiveMinVar::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<values_.size(); i++) {
    // equalities
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		lagrangian+=(sigma_*diff-eq_lagrange_mult_[i])*diff;
	}
	return lagrangian;
}

void BigSdpNmfObjectiveMinVar::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    // equalities
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		eq_lagrange_mult_[i]-=sigma_*diff;
 	}
}

void BigSdpNmfObjectiveMinVar::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMinVar::set_sigma(double sigma) {
  sigma_=sigma;
}

void BigSdpNmfObjectiveMinVar::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(rank_, (num_of_rows_+num_of_columns_)*new_dim_);
	for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random());
		}
	}
}


bool BigSdpNmfObjectiveMinVar::IsDiverging(double objective) {
  return false;
} 

///////////////////////////////////////////////////////////////////////////////
void BigSdpNmfObjectiveMinVarIneq::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dim_=fx_param_int(module_, "new_dim", 5); 
  rank_=fx_param_int(module_, "rank", 3);
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	ineq_lagrange_mult_.Init(values_.size());
  ineq_lagrange_mult_.SetAll(1.0);
	offset_h_ = num_of_rows_*new_dim_;
}

void BigSdpNmfObjectiveMinVarIneq::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
	ineq_lagrange_mult_.Destruct();
}

void BigSdpNmfObjectiveMinVarIneq::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  // gradient of the variance
  gradient->CopyValues(coordinates);
	la::Scale(2.0, gradient);
  // Gradient on the equalities and inequalities  
	for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
    if (sigma_*diff <= ineq_lagrange_mult_[i]) {
	    la::AddExpert(rank_*new_dim_, -ineq_lagrange_mult_[i]+2*sigma_*diff,
			    coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		  la::AddExpert(rank_*new_dim_, -ineq_lagrange_mult_[i]+2*sigma_*diff, 
				  coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
    }
  }	
}

void BigSdpNmfObjectiveMinVarIneq::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
  for(index_t i=0; i<coordinates.n_cols(); i++) {
	   *objective+=la::Dot(coordinates.n_rows(), 
				 coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
	}
}

void BigSdpNmfObjectiveMinVarIneq::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
    if (diff < 0) {
		  *error=std::max(*error, fabs(diff));
 	  }
  }
}


double BigSdpNmfObjectiveMinVarIneq::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<values_.size(); i++) {
    // inequalities
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		if (sigma_*diff <= ineq_lagrange_mult_[i]) {
      lagrangian+=(sigma_*diff-ineq_lagrange_mult_[i])*diff;
    } else {
      lagrangian+=-math::Pow<2,1>(ineq_lagrange_mult_[i])/sigma_;
    }
	}
	return lagrangian;
}

void BigSdpNmfObjectiveMinVarIneq::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    // inequalities
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		ineq_lagrange_mult_[i]=std::max(ineq_lagrange_mult_[i]-sigma_*diff, 0.0);
 	}
}

void BigSdpNmfObjectiveMinVarIneq::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMinVarIneq::set_sigma(double sigma) {
  sigma_=sigma;
}

void BigSdpNmfObjectiveMinVarIneq::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(rank_, (num_of_rows_+num_of_columns_)*new_dim_);
  double	mean_value=0.0;
  for(index_t i=0; i<values_.size(); i++) {
    mean_value+=values_[i];
  }
  mean_value/=values_.size();
  for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random(0, 1e-5));
		}
	}
}


bool BigSdpNmfObjectiveMinVarIneq::IsDiverging(double objective) {
  return false;
} 

////////////////////////////////////////////////////////////////////////
void BigSdpNmfObjectiveMinVarDiagonalDominance::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dim_=fx_param_int(module_, "new_dim", 5); 
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	eq_lagrange_mult_.Init(values_.size());
  eq_lagrange_mult_.SetAll(1.0);
  ineq_lagrange_mult_.Init(num_of_columns_+num_of_rows_);
  ineq_lagrange_mult_.SetAll(1.0);
  ineq_lagrange_mult1_.Init(new_dim_*(num_of_columns_+num_of_rows_));
  ineq_lagrange_mult1_.SetAll(1.0);
	offset_h_ = num_of_rows_;
}

void  BigSdpNmfObjectiveMinVarDiagonalDominance::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
  eq_lagrange_mult_.Destruct();
	ineq_lagrange_mult_.Destruct();
}

void BigSdpNmfObjectiveMinVarDiagonalDominance::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  // gradient of the variance
  gradient->CopyValues(coordinates);
	la::Scale(2.0, gradient);
  // Gradient on the equalities and inequalities  

  for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i];
    index_t h =offset_h_+columns_[i];
    double diff=la::Dot(new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	  la::AddExpert(new_dim_, -2*eq_lagrange_mult_[i]+2*sigma_*diff,
	      coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(new_dim_, -2*eq_lagrange_mult_[i]+2*sigma_*diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
  }	
  
  // for the diagonal dominance
  Matrix ones;
  ones.Init(coordinates.n_cols(), 1);
  ones.SetAll(1.0);
  Matrix sums1;
  la::MulInit(coordinates, ones, &sums1);
  Matrix sums2;
  la::MulTransAInit(coordinates, sums1, &sums2);
  Matrix augmented_gradient;
  augmented_gradient.Init(coordinates.n_rows(), coordinates.n_cols());
  augmented_gradient.SetAll(0.0);
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    double diagonal_element=la::Dot(new_dim_, 
        coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
    double diff=2*diagonal_element-sums2.get(i, 0);
    if (sigma2_*diff <= ineq_lagrange_mult_[i]) {
      Matrix temp;
      temp.Init(new_dim_, 1);
      temp.CopyColumnFromMat(0, i, coordinates);
      la::Scale(2, &temp);
      la::SubFrom(sums1, &temp);
      la::AddExpert(new_dim_, -2*ineq_lagrange_mult_[i]+2*sigma2_*diff,
          temp.ptr(), augmented_gradient.GetColumnPtr(i));
    }
  }
  
  //  for the non-negativity
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    for(index_t j=0; j<coordinates.n_rows(); j++) {
      double diff=coordinates.get(j, i);
      if (sigma2_*diff <= ineq_lagrange_mult1_[i*new_dim_+j]) {
        augmented_gradient.set(j, i, 
            augmented_gradient.get(j, i)+
            sigma2_*diff-ineq_lagrange_mult1_[i*new_dim_+j]);
      }
    }
  }
  la::AddTo(augmented_gradient, gradient);
}

void BigSdpNmfObjectiveMinVarDiagonalDominance::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
  Vector ones;
  ones.Init(new_dim_);
  ones.SetAll(1.0);
  for(index_t i=0; i<coordinates.n_cols(); i++) {
	   *objective+=la::Dot(coordinates.n_rows(), 
				 coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
	}
  //*objective=-*objective;
}

void BigSdpNmfObjectiveMinVarDiagonalDominance::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i];
    index_t h =offset_h_+columns_[i];
    double diff=la::Dot(new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		  *error+=diff*diff;
  }
  printf("eq_infeasibility:%lg ", *error);
  // diagonal dominance
  Matrix ones;
  ones.Init(coordinates.n_cols(), 1);
  ones.SetAll(1.0);
  Matrix sums1;
  la::MulInit(coordinates, ones, &sums1);
  Matrix  sums2;
  la::MulTransAInit(coordinates, sums1, &sums2);
  double diagonal_element;
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    diagonal_element=la::Dot(new_dim_, 
        coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
    double diff=2*diagonal_element-sums2.get(i, 0);
    if (diff<0) {
      *error+=diff*diff;
    }
  }	
  // for non-negativity
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    for(index_t j=0; j<coordinates.n_rows(); j++) {
      double diff=coordinates.get(j, i);
      if (diff<0) {
        *error+=diff*diff;
      }
    }
  }
 
  printf("total_infeasibility:%lg\n", *error);
}


double BigSdpNmfObjectiveMinVarDiagonalDominance::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  
  for(index_t i=0; i<values_.size(); i++) {
    // equalities
    index_t w =rows_[i];
    index_t h =offset_h_+columns_[i];
    double diff=la::Dot(new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
    lagrangian+=(sigma_*diff/2-eq_lagrange_mult_[i])*diff;
  }
  
  // diagonal dominance
  Matrix ones;
  ones.Init(coordinates.n_cols(), 1);
  ones.SetAll(1.0);
  Matrix sums1;
  la::MulInit(coordinates, ones, &sums1);
  Matrix  sums2;
  la::MulTransAInit(coordinates, sums1, &sums2);
  double diagonal_element;
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    diagonal_element=la::Dot(new_dim_, 
        coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
    double diff=2*diagonal_element-sums2.get(i, 0);
  	if (sigma2_*diff <= ineq_lagrange_mult_[i]) {
      lagrangian+=(sigma2_*diff/2-ineq_lagrange_mult_[i])*diff;
    } else {
      lagrangian+=-math::Pow<2,1>(ineq_lagrange_mult_[i])/(2*sigma2_);
    }
  }
  
  // for non-negativity
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    for(index_t j=0; j<coordinates.n_rows(); j++) {
      double diff=coordinates.get(j, i);
      if (sigma2_*diff < ineq_lagrange_mult1_[i*new_dim_+j]) {
        lagrangian+=(sigma2_*diff-ineq_lagrange_mult1_[i*new_dim_+j])*diff;
      } else {
        lagrangian+=-math::Pow<2,1>(
            ineq_lagrange_mult1_[i*new_dim_+j])/(2*sigma2_);
      }
    }
  }
  
	return lagrangian;
}

void BigSdpNmfObjectiveMinVarDiagonalDominance::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i];
    index_t h =offset_h_+columns_[i];
    // equalities
    double diff=la::Dot(new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		eq_lagrange_mult_[i]=eq_lagrange_mult_[i]-sigma_*diff;
 	}
  // diagonal dominance
  Matrix ones;
  ones.Init(coordinates.n_cols(), 1);
  ones.SetAll(1.0);
  Matrix sums1;
  la::MulInit(coordinates, ones, &sums1);
  Matrix sums2;
  la::MulTransAInit(coordinates, sums1, &sums2);
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    double diagonal_element=la::Dot(new_dim_, 
        coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
    double diff=2*diagonal_element-sums2.get(i, 0);
    ineq_lagrange_mult_[i]=std::max(ineq_lagrange_mult_[i]-sigma2_*diff, 0.0);
    //ineq_lagrange_mult_[i]=ineq_lagrange_mult_[i]-sigma2_*diff;
  }
  // for non-negativity
  for(index_t i=0; i<coordinates.n_cols(); i++) {
    for(index_t j=0; j<coordinates.n_rows(); j++) {
      double diff=coordinates.get(j, i);
      ineq_lagrange_mult_[i]=
        std::max(ineq_lagrange_mult1_[i*new_dim_+j]-sigma2_*diff, 0.0);
    }
  }
}

void BigSdpNmfObjectiveMinVarDiagonalDominance::Project(Matrix *coordinates) {
  //OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMinVarDiagonalDominance::set_sigma(double sigma) {
  sigma_=sigma;
  sigma2_=sigma/0.0001;
}

void BigSdpNmfObjectiveMinVarDiagonalDominance::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dim_, num_of_rows_+num_of_columns_);
  double	max_value=0.0;
  for(index_t i=0; i<values_.size(); i++) {
    if (max_value<values_[i]) {
      max_value=values_[i];
    }
  }

  for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random(0, max_value));
		}
	}
}

bool BigSdpNmfObjectiveMinVarDiagonalDominance::IsDiverging(double objective) {
  return false;
} 

bool BigSdpNmfObjectiveMinVarDiagonalDominance::IsOptimizationOver(
    Matrix &coordinates, Matrix &gradient, double step) {
 return false; 
}

bool BigSdpNmfObjectiveMinVarDiagonalDominance::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  return false;  
}

////////////////////////////////////////////////////////////////////////
void BigSdpNmfObjectiveMaxVarIsometric::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dimension_=fx_param_int(module_, "new_dimension", 5); 
  grad_tolerance_=fx_param_double(module_, "grad_tolerance", 1);
  desired_feasibility_error_=fx_param_double(module_, "feasibility_error", 0.1);
  // Initializations for the dot products on v
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	h_offset_=0;
  w_offset_=num_of_rows_;
  eq_lagrange_mult1_.Init(values_.size());
  eq_lagrange_mult1_.SetAll(1.0);
  sigma_ratio_=0.0;
  infeasibility1_=0;
  previous_infeasibility1_=DBL_MAX;
  infeasibility2_=0;
  
  // compute the norm of values
  v_norm_=0;
  for(index_t i=0; i<values.size(); i++) {
    v_norm_+=values[i]*values[i];    
  }
  
  // we need to put the data back to a matrix to build the 
  // tree and etc
  Matrix data_mat;
  data_mat.Init(num_of_rows_, num_of_columns_);
  data_mat.SetAll(0.0);
/*  index_t count=0;
 	for(index_t i=0; i<data_mat.n_rows(); i++) {
		for(index_t j=0; j< data_mat.n_cols(); j++) {
			  data_mat.set(i, j, values[count]);
        count++;
		}
	}
*/
  for(index_t i=0; i<rows_.size(); i++) {
    data_mat.set(rows_[i], columns_[i], values_[i]);
  }
  // Initializations for the local isometries
  knns_ = fx_param_int(module_, "knns", 3);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  if (knns_==0) {
    allknn_.Init(data_mat, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(data_mat, leaf_size_, knns_); 
  }
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");
  if (knns_==0) {
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data_mat.n_cols(),
        data_mat.n_rows(),
        &knns_); 
    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i",knns_);
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  }
 
  // compute the sum of all distances 
  sum_all_distances_=0; 
  for(index_t i=0; i<nearest_distances_.size(); i++) {
    sum_all_distances_+=nearest_distances_[i]*nearest_distances_[i];
  }
  
  eq_lagrange_mult2_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult2_.SetAll(0.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data_mat.n_cols()*data_mat.n_cols();
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);

}

void  BigSdpNmfObjectiveMaxVarIsometric::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
  eq_lagrange_mult1_.Destruct();
	eq_lagrange_mult2_.Destruct();
}

void BigSdpNmfObjectiveMaxVarIsometric::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  // gradient of the variance
  gradient->SetAll(0.0);
  gradient->CopyColumnFromMat(w_offset_, w_offset_, num_of_columns_, coordinates);
	la::Scale(-2.0, gradient);
  
  // Gradient on the dot_product equalities  
  for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_+columns_[i];
    index_t h = h_offset_+rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	  la::AddExpert(new_dimension_, -2*eq_lagrange_mult1_[i]+2*sigma_*diff,
	      coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(new_dimension_, -2*eq_lagrange_mult1_[i]+2*sigma_*diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
  }	
  // Gradient on the local isometries
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[new_dimension_];
    index_t n1=w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2=w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(new_dimension_, point2, point1, a_i_r);

   // equality constraints
   la::AddExpert(new_dimension_,
       -2*eq_lagrange_mult2_[i]+2*dist_diff*sigma2_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
   la::AddExpert(new_dimension_,
        2*eq_lagrange_mult2_[i]-2*dist_diff*sigma2_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  } 
}
void BigSdpNmfObjectiveMaxVarIsometric::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
  for(index_t i=0; i<num_of_columns_; i++) {
	   *objective+=la::Dot(coordinates.n_rows(), 
				 coordinates.GetColumnPtr(w_offset_+i), coordinates.GetColumnPtr(w_offset_+i));
	}
  *objective=-*objective;
}

void BigSdpNmfObjectiveMaxVarIsometric::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		*error+=diff*diff;
  }
  infeasibility1_=math::Pow<1,2>(*error/v_norm_)*100;
  printf("dot_infeasibility:%lg%% ", math::Pow<1,2>(*error/v_norm_)*100); ;
  // local isometry
  double error2=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    *error += math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);
    error2+= math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);

  }
  infeasibility2_=math::Pow<1,2>(error2/sum_all_distances_)*100;
  printf("dist_infeasibility:%lg %%\n", math::Pow<1,2>(error2/sum_all_distances_)*100);
  printf("sigma_ratio:%lg\n", sigma_ratio_);
//  sigma_ratio_=infeasibility2_/infeasibility1_;
}


double BigSdpNmfObjectiveMaxVarIsometric::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
 // dot products 
  for(index_t i=0; i<values_.size(); i++) {
    // equalities
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
    lagrangian+=(sigma_*diff/2-eq_lagrange_mult1_[i])*diff;
  }
  
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma2_/2
        -eq_lagrange_mult2_[i]*dist_diff;
  }
  return lagrangian;
}

void BigSdpNmfObjectiveMaxVarIsometric::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =w_offset_ + columns_[i];
    index_t h =h_offset_ + rows_[i];
    // equalities
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		eq_lagrange_mult1_[i]-=sigma_*diff;
 	}
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                            -nearest_distances_[i];
    eq_lagrange_mult2_[i]-=sigma2_*dist_diff;
  }
}

void BigSdpNmfObjectiveMaxVarIsometric::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMaxVarIsometric::set_sigma(double sigma) {
/* if (infeasibility1_<infeasibility2_) {
    sigma_ratio_*=3;
  } else{
    sigma_ratio_/=3;
  }
*/ 
  if (infeasibility1_==0) {
    sigma_ratio_=0.0;
  } else {
    sigma_ratio_=(infeasibility2_)/(infeasibility1_);
  }

  sigma_=sigma;
  sigma2_=sigma*sigma_ratio_;
}

void BigSdpNmfObjectiveMaxVarIsometric::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_+num_of_columns_);
  double	max_value=0.0;
  for(index_t i=0; i<values_.size(); i++) {
    if (max_value<values_[i]) {
      max_value=values_[i];
    }
  }
  for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random(0, max_value));
		}
	}
}

bool BigSdpNmfObjectiveMaxVarIsometric::IsDiverging(double objective) {
  if (objective<sum_of_furthest_distances_) {
    NONFATAL("Optimization diverging %lg < %lg", objective, sum_of_furthest_distances_);
    return true;
  }
  return false;
} 

bool BigSdpNmfObjectiveMaxVarIsometric::IsOptimizationOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (feasibility_error<desired_feasibility_error_ || 
      fabs(infeasibility1_-previous_infeasibility1_)<0.1)  {
    return true;
  } else {
    previous_infeasibility1_=infeasibility1_;
    return false; 
  }
}

bool BigSdpNmfObjectiveMaxVarIsometric::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);

  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;

}

////////////////////////////////////////////////////////////////////////
void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dimension_=fx_param_int(module_, "new_dimension", 5); 
  grad_tolerance_=fx_param_double(module_, "grad_tolerance", 1);
  desired_feasibility_error_=fx_param_double(module_, "feasibility_error", 0.1);
  // Initializations for the dot products on v
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	h_offset_=0;
  w_offset_=num_of_rows_;
  eq_lagrange_mult1_.Init(values_.size());
  eq_lagrange_mult1_.SetAll(1.0);
  sigma_ratio_=0.0;
  infeasibility1_=100;
  infeasibility2_=1;
  
  // compute the norm of values
  v_norm_=0;
  for(index_t i=0; i<values.size(); i++) {
    v_norm_+=values[i]*values[i];    
  }
  
  // we need to put the data back to a matrix to build the 
  // tree and etc
  Matrix data_mat;
  data_mat.Init(num_of_rows_, num_of_columns_);
  data_mat.SetAll(0.0);
/*  index_t count=0;
 	for(index_t i=0; i<data_mat.n_rows(); i++) {
		for(index_t j=0; j< data_mat.n_cols(); j++) {
			  data_mat.set(i, j, values[count]);
        count++;
		}
	}
*/
  for(index_t i=0; i<rows_.size(); i++) {
    data_mat.set(rows_[i], columns_[i], values_[i]);
  }
  // Initializations for the local isometries
  knns_ = fx_param_int(module_, "knns", 3);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  if (knns_==0) {
    allknn_.Init(data_mat, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(data_mat, leaf_size_, knns_); 
  }
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");
  if (knns_==0) {
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data_mat.n_cols(),
        data_mat.n_rows(),
        &knns_); 
    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i",knns_);
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  }
 
  // compute the sum of all distances 
  sum_all_distances_=0; 
  for(index_t i=0; i<nearest_distances_.size(); i++) {
    sum_all_distances_+=nearest_distances_[i]*nearest_distances_[i];
  }
  
  eq_lagrange_mult2_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult2_.SetAll(1.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data_mat.n_cols()*data_mat.n_cols();
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);

}

void  BigSdpNmfObjectiveMaxVarIsometricRemoveMean::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
  eq_lagrange_mult1_.Destruct();
	eq_lagrange_mult2_.Destruct();
}

void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  gradient->SetAll(0.0);
  
  // gradient of the variance
  Vector mean;
  mean.Init(coordinates.n_rows());
  mean.SetAll(0.0);
  
  for(index_t i=0; i<num_of_columns_; i++) {
    la::AddTo(coordinates.n_rows(), 
              coordinates.GetColumnPtr(w_offset_+i),
              mean.ptr());
  }
  la::Scale(-1.0/num_of_columns_, &mean);
  gradient->CopyColumnFromMat(w_offset_, w_offset_, num_of_columns_, coordinates);
  for(index_t i=0; i<num_of_columns_; i++) {
    la::AddTo(coordinates.n_rows(),
               mean.ptr(),
               gradient->GetColumnPtr(w_offset_+i));
	}
	la::Scale(-2.0*(num_of_columns_-1)/num_of_columns_, gradient);
  
  // Gradient on the dot_product equalities  
  for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_+columns_[i];
    index_t h = h_offset_+rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	  la::AddExpert(new_dimension_, -2*eq_lagrange_mult1_[i]+2*sigma_*diff,
	      coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(new_dimension_, -2*eq_lagrange_mult1_[i]+2*sigma_*diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
  }	
  // Gradient on the local isometries
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[new_dimension_];
    index_t n1=w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2=w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(new_dimension_, point2, point1, a_i_r);

   // equality constraints
   la::AddExpert(new_dimension_,
       -2*eq_lagrange_mult2_[i]+2*dist_diff*sigma2_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
   la::AddExpert(new_dimension_,
        2*eq_lagrange_mult2_[i]-2*dist_diff*sigma2_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  } 
}
void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
  double temp[coordinates.n_rows()];
  Vector mean;
  mean.Init(coordinates.n_rows());
  mean.SetAll(0.0);
  
  for(index_t i=0; i<num_of_columns_; i++) {
    la::AddTo(coordinates.n_rows(), 
              coordinates.GetColumnPtr(w_offset_+i),
              mean.ptr());
  }
  la::Scale(-1.0/num_of_columns_, &mean);
  for(index_t i=0; i<num_of_columns_; i++) {
     la::AddOverwrite(coordinates.n_rows(), mean.ptr(), 
                      coordinates.GetColumnPtr(w_offset_+i),
                      temp); 
	   *objective+=la::Dot(coordinates.n_rows(), 
				                 temp, temp);
	}
  *objective=-*objective;
}

void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		*error+=diff*diff;
  }
  infeasibility1_=math::Pow<1,2>(*error/v_norm_)*100;
  printf("dot_infeasibility:%lg%% ", infeasibility1_); ;
  // local isometry
  double error2=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    *error += math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);
    error2+= math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);

  }
  infeasibility2_=math::Pow<1,2>(error2/sum_all_distances_)*100;
  printf("dist_infeasibility:%lg %%\n", infeasibility2_);
}


double BigSdpNmfObjectiveMaxVarIsometricRemoveMean::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
 // dot products 
  for(index_t i=0; i<values_.size(); i++) {
    // equalities
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
    lagrangian+=(sigma_*diff/2-eq_lagrange_mult1_[i])*diff;
  }
  
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma2_/2
        -eq_lagrange_mult2_[i]*dist_diff;
  }
  return lagrangian;
}

void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =w_offset_ + columns_[i];
    index_t h =h_offset_ + rows_[i];
    // equalities
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		eq_lagrange_mult1_[i]-=sigma_*diff;
 	}
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                            -nearest_distances_[i];
    eq_lagrange_mult2_[i]-=sigma2_*dist_diff;
  }
}

void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::set_sigma(double sigma) {
/* if (infeasibility1_<infeasibility2_) {
    sigma_ratio_*=3;
  } else{
    sigma_ratio_/=3;
  }
*/ 
  if (infeasibility2_==0) {
    sigma_ratio_=0.0;
  } else {
    sigma_ratio_=(infeasibility2_)/(infeasibility1_);
  }
  
  sigma_=sigma;
  sigma2_=sigma*sigma_ratio_;
}

void BigSdpNmfObjectiveMaxVarIsometricRemoveMean::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_+num_of_columns_);
  double	max_value=0.0;
  for(index_t i=0; i<values_.size(); i++) {
    if (max_value<values_[i]) {
      max_value=values_[i];
    }
  }

  for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random(0, max_value));
		}
	}
}

bool BigSdpNmfObjectiveMaxVarIsometricRemoveMean::IsDiverging(double objective) {
  return false;
} 

bool BigSdpNmfObjectiveMaxVarIsometricRemoveMean::IsOptimizationOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (feasibility_error<desired_feasibility_error_)  {
    return true;
  } else {
    return false; 
  }
}

bool BigSdpNmfObjectiveMaxVarIsometricRemoveMean::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);

  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;

}

////////////////////////////////////////////////////////////////////////
void BigSdpNmfObjectiveMaxFurthestIsometric::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dimension_=fx_param_int(module_, "new_dimension", 5); 
  grad_tolerance_=fx_param_double(module_, "grad_tolerance", 1);
  desired_feasibility_error_=fx_param_double(module_, "feasibility_error", 0.1);
  // Initializations for the dot products on v
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	h_offset_=0;
  w_offset_=num_of_rows_;
  eq_lagrange_mult1_.Init(values_.size());
  eq_lagrange_mult1_.SetAll(1.0);
  sigma_ratio_=10.0;
  infeasibility1_=0;
  previous_infeasibility1_=DBL_MAX;
  infeasibility2_=0;
  
  // compute the norm of values
  v_norm_=0;
  for(index_t i=0; i<values.size(); i++) {
    v_norm_+=values[i]*values[i];    
  }
  
  // we need to put the data back to a matrix to build the 
  // tree and etc
  Matrix data_mat;
  data_mat.Init(num_of_rows_, num_of_columns_);
  data_mat.SetAll(0.0);
/*  index_t count=0;
 	for(index_t i=0; i<data_mat.n_rows(); i++) {
		for(index_t j=0; j< data_mat.n_cols(); j++) {
			  data_mat.set(i, j, values[count]);
        count++;
		}
	}
*/
  for(index_t i=0; i<rows_.size(); i++) {
    data_mat.set(rows_[i], columns_[i], values_[i]);
  }
  // Initializations for the local isometries
  knns_ = fx_param_int(module_, "knns", 3);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  if (knns_==0) {
    allknn_.Init(data_mat, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(data_mat, leaf_size_, knns_); 
  }
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");
  if (knns_==0) {
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data_mat.n_cols(),
        data_mat.n_rows(),
        &knns_); 
    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i",knns_);
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  }
  allkfn_.Init(data_mat, leaf_size_, 1); 
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing furthest neighborhoods ...\n");
  from_tree_neighbors.Renew();
  from_tree_distances.Renew();
  allkfn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Furthest Neighbors computed...\n");
  NOTIFY("Consolidating neighbors...\n");
  MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
      from_tree_distances,
      1,
      1,
      &furthest_neighbor_pairs_,
      &furthest_distances_,
      &num_of_furthest_pairs_);

  // compute the sum of all distances 
  sum_all_distances_=0; 
  for(index_t i=0; i<nearest_distances_.size(); i++) {
    sum_all_distances_+=nearest_distances_[i]*nearest_distances_[i];
  }
  
  eq_lagrange_mult2_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult2_.SetAll(0.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data_mat.n_cols()*data_mat.n_cols();
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);

}

void  BigSdpNmfObjectiveMaxFurthestIsometric::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
  eq_lagrange_mult1_.Destruct();
	eq_lagrange_mult2_.Destruct();
}

void BigSdpNmfObjectiveMaxFurthestIsometric::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  // gradient of the variance
  gradient->SetAll(0.0);
  for(index_t i=0; i<furthest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + furthest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + furthest_neighbor_pairs_[i].second;
	  double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double a_i_r[new_dimension_];
    la::SubOverwrite(new_dimension_, point2, point1, a_i_r);
	  // equality constraints
    la::AddTo(new_dimension_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
    la::AddExpert(new_dimension_,
       -1,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  }

	la::Scale(-2.0, gradient);
  
  // Gradient on the dot_product equalities  
  for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_+columns_[i];
    index_t h = h_offset_+rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	  la::AddExpert(new_dimension_, -2*eq_lagrange_mult1_[i]+2*sigma_*diff,
	      coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(new_dimension_, -2*eq_lagrange_mult1_[i]+2*sigma_*diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
  }	
  // Gradient on the local isometries
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[new_dimension_];
    index_t n1=w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2=w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(new_dimension_, point2, point1, a_i_r);

   // equality constraints
   la::AddExpert(new_dimension_,
       -2*eq_lagrange_mult2_[i]+2*dist_diff*sigma2_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
   la::AddExpert(new_dimension_,
        2*eq_lagrange_mult2_[i]-2*dist_diff*sigma2_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  } 
}
void BigSdpNmfObjectiveMaxFurthestIsometric::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
  for(index_t i=0; i<furthest_neighbor_pairs_.size(); i++) {
    index_t n1 = w_offset_ + furthest_neighbor_pairs_[i].first;
    index_t n2 = w_offset_ + furthest_neighbor_pairs_[i].second;
	  double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    *objective +=la::DistanceSqEuclidean(new_dimension_, point1, point2); 
	}
  *objective=-*objective;
}

void BigSdpNmfObjectiveMaxFurthestIsometric::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		*error+=diff*diff;
  }
  infeasibility1_=math::Pow<1,2>(*error/v_norm_)*100;
  printf("dot_infeasibility:%lg%% ", math::Pow<1,2>(*error/v_norm_)*100); ;
  // local isometry
  double error2=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    *error += math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);
    error2+= math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);

  }
  infeasibility2_=math::Pow<1,2>(error2/sum_all_distances_)*100;
  printf("dist_infeasibility:%lg %%\n", math::Pow<1,2>(error2/sum_all_distances_)*100);
  printf("sigma_ratio:%lg\n", sigma_ratio_);
//  sigma_ratio_=infeasibility2_/infeasibility1_;
}


double BigSdpNmfObjectiveMaxFurthestIsometric::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
 // dot products 
  for(index_t i=0; i<values_.size(); i++) {
    // equalities
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
    lagrangian+=(sigma_*diff/2-eq_lagrange_mult1_[i])*diff;
  }
  
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma2_/2
        -eq_lagrange_mult2_[i]*dist_diff;
  }
  return lagrangian;
}

void BigSdpNmfObjectiveMaxFurthestIsometric::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =w_offset_ + columns_[i];
    index_t h =h_offset_ + rows_[i];
    // equalities
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		eq_lagrange_mult1_[i]-=sigma_*diff;
 	}
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                            -nearest_distances_[i];
    eq_lagrange_mult2_[i]-=sigma2_*dist_diff;
  }
}

void BigSdpNmfObjectiveMaxFurthestIsometric::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMaxFurthestIsometric::set_sigma(double sigma) {
/* if (infeasibility1_<infeasibility2_) {
    sigma_ratio_*=3;
  } else{
    sigma_ratio_/=3;
  }
*/ 
/*  if (infeasibility1_==0) {
    sigma_ratio_=0.0;
  } else {
    sigma_ratio_=(infeasibility2_)/(infeasibility1_);
  }
*/
  sigma_=sigma;
  sigma2_=sigma*sigma_ratio_;
}

void BigSdpNmfObjectiveMaxFurthestIsometric::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_+num_of_columns_);
  double	max_value=0.0;
  for(index_t i=0; i<values_.size(); i++) {
    if (max_value<values_[i]) {
      max_value=values_[i];
    }
  }
  for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random(0, max_value));
		}
	}
}

bool BigSdpNmfObjectiveMaxFurthestIsometric::IsDiverging(double objective) {
  if (objective<sum_of_furthest_distances_) {
    NONFATAL("Optimization diverging %lg < %lg", objective, sum_of_furthest_distances_);
    return true;
  }
  return false;
} 

bool BigSdpNmfObjectiveMaxFurthestIsometric::IsOptimizationOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (feasibility_error<desired_feasibility_error_ || 
      fabs(infeasibility1_-previous_infeasibility1_)<0.1)  {
    return true;
  } else {
    previous_infeasibility1_=infeasibility1_;
    return false; 
  }
}

bool BigSdpNmfObjectiveMaxFurthestIsometric::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);

  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;

}

////////////////////////////////////////////////////////////////////////
void NmfObjectiveIsometric::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dimension_=fx_param_int(module_, "new_dimension", 5); 
  grad_tolerance_=fx_param_double(module_, "grad_tolerance", 1);
  desired_feasibility_error_=fx_param_double(module_, "feasibility_error", 0.1);
  // Initializations for the dot products on v
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	h_offset_=0;
  w_offset_=num_of_rows_;
  eq_lagrange_mult2_.Init(values_.size());
  previous_infeasibility2_=DBL_MAX;
  infeasibility2_=0;
  
  // compute the norm of values
  v_norm_=0;
  for(index_t i=0; i<values.size(); i++) {
    v_norm_+=values[i]*values[i];    
  }
  
  // we need to put the data back to a matrix to build the 
  // tree and etc
  Matrix data_mat;
  data_mat.Init(num_of_rows_, num_of_columns_);
  data_mat.SetAll(0.0);
/*  index_t count=0;
 	for(index_t i=0; i<data_mat.n_rows(); i++) {
		for(index_t j=0; j< data_mat.n_cols(); j++) {
			  data_mat.set(i, j, values[count]);
        count++;
		}
	}
*/
  for(index_t i=0; i<rows_.size(); i++) {
    data_mat.set(rows_[i], columns_[i], values_[i]);
  }
  // Initializations for the local isometries
  knns_ = fx_param_int(module_, "knns", 3);
  leaf_size_ = fx_param_int(module_, "leaf_size", 20);
  NOTIFY("Data loaded ...\n");
  NOTIFY("Nearest neighbor constraints ...\n");
  NOTIFY("Building tree with data ...\n");
  if (knns_==0) {
    allknn_.Init(data_mat, leaf_size_, MAX_KNNS); 
  } else {
    allknn_.Init(data_mat, leaf_size_, knns_); 
  }
  NOTIFY("Tree built ...\n");
  NOTIFY("Computing neighborhoods ...\n");
  ArrayList<index_t> from_tree_neighbors;
  ArrayList<double>  from_tree_distances;
  allknn_.ComputeNeighbors(&from_tree_neighbors,
                           &from_tree_distances);
  NOTIFY("Neighborhoods computed...\n");
  if (knns_==0) {
    NOTIFY("Auto-tuning the knn...\n" );
    MaxVarianceUtils::EstimateKnns(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS, 
        data_mat.n_cols(),
        data_mat.n_rows(),
        &knns_); 
    NOTIFY("Optimum knns is %i", knns_);
    fx_format_result(module_, "optimum_knns", "%i",knns_);
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        MAX_KNNS,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  } else { 
    NOTIFY("Consolidating neighbors...\n");
    MaxVarianceUtils::ConsolidateNeighbors(from_tree_neighbors,
        from_tree_distances,
        knns_,
        knns_,
        &nearest_neighbor_pairs_,
        &nearest_distances_,
        &num_of_nearest_pairs_);
  }
 
  // compute the sum of all distances 
  sum_all_distances_=0; 
  for(index_t i=0; i<nearest_distances_.size(); i++) {
    sum_all_distances_+=nearest_distances_[i]*nearest_distances_[i];
  }
  
  eq_lagrange_mult2_.Init(num_of_nearest_pairs_);
  eq_lagrange_mult2_.SetAll(1.0);
  double max_nearest_distance=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    max_nearest_distance=std::max(nearest_distances_[i], max_nearest_distance);
  }
  sum_of_furthest_distances_=-max_nearest_distance*
      data_mat.n_cols()*data_mat.n_cols();
 
  NOTIFY("Lower bound for optimization %lg", sum_of_furthest_distances_);
  fx_format_result(module_, "num_of_constraints", "%i", num_of_nearest_pairs_);
  fx_format_result(module_, "lower_optimal_bound", "%lg", sum_of_furthest_distances_);

}

void  NmfObjectiveIsometric::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
	eq_lagrange_mult2_.Destruct();
}

void NmfObjectiveIsometric::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  // gradient of the variance
  gradient->SetAll(0.0);
  
  // Gradient on the dot_product equalities  
  for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_+columns_[i];
    index_t h = h_offset_+rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	  la::AddExpert(new_dimension_, diff,
	      coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(new_dimension_,  diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
  }	
  // Gradient on the local isometries
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    double a_i_r[new_dimension_];
    index_t n1=w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2=w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    la::SubOverwrite(new_dimension_, point2, point1, a_i_r);

   // equality constraints
   la::AddExpert(new_dimension_,
       -2*eq_lagrange_mult2_[i]+2*dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n1));
   la::AddExpert(new_dimension_,
        2*eq_lagrange_mult2_[i]-2*dist_diff*sigma_,
        a_i_r, 
        gradient->GetColumnPtr(n2));
  } 
}
void NmfObjectiveIsometric::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
  for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_+columns_[i];
    index_t h = h_offset_+rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
    *objective+=diff*diff;
  }
}

void NmfObjectiveIsometric::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    *error += math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);
    *error+= math::Sqr(la::DistanceSqEuclidean(new_dimension_, 
                                               point1, point2) 
                                          -nearest_distances_[i]);

  }
  infeasibility2_=math::Pow<1,2>(*error/sum_all_distances_)*100;
  printf("dist_infeasibility:%lg %%\n", math::Pow<1,2>(*error/sum_all_distances_)*100);
}


double NmfObjectiveIsometric::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
 
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff = la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                           -nearest_distances_[i];
    lagrangian+=dist_diff*dist_diff*sigma_/2
        -eq_lagrange_mult2_[i]*dist_diff;
  }
  return lagrangian;
}

void NmfObjectiveIsometric::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<num_of_nearest_pairs_; i++) {
    index_t n1= w_offset_ + nearest_neighbor_pairs_[i].first;
    index_t n2= w_offset_ + nearest_neighbor_pairs_[i].second;
    double *point1 = coordinates.GetColumnPtr(n1);
    double *point2 = coordinates.GetColumnPtr(n2);
    double dist_diff =la::DistanceSqEuclidean(new_dimension_, point1, point2) 
                            -nearest_distances_[i];
    eq_lagrange_mult2_[i]-=sigma_*dist_diff;
  }
}

void NmfObjectiveIsometric::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void NmfObjectiveIsometric::set_sigma(double sigma) {
  sigma_=sigma;
}

void NmfObjectiveIsometric::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, num_of_rows_+num_of_columns_);
  double	max_value=0.0;
  for(index_t i=0; i<values_.size(); i++) {
    if (max_value<values_[i]) {
      max_value=values_[i];
    }
  }
  for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random(0, max_value));
		}
	}
}

bool NmfObjectiveIsometric::IsDiverging(double objective) {
    return  false;
} 

bool NmfObjectiveIsometric::IsOptimizationOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);
  if (feasibility_error<desired_feasibility_error_ || 
      fabs(infeasibility2_-previous_infeasibility2_)<0.1)  {
    return true;
  } else {
    previous_infeasibility2_=infeasibility2_;
    return false; 
  }
}

bool NmfObjectiveIsometric::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=math::Pow<1,2>(la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr()));
  double feasibility_error;
  ComputeFeasibilityError(coordinates, &feasibility_error);

  if (norm_gradient*step < grad_tolerance_ 
      ||  feasibility_error<desired_feasibility_error_) {
    return true;
  }
  return false;

}



////////////////////////////////////////////////////////////////////////
void ClassicNmfObjective::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dimension_=fx_param_int(module_, "new_dimension", 5); 
  grad_tolerance_=fx_param_double(module_, "grad_tolerance",0.01);
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	h_offset_ = 0;
  w_offset_ = num_of_rows_;
}

void ClassicNmfObjective::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
}

void ClassicNmfObjective::ComputeGradient(Matrix &coordinates, 
    Matrix *gradient) {
  // Gradient on the objective  
	gradient->SetAll(0.0);
  for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
	    la::AddExpert(new_dimension_, diff,
			    coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		  la::AddExpert(new_dimension_, diff, 
				  coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w)); 
  }
  //NOTIFY(" gradient_norm %lg", la::Dot(gradient->n_elements(), 
  //      gradient->ptr(), gradient->ptr())); 
  //NOTIFY("coordinate_norm%lg", la::Dot(coordinates.n_elements(), 
  //      coordinates.ptr(), coordinates.ptr()));  

}

void ClassicNmfObjective::ComputeObjective(Matrix &coordinates, 
    double *objective) {
  *objective=0;	
	for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		  *objective+=diff*diff;
  }
}

void ClassicNmfObjective::ComputeFeasibilityError(Matrix &coordinates, 
    double *error) {
  *error=0;
	for(index_t i=0; i<values_.size(); i++) {
    index_t w = w_offset_ + columns_[i];
    index_t h = h_offset_ + rows_[i];
    double diff=la::Dot(new_dimension_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		  *error+=diff*diff;
  }
}


double ClassicNmfObjective::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
	return lagrangian;
}

void ClassicNmfObjective::UpdateLagrangeMult(Matrix &coordinates) {
}

void ClassicNmfObjective::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void ClassicNmfObjective::set_sigma(double sigma) {
  sigma_=sigma;
}

void ClassicNmfObjective::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dimension_, (num_of_rows_+num_of_columns_));
  for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random(0, 0.1));
		}
	}
}


bool ClassicNmfObjective::IsDiverging(double objective) {
  return false;
} 

bool ClassicNmfObjective::IsOptimizationOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;

}

bool ClassicNmfObjective::IsIntermediateStepOver(
    Matrix &coordinates, Matrix &gradient, double step) {
  double norm_gradient=la::Dot(gradient.n_elements(), 
                               gradient.ptr(), 
                               gradient.ptr());
  if (norm_gradient*step < grad_tolerance_) {
    return true;
  }
  return false;

}




