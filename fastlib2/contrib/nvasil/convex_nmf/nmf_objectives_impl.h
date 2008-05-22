
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
	  la::AddExpert(rank_*new_dim_, -eq_lagrange_mult_[i]+2*sigma_*diff,
				 coordinates.GetColumnPtr(w), gradient->GetColumnPtr(h));	
		la::AddExpert(rank_*new_dim_, -eq_lagrange_mult_[i]+2*sigma_*diff, 
				coordinates.GetColumnPtr(h), gradient->GetColumnPtr(w));
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
		*error+=diff*diff;
 	}
}


double BigSdpNmfObjectiveMinVarIneq::ComputeLagrangian(Matrix &coordinates) {
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

void BigSdpNmfObjectiveMinVarIneq::UpdateLagrangeMult(Matrix &coordinates) {
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

void BigSdpNmfObjectiveMinVarIneq::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void BigSdpNmfObjectiveMinVarIneq::set_sigma(double sigma) {
  sigma_=sigma;
}

void BigSdpNmfObjectiveMinVarIneq::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(rank_, (num_of_rows_+num_of_columns_)*new_dim_);
	for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random());
		}
	}
}


bool BigSdpNmfObjectiveMinVarIneq::IsDiverging(double objective) {
  return false;
} 


