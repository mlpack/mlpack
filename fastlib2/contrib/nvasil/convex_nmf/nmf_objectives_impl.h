
void BigSdpNmfObjective::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	eq_lagrange_mult_.Init(values_.size());
	for(index_t i=0; i<eq_lagrange_mult_.length(); i++) {
    eq_lagrange_mult_[i]=math::Random(-0.0, 1.0);
  }
  rank_=fx_param_int(module_, "rank", 3);
	new_dim_=fx_param_int(module_, "new_dim", 5);
	offset_h_ = num_of_rows_*new_dim_;
}

void BigSdpNmfObjective::Destruct() {
  rows_.Renew();
	columns_.Renew();
  values_.Renew();
	eq_lagrange_mult_.Destruct();
}

void BigSdpNmfObjective::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  gradient->CopyValues(coordinates);
	la::Scale(2.0, gradient);
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

void BigSdpNmfObjective::ComputeObjective(Matrix &coordinates, double *objective) {
  *objective=0;	
	for(index_t i=0; i<coordinates.n_cols(); i++) {
	   *objective+=la::Dot(coordinates.n_rows(), 
				 coordinates.GetColumnPtr(i), coordinates.GetColumnPtr(i));
	}
}

void BigSdpNmfObjective::ComputeFeasibilityError(Matrix &coordinates, double *error) {
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

double BigSdpNmfObjective::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  ComputeObjective(coordinates, &lagrangian);
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		lagrangian+=(sigma_*diff-eq_lagrange_mult_[i])*diff;
	}
	return lagrangian;
}

void BigSdpNmfObjective::UpdateLagrangeMult(Matrix &coordinates) {
  for(index_t i=0; i<values_.size(); i++) {
    index_t w =rows_[i]*new_dim_;
    index_t h =offset_h_+columns_[i]*new_dim_;
    double diff=la::Dot(rank_*new_dim_,
				                coordinates.GetColumnPtr(w),
												coordinates.GetColumnPtr(h))-values_[i];
		eq_lagrange_mult_[i]-=sigma_*diff;
	}
}

void BigSdpNmfObjective::Project(Matrix *coordinates) {
 for(index_t i=0; i<coordinates->n_cols(); i++) {
    for(index_t j=0; j< coordinates->n_rows(); j++) {
	    if (coordinates->get(j, i)<0.0) {
		    coordinates->set(j, i, 0.0);
		  }
    }
	}
 /* Matrix u_mat, vt_mat;
  Vector s;
  success_t success=la::SVDInit(*coordinates, &s, &u_mat, &vt_mat);
  if (success==SUCCESS_FAIL) {
    FATAL("Svd failed...\n");
  }
  index_t positives=0;
  index_t negatives=0;
  for(index_t i=0; i<vt_mat.n_cols(); i++) {
    if (vt_mat.get(0, i)<0) {
     negatives++;
    } else {
      positives++;
    }
  }
  if (positives>=negatives) {
    for(index_t i=0; i<vt_mat.n_cols(); i++) {
      if (vt_mat.get(0, i)<0) {
        vt_mat.set(0, i, vt_mat.get(0, i));
      }
    }
  } else {
    for(index_t i=0; i<vt_mat.n_cols(); i++) {
      if (vt_mat.get(0, i)>0) {
        vt_mat.set(0, i, vt_mat.get(0, i));
      }
    }
  }

  Matrix s_mat;
  s_mat.Init(s.length(), s.length());
  s_mat.SetDiagonal(s);
  Matrix tmp;
  coordinates->Destruct();
  la::MulInit(u_mat, s_mat, &tmp);
  la::MulInit(tmp, vt_mat, coordinates);  
*/
  
}

void BigSdpNmfObjective::set_sigma(double sigma) {
  sigma_=sigma;
}

void BigSdpNmfObjective::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(rank_, (num_of_rows_+num_of_columns_)*new_dim_);
	for(index_t i=0; i<init_data->n_rows(); i++) {
	  for(index_t j=0; j<init_data->n_cols(); j++) {
		  init_data->set(i, j, math::Random());
		}
	}
}


bool BigSdpNmfObjective::IsDiverging(double objective) {
  return false;
} 

