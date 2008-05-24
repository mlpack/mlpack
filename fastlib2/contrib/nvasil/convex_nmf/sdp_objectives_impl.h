/*
 * =====================================================================================
 * 
 *       Filename:  sdp_objectives_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/23/2008 01:29:09 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

void SmallSdpNmf::Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values) {
  module_=module;
	new_dim_=fx_param_int(module_, "new_dim", 5); 
  desired_duality_gap_=fx_param_double(module_, "desired_duality_gap", 1e-4);
  rows_.InitCopy(rows);
	columns_.InitCopy(columns);
	values_.InitCopy(values);
	num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
	num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	// It assumes an N x new_dim_ array, where N=2*(num_rows+num_columns)+ values_.size()
  offset_h_ = num_of_rows_;
  offset_tw_ = offset_h_+num_of_columns_;
  offset_th_ = offset_tw + num_of_rows_;
  offset_v_; = offset_th + num_of_columns;
  number_of_cones_ = values_.size()*k // sdp cones
     +values_size(); // LP cones;
}

void SmallSdpNmf::ComputeGradient(Matrix &coordinates, Matrix *gradient) {
  // from the objective functions
  gradient->CopyValues(objective_factor);
  la::Scale(sigma_, gradient);
  // from the LP cones
  for(index_t i=0; i<values_.size(); i++) {
    index_t v_i=offset_v+i;
    double diff=0;
    for(index_t j=0; j<new_dim_; j++) {
      diff+=coordinates.get(v_i, j);
    }
    diff-=values_[i];
    for(index_t j=0; j<new_dim_; j++) {
      gradient->set(v_i, j, gradient->get(v_i, j)-1.0/diff);
    }     
  } 
  // from the SDP cones
  // determinant=(t1-w*w)*(t2-h*h)-math::Pow<2,1>(w*h-v);
  // dw=-2*w*(t2-h*h)-2*h*(w*h-v);
  // dh=-2*h*(t1-w*w)-2*w*(w*h-v);
  // dt1=t2-h*h;
  // dt2=t1-w*w;
  // dv=+2*(w*h-v);
  for(index_t i=0; i<values_.size(); i++) {
    index_t w_i=rows[i];
    index_t h_i=offset_h+columns_[i];
    index_t t1_i=offset_t1+rows[i];
    index_t t2_i=offset_t2+columns_[i];
    index_t v_i=offset_v+i;
    for(index_t j=0; j<new_dim_; j++) {
      double w=coordinates.get(w_i, j);
      double h=coordinates.get(h_i, j);
      double t1=coordinates.get(t1_i, j);
      double t2=coordinates.get(t2_i, j);
      double v=coordinates.get(v_i, j);
      double t1_minus_ww=(t1-w*w);
      double t2_minus_hh=(t2-h*h);
      double wh_minus_v=(w*h-v);
      double determinant=t1_minus_ww*t2_minus_hh-math::Pow<2,1>(wh_minus_v);
      double dw=(-2*w*(t2_minus_hh)-2*h*(wh_minus_v))/determinant;
      double dh=(-2*h*(t1_minus_ww)-2*w*(wh_minus_v))/determinant;
      double dt1=(t2_minus_hh)/determinant;
      double dt2=(t1_minus_ww)/determinant;
      double dv=+2*(wh_minus_v)/determinant;
      gradient->set(w_i, j, gradient.get(w_i, j)-dw);
      gradient->set(h_i, j, gradient.get(h_i, j)-dh);
      gradient->set(t1_i, j, gradient.get(t1_i, j)-dt1);
      gradient->set(t2_i, j, gradient.get(t1_i, j)-dt2);
      gradient->set(v_i, j, gradient.get(v_i, j)-dv);  
    }  
  }
}

void SmallSdpNmf::ComputeObjective(Matrix &coordinates, double *objective) {
  *objective=0.0;
  for(index_t i=0; i<coordinates.n_elements(); i++) {
    *objective+=objective_factor.ptr()[i]*coordinates.ptr()[i];
  }
}

double SmallSdpNmf::ComputeLagrangian(Matrix &coordinates) {
  double lagrangian=0;
  // from the objective functions
  ComputeObjective(coordinates, &lagrangian);
  lagrangian*=sigma_;
  // from the LP cones
  for(index_t i=0; i<values_.size(); i++) {
    index_t v_i=offset_v+i;
    double diff=0;
    for(index_t j=0; j<new_dim_; j++) {
      diff+=coordinates.get(v_i, j);
    }
    diff-values_[i];
    DEBUG_ERR_MSG_IF(diff<=0, "LP cone is invalid, you are "
       " out of the feasible region");
    lagrangian-=log(diff);   
  } 
  // from the SDP cones
  // determinant=(t1-w*w)*(t2-h*h)-math::Pow<2,1>(w*h-v);
 for(index_t i=0; i<values_.size(); i++) {
    index_t w_i=rows[i];
    index_t h_i=offset_h+columns_[i];
    index_t t1_i=offset_t1+rows[i];
    index_t t2_i=offset_t2+columns_[i];
    index_t v_i=offset_v+i;
    for(index_t j=0; j<new_dim_; j++) {
      double w=coordinates.get(w_i, j);
      double h=coordinates.get(h_i, j);
      double t1=coordinates.get(t1_i, j);
      double t2=coordinates.get(t2_i, j);
      double v=coordinates.get(v_i, j);
      double t1_minus_ww=(t1-w*w);
      double t2_minus_hh=(t2-h*h);
      double wh_minus_v=(w*h-v);
      double determinant=t1_minus_ww*t2_minus_hh-math::Pow<2,1>(wh_minus_v);
      DEBUG_ERR_MSG_IF(determinant<=0, "SDP cone is invalid, you are "
       " out of the feasible region");
      lagrangian-=log(determinant);
    }  
  }
}

void SmallSdpNmf::UpdateLagrangeMult(Matrix &coordinates) {

}

void SmallSdpNmf::Project(Matrix *coordinates) {
  OptUtils::NonNegativeProjection(coordinates);
}

void SmallSdpNmf::set_sigma(double sigma) {
  sigma_=sigma;
}

void SmallSdpNmf::GiveInitMatrix(Matrix *init_data) {
  init_data->Init(new_dim_, 
      2*(num_of_rows_+num_of_columns_)+values_.size());
  for(index_t i=0; i<num_of_rows_+num_of_columns_; i++) {
    for(index_t j=0; j<new_dim_; j++) {
      init_data->set(i, j, math::Random(0.0, 1.0));
    }
  }
  for(index_t i=0; i<values_.size(); i++) {
    index_t w_i=rows[i];
    index_t h_i=offset_h+columns_[i];
    index_t t1_i=offset_t1+rows[i];
    index_t t2_i=offset_t2+columns_[i];
    index_t v_i=offset_v+i;
    for(index_t j=0; j<new_dim_; j++) {
      double w=coordinates.get(w_i, j);
      double h=coordinates.get(h_i, j);
      double t1=coordinates.get(t1_i, j);
      double t2=coordinates.get(t2_i, j);
      double v=coordinates.get(v_i, j);
     
      // ensure that Sum w_ij*hij > v_ij 
      coordinates.set(v_i, j, std::max(w*h+math::Random(), values_[i]));
      coordinate.set(t1_i, j, fabs(w*h-v)+w*w+math::Random());
      coordinate.set(t2_i, j, fabs(w*h-v)+h*h+math::Random());
    }
  }  
}

bool SmallSdpNmf::IsDiverging(double objective) {
  return false;
}

bool SmallSdpNmf::IsOptimizationOver(Matrix &coordinates, Matrix &gradient) {
  if (number_of_cones_/sigma_ < desired_duality_gap_) {
    return true;
  } else {
    return false;
  }
}

bool SmallSdpNmf::IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient) {
  double norm_gradient = la::Dot(gradient.n_elements(), 
      gradient.ptr(), gradient.ptr());
  if (norm_gradient < gradient_tolerance_) {
    return true;
  } else {
    return false;
  }
}
 
