/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file sparsepca.cc
 *
 * Implementation of the functions of the class SparsePCA
 * declared in the file 'sparsepca.h'
 *
 */

#include "sparsepca.h"

void SparsePCA::SparsifyMicroArray() {

  Matrix alpha_mat, beta_mat, temp_mat;
  index_t k = fx_param_int_req(spca_module_, "K");
  index_t d = fx_param_int_req(spca_module_, "D");
  index_t MAX_ITERS = fx_param_int(spca_module_, "MAX_ITERS", 200);
  double EPS = fx_param_double(spca_module_, "EPS", 1.0e-3);


  // alpha <- V[:,1:k]
  v().MakeColumnSlice(0, k, &temp_mat);
  alpha_mat.Copy(temp_mat);

  beta_mat.Init(d, k);
  beta_mat.SetZero();

  // for i <- 1 to K
  //    y <- X_centered^T * X_centered * alpha_i ;
  //    beta_i <- Soft_(y, lambda_1i);

  for (index_t i = 0; i < k; i++) {

    Vector alpha_i_vec, beta_i_vec, y_vec, temp_vec;

    alpha_mat.MakeColumnVector(i, &alpha_i_vec);
    la::MulInit(alpha_i_vec, x_centered(), &temp_vec);
    la::MulInit(x_centered(), temp_vec, &y_vec);

    beta_mat.MakeColumnVector(i, &beta_i_vec);
    Soft_(y_vec, &beta_i_vec, lambda_L1(i));
  }

  Matrix x_trans_x_mat, temp_beta_mat, temp_x_trans_mat;

  la::TransposeInit(x_centered(), &temp_x_trans_mat);
  la::MulInit(x_centered(), temp_x_trans_mat, &x_trans_x_mat);

  // temp_beta_i = beta_i / || beta_i ||, i = 1, 2, ..., K

  temp_beta_mat.Copy(beta_mat);
  for (index_t i = 1; i < k; i++) {
    Vector temp_beta_i_vec;

    temp_beta_mat.MakeColumnVector(i, &temp_beta_i_vec);

    double norm_temp_beta_i = sqrt(la::Dot(temp_beta_i_vec, 
					   temp_beta_i_vec));
    if (norm_temp_beta_i == 0.0) {
      norm_temp_beta_i = 1;
    }

    la::Scale(norm_temp_beta_i, &temp_beta_i_vec);
  }

  index_t iters = 0;
  double difference = 1.0;

  while ((iters < MAX_ITERS) && (difference > EPS)) {

    ++iters;

    // U * L * V^T = SVD(X^T * X * beta)
    // alpha <- U * V^T

    Matrix u_mat, v_t_mat;
    Vector dummy_vec;
    la::MulOverwrite(x_trans_x_mat, beta_mat, &alpha_mat);
    success_t svd_op = la::SVDInit(alpha_mat, &dummy_vec, 
				   &u_mat, &v_t_mat);
    DEBUG_ASSERT_MSG(svd_op == SUCCESS_PASS, 
		 "SVD of X^T*X*beta failed\n");
    la::MulOverwrite(u_mat, v_t_mat, &alpha_mat);

    // for i <- 1 to K
    //     y = X_centered^T * X_centered * alpha_i
    //     beta_i <- Soft_(y, lambda_1i);

    for (index_t i = 0; i < k; i++) {
      Vector alpha_i_vec, beta_i_vec, y_vec, temp_vec;
	
      alpha_mat.MakeColumnVector(i, &alpha_i_vec);
      la::MulInit(alpha_i_vec, x_centered(), &temp_vec);
      la::MulInit(x_centered(), temp_vec, &y_vec);
	
      beta_mat.MakeColumnVector(i, &beta_i_vec);
      Soft_(y_vec, &beta_i_vec, lambda_L1(i));
    }

    // beta_2_i = beta_i / || beta_i ||, i = 1, 2, ..., K

    Matrix beta_2_mat;
    beta_2_mat.Copy(beta_mat);
    for (index_t i = 1; i < k; i++) {
      Vector beta_2_i_vec;
	
      beta_2_mat.MakeColumnVector(i, &beta_2_i_vec);

      double norm_beta_i = sqrt(la::Dot(beta_2_i_vec, 
					beta_2_i_vec));
      if (norm_beta_i == 0.0) {
	norm_beta_i = 1;
      }
	
      la::Scale(norm_beta_i, &beta_2_i_vec);
    }

    // Noting the difference between previous beta 
    // and the present beta
    // difference <- max( |beta_2 - temp_beta| );

    la::SubFrom(beta_2_mat, &temp_beta_mat);
    difference = MaxAbsValue_(temp_beta_mat);

    // temp_beta <- beta_2

    temp_beta_mat.CopyValues(beta_2_mat);
  }

  set_v_sparse(temp_beta_mat);

  // R <- qr.R(qr(X_centered * V_sparse))
  // explainedvariance <- sum(diag(R^2))

  Matrix x_centered_v_sparse_mat, q_mat, r_mat;

  la::MulInit(temp_x_trans_mat, v_sparse(), &x_centered_v_sparse_mat);
  success_t qr_op = la::QRInit(x_centered_v_sparse_mat, &q_mat, &r_mat);

  DEBUG_ASSERT_MSG(qr_op == SUCCESS_PASS, 
		   "Sparsify() : Q * R = X * V_sparse failed\n");

  DEBUG_ASSERT_MSG(k == r_mat.n_cols(), "Sparsify() : K != dim(R)\n");
  double var = 0.0;
  for (index_t i = 0; i < r_mat.n_cols(); i++) {
    double temp_var = r_mat.get(i, i);
    var += temp_var * temp_var;
  }
  set_var_explained(var);

  return;
}

void SparsePCA::Soft_(Vector& a_vec, Vector *beta_vec, double lambda_1) {

  Vector temp_beta_vec, a_sign_vec;
  Matrix temp_a_sign_mat;
  index_t vec_length = a_vec.length();

  // beta <- |a| - lambda_1 / 2
  // beta <- (beta + |beta|) / 2
  temp_beta_vec.Init(vec_length);
  for (index_t i = 0; i < vec_length; i++) {
    temp_beta_vec.ptr()[i] = fabs(a_vec.get(i)) - lambda_1 / 2;
    temp_beta_vec.ptr()[i] = fabs(temp_beta_vec.get(i)) + temp_beta_vec.get(i);
  }

  la::Scale(0.5, &temp_beta_vec);

  // beta <- sign(a) * beta
  a_sign_vec.Init(vec_length);
  for (index_t i = 0; i < vec_length; i++) {
    a_sign_vec.ptr()[i] = a_vec.get(i) > 0 ? 1 : -1;
  }

  temp_a_sign_mat.Init(vec_length, vec_length);
  temp_a_sign_mat.SetZero();
  temp_a_sign_mat.SetDiagonal(a_sign_vec);

  la::MulOverwrite(temp_a_sign_mat, temp_beta_vec, beta_vec);

  return;
}

void SparsePCA::Sparsify() {

  Matrix alpha_mat, beta_mat, temp_mat;
  index_t k = fx_param_int_req(spca_module_, "K");
  index_t d = fx_param_int_req(spca_module_, "D");
  index_t MAX_ITERS = fx_param_int(spca_module_, "MAX_ITERS", 200);
  double EPS = fx_param_double(spca_module_, "EPS", 1.0e-3);


  // alpha <- V[:,1:k]
  v().MakeColumnSlice(0, k, &temp_mat);
  alpha_mat.Copy(temp_mat);

  beta_mat.Init(d, k);
  beta_mat.SetZero();

  // for i <- 1 to K
  //    y <- X_centered * alpha_i ;
  //    beta_i <- SolveForBeta_(X_centered, y, lambda_2, lambda_1i);

  for (index_t i = 0; i < k; i++) {

    Vector alpha_i_vec, beta_i_vec, y_vec;

    alpha_mat.MakeColumnVector(i, &alpha_i_vec);
    la::MulInit(alpha_i_vec, x_centered(), &y_vec);

    beta_mat.MakeColumnVector(i, &beta_i_vec);
    SolveForBeta_(y_vec, &beta_i_vec, lambda_quad(), lambda_L1(i));
  }

  Matrix x_trans_x_mat, temp_beta_mat, temp_x_trans_mat;

  la::TransposeInit(x_centered(), &temp_x_trans_mat);
  la::MulInit(x_centered(), temp_x_trans_mat, &x_trans_x_mat);

  // temp_beta_i = beta_i / || beta_i ||, i = 1, 2, ..., K

  temp_beta_mat.Copy(beta_mat);
  for (index_t i = 1; i < k; i++) {
    Vector temp_beta_i_vec;

    temp_beta_mat.MakeColumnVector(i, &temp_beta_i_vec);

    double norm_temp_beta_i = sqrt(la::Dot(temp_beta_i_vec, 
					   temp_beta_i_vec));
    if (norm_temp_beta_i == 0.0) {
      norm_temp_beta_i = 1;
    }

    la::Scale(norm_temp_beta_i, &temp_beta_i_vec);
  }

  index_t iters = 0;
  double difference = 1.0;

  while ((iters < MAX_ITERS) && (difference > EPS)) {

    ++iters;

    // U * L * V^T = SVD(X^T * X * beta)
    // alpha <- U * V^T

    Matrix u_mat, v_t_mat;
    Vector dummy_vec;
    la::MulOverwrite(x_trans_x_mat, beta_mat, &alpha_mat);
    success_t svd_op = la::SVDInit(alpha_mat, &dummy_vec, 
				   &u_mat, &v_t_mat);
    DEBUG_ASSERT_MSG(svd_op == SUCCESS_PASS, 
		 "SVD of X^T*X*beta failed\n");
    la::MulOverwrite(u_mat, v_t_mat, &alpha_mat);

    // for i <- 1 to K
    //     y = X_centered * alpha_i
    //     beta_i <- SolveForBeta_(X_centered, y, lambda_2, lambda_1i);

    for (index_t i = 0; i < k; i++) {
      Vector alpha_i_vec, beta_i_vec, y_vec;
	
      alpha_mat.MakeColumnVector(i, &alpha_i_vec);
      la::MulInit(alpha_i_vec, x_centered(), &y_vec);
	
      beta_mat.MakeColumnVector(i, &beta_i_vec);
      SolveForBeta_(y_vec, &beta_i_vec, lambda_quad(), lambda_L1(i));
    }

    // beta_2_i = beta_i / || beta_i ||, i = 1, 2, ..., K

    Matrix beta_2_mat;
    beta_2_mat.Copy(beta_mat);
    for (index_t i = 1; i < k; i++) {
      Vector beta_2_i_vec;
	
      beta_2_mat.MakeColumnVector(i, &beta_2_i_vec);

      double norm_beta_i = sqrt(la::Dot(beta_2_i_vec, 
					beta_2_i_vec));
      if (norm_beta_i == 0.0) {
	norm_beta_i = 1;
      }
	
      la::Scale(norm_beta_i, &beta_2_i_vec);
    }

    // Noting the difference between previous beta 
    // and the present beta
    // difference <- max( |beta_2 - temp_beta| );

    la::SubFrom(beta_2_mat, &temp_beta_mat);
    difference = MaxAbsValue_(temp_beta_mat);

    // temp_beta <- beta_2

    temp_beta_mat.CopyValues(beta_2_mat);
  }

  set_v_sparse(temp_beta_mat);

  // R <- qr.R(qr(X_centered * V_sparse))
  // explainedvariance <- sum(diag(R^2))

  Matrix x_centered_v_sparse_mat, q_mat, r_mat;

  la::MulInit(temp_x_trans_mat, v_sparse(), &x_centered_v_sparse_mat);
  success_t qr_op = la::QRInit(x_centered_v_sparse_mat, &q_mat, &r_mat);

  DEBUG_ASSERT_MSG(qr_op == SUCCESS_PASS, 
		   "Sparsify() : Q * R = X * V_sparse failed\n");

  DEBUG_ASSERT_MSG(k == r_mat.n_cols(), "Sparsify() : K != dim(R)\n");
  double var = 0.0;
  for (index_t i = 0; i < r_mat.n_cols(); i++) {
    double temp_var = r_mat.get(i, i);
    var += temp_var * temp_var;
  }
  set_var_explained(var);

  return;
}

void SparsePCA::SolveForBeta_(Vector& y_vec, Vector *beta_vec,
			      double lambda_2, double lambda_1) {

  datanode *solver_module = fx_submodule(spca_module_, "BetaSolver");
  index_t n = x_centered().n_cols(), d = x_centered().n_rows();
  index_t max_vars = -1;
  double d1 = sqrt(lambda_2), d2 = 1.0 / sqrt(1 + lambda_2);
  Vector c_vec, residual_1_vec, residual_2_vec;
  double EPS = fx_param_double(solver_module, "EPS", 1.0e-3);
  ArrayList<double> penalties;

  c_vec.Init(d);
  residual_1_vec.Init(n);
  residual_2_vec.Init(d);

  if (lambda_2 > 0) {
    max_vars = d;
  }
  if (lambda_2 == 0) {
    if (d == n) {
      max_vars = d;
    }
    else {
      max_vars = (d < n - 1) ? d : n - 1;
    }
  }
  index_t MAX_STEPS = fx_param_int(solver_module, "MAX_STEPS", 50*max_vars);

  // Cvec <- X^T * y / sqrt(1 + lambda_2);

  la::MulExpert(d2, x_centered(), y_vec, 0.0, &c_vec);

  // residuals <- [ y 0_D ]

  residual_1_vec.CopyValues(y_vec);
  residual_2_vec.SetZero();

  // penalty <- max(abs(Cvec))
  penalties.Init(1);
  penalties[0] = MaxAbsValue_(c_vec);

  if (penalties[0] * 2 / d2 <= lambda_1) {

    beta_vec->SetZero();

  }
  else {

    beta_vec->SetZero();
      
    // active <- EMPTY, ignore <- EMPTY

    index_t active = 1, ignore = 2, inactive = 0, undecided = 3;
    index_t *beta_var_state;
    ArrayList<index_t> active_set;

    beta_var_state = (index_t*) malloc (d * sizeof(index_t));
    for (index_t i = 0; i < d; i++) {
      beta_var_state[i] = inactive;
    }
    active_set.Init(0);

    // Sign <- NULL

    double *beta_sign;

    beta_sign = (double*) malloc (d * sizeof(double));
    for (index_t i = 0; i < d; i++) {
      beta_sign[i] = 0;
    }

    // R <- NULL

    Matrix r_mat;
    r_mat.Init(0, 0);
    index_t rank_r = 0;


    // drops = FALSE

    bool drops = 0;

    index_t iters = 0;

    Vector c_inactive_vec;
    c_inactive_vec.Init(0);
    Matrix x_trans_mat, x_active_mat;

    la::TransposeInit(x_centered(), &x_trans_mat);
    x_active_mat.Init(0,0);

    // while iters < MAX_STEPS & |A| < max_vars - |I|
    while ((iters < MAX_STEPS) && 
	   (SubsetLength_(beta_var_state, d, active) 
	    < max_vars - SubsetLength_(beta_var_state, d, ignore))) {

      ++iters;

      // INACTIVE_SET <- (A union I)^C
      // C_INACTIVE <- {C_j}, j \in INACTIVE_SET
      // C_MAX <- max(abs(C_INACTIVE))
	
      MakeSubvector_(c_vec, beta_var_state, inactive, &c_inactive_vec);
      // index_t c_max_index;
      double c_max = MaxAbsValue_(c_inactive_vec);

      if (!drops) {

	// new_indices <- {j : j \in INACTIVE_SET & |C_j| == C_MAX}
	// C_INACTIVE <- C_INACTIVE - {C_j : j \in new_indices}

	ArrayList<index_t> new_indices;
	new_indices.Init(0);

	for (index_t i = 0, j = 0; i < d; i++) {
	  if (beta_var_state[i] == inactive) {
	    if (fabs(c_inactive_vec.get(j)) == c_max) {
	      beta_var_state[i] = undecided;
	      *(new_indices.AddBack()) = i;
	    }
	    j++;
	  }
	}
	MakeSubvector_(c_vec, beta_var_state, inactive, &c_inactive_vec);

	for (index_t i = 0; i < new_indices.size(); i++) {

	  // R <- updateR(x_i, X_ACTIVE, R, lambda_2)

	  MakeSubmatrix_(x_trans_mat, active_set, &x_active_mat);
	  Vector x_i_vec;
	  x_trans_mat.MakeColumnVector(new_indices[i], &x_i_vec);

	  UpdateR_(x_i_vec, x_active_mat, lambda_2, &rank_r, &r_mat);

	  // if rank(R) == |A|

	  if(rank_r == SubsetLength_(beta_var_state, d, active)) {

	    // IGNORE <- IGNORE union {new_indices[i]}
	    DEBUG_ASSERT_MSG(beta_var_state[new_indices[i]] == undecided, 
			     "SolveForBeta_() : Addition to Ignore list failed\n");
	    beta_var_state[new_indices[i]] = ignore;
	  }
	  else {

	    // ACTIVE <- ACTIVE union {new_indices[i]}
	    DEBUG_ASSERT_MSG(beta_var_state[new_indices[i]] == undecided, 
			     "SolveForBeta_() : Addition to Active list failed\n");
	    beta_var_state[new_indices[i]] = active;
	    *(active_set.AddBack()) = new_indices[i];

	    // Sign <- Sign union sign(Cvec[new_indices[i]])
	    DEBUG_ASSERT_MSG(c_vec.get(new_indices[i]) != 0.0, 
			     "SolveForBeta() : C_vec(i) == 0\n");
	    beta_sign[new_indices[i]] = c_vec.get(new_indices[i]) > 0 ? 1 : -1;
	  }
	} // end for
      } // end if

      // G_i <- v such that R^T * R * v = Sign_ACTIVE
      Vector beta_sign_vec, beta_sign_active_vec;
      Matrix r_trans_mat, r_trans_r_mat;
      Vector g_i_vec;

      beta_sign_vec.Copy(beta_sign, d);
      beta_sign_active_vec.Init(0);
      la::TransposeInit(r_mat, &r_trans_mat);
      la::MulInit(r_trans_mat, r_mat, &r_trans_r_mat);

      // Making Sign_ACTIVE
      MakeSubvector_(beta_sign_vec, active_set, &beta_sign_active_vec);
      
      success_t solve_op = la::SolveInit(r_trans_r_mat, 
					 beta_sign_active_vec, &g_i_vec);

      DEBUG_ASSERT_MSG(solve_op == SUCCESS_PASS, 
		       "SolveForBeta_() : R^T * R * G_i = Sign_ACTIVE failed\n");

      // A <- 1 / sqrt( sum_{j=1}^{|ACTIVE|} G_i_j * Sign_ACTIVE_j
      Matrix diag_g_i_mat;
      Vector one_vec, temp_g_i_sign_vec;
      double a, temp_sum_g_i_sign;

      diag_g_i_mat.Init(active_set.size(), active_set.size());
      diag_g_i_mat.SetDiagonal(g_i_vec);
      one_vec.Init(active_set.size());
      one_vec.SetAll(1.0);
      la::MulInit(diag_g_i_mat, beta_sign_active_vec, &temp_g_i_sign_vec);
      temp_sum_g_i_sign = la::Dot(temp_g_i_sign_vec, one_vec);
      a = 1 / sqrt(temp_sum_g_i_sign);

      // w <- A * G_i 
      Vector w_vec;
      la::ScaleInit(a, g_i_vec, &w_vec);

      // u1 <- X_ACTIVE * w * d2
      // u2 <- [0]_m
      // u2_ACTIVE <- d1 * d2 * w
      Vector u1_vec, u2_vec;

      MakeSubmatrix_(x_trans_mat, active_set, &x_active_mat);
      la::MulInit(x_active_mat, w_vec, &u1_vec);
      la::Scale(d2, &u1_vec);
      u2_vec.Init(d);
      u2_vec.SetZero();
      for (index_t i = 0; i < active_set.size(); i++) {
	u2_vec.ptr()[active_set[i]] = d1 * d2 * w_vec.get(i);
      }

      if (lambda_2 > 0) {
	max_vars = d - SubsetLength_(beta_var_state, d, ignore);
      }
      if (lambda_2 == 0) {
	max_vars = (d - SubsetLength_(beta_var_state, d, ignore)) < (n - 1) ?
	  (d - SubsetLength_(beta_var_state, d, ignore)) : (n - 1);
      }

      double gam_hat;
      Vector a_vec;

      a_vec.Init(SubsetLength_(beta_var_state, d, inactive));

      if (SubsetLength_(beta_var_state, d, active) == 
	  max_vars - SubsetLength_(beta_var_state, d, ignore)) {

	gam_hat = c_max / a;
      }
      else {

	// a <- u1 * X_INACTIVE + d1 * u2_INACTIVE * d2
	Matrix x_inactive_mat;
	Vector u2_inactive_vec;

	x_inactive_mat.Init(0, 0);
	u2_inactive_vec.Init(0);

	MakeSubmatrix_(x_trans_mat, beta_var_state, inactive, &x_inactive_mat);
	la::MulOverwrite(u1_vec, x_inactive_mat, &a_vec);

	MakeSubvector_(u2_vec, beta_var_state, inactive, &u2_inactive_vec);
	la::AddExpert(d1 * d2, u2_inactive_vec, &a_vec);

	// gam <- { (Cmax - c_i)/(A - a_i), (Cmax + c_i)/(A + a_i) } 
	// for all i in INACTIVE
	// gam_hat <- min( gam (gam > EPS), Cmax / A)
	double temp_val, gam = FLT_MAX;
	for (index_t i = 0; i < a_vec.length(); i++) {

	  temp_val = (c_max - c_inactive_vec.get(i)) / (a - a_vec.get(i));
	  if (temp_val > EPS) {
	    if (temp_val < gam) {
	      gam = temp_val;
	    }
	  }

	  temp_val = (c_max + c_inactive_vec.get(i)) / (a - a_vec.get(i));
	  if (temp_val > EPS) {
	    if (temp_val < gam) {
	      gam = temp_val;
	    }
	  }
	}
	temp_val = c_max / a;
	if (temp_val > gam) {
	  gam_hat = gam;
	}
	else {
	  gam_hat = temp_val;
	}
      }

      // z1 <- - beta_ACTIVE / w
      // zmin <- min (z1[z1 > EPS], gam_hat)
      // if zmin < gam_hat
      //    gam_hat <- zmin
      //    drop_indices <- z1 == zmin
      //    drop <- TRUE
      // else
      //    drop <- FALSE

      Vector beta_active_vec, z1_vec;

      beta_active_vec.Init(0);
      z1_vec.Init(SubsetLength_(beta_var_state, d, active));

      MakeSubvector_(*beta_vec, active_set, &beta_active_vec);
      for (index_t i = 0; i < z1_vec.length(); i++) {
	z1_vec.ptr()[i] = - beta_active_vec.get(i) / w_vec.get(i);
      }

      double zmin = gam_hat;
      for (index_t i = 0; i < z1_vec.length(); i++) {
	if (z1_vec.get(i) > EPS) {
	  if (z1_vec.get(i) < zmin) {
	    zmin = z1_vec.get(i);
	  }
	}
      }

      ArrayList<index_t> drop_indices;
      drop_indices.Init(0);

      if (zmin < gam_hat) {
	drops = 1;
	gam_hat = zmin;
	for (index_t i = 0; i < z1_vec.length(); i++) {
	  if (zmin == z1_vec.get(i)) {
	    *(drop_indices.AddBack()) = i;
	  }
	}
      }
      else {
	drops = 0;
      }

      // temp_beta <- beta
      Vector temp_beta_vec;
      temp_beta_vec.Copy(*beta_vec);

      // beta_ACTIVE <- beta_ACTIVE + gam_hat * w
      for (index_t i = 0; i < active_set.size(); i++) {
	beta_vec->ptr()[active_set[i]] = 
	  beta_vec->get(active_set[i]) + gam_hat * w_vec.get(i);
      }

      // residuals <- residuals - gam_hat * u
      la::AddExpert(-gam_hat, u1_vec, &residual_1_vec);
      la::AddExpert(-gam_hat, u2_vec, &residual_2_vec);

      // Cvec <- residuals[1:N] * X + d1 * residuals[N+1:N+D] * d2
      la::MulOverwrite(residual_1_vec, x_trans_mat, &c_vec);
      la::AddExpert(d1 * d2, residual_2_vec, &c_vec);

      // penalties <- penalties union penalties[k-1] - abs(gam_hat*A)
      *(penalties.AddBack()) = penalties[iters - 1] - fabs(gam_hat * a);

      // exit strategy
      if (*(penalties.end()) * 2 / d2 <= lambda_1) {
	double s1 = *(penalties.PopBackPtr()) * 2 / d2;
	double s2 = *(penalties.end()) * 2 / d2;
	double factor_1 = (s2 - lambda_1) / (s2 - s1);
	double factor_2 = (lambda_1 - s1) / (s2 - s1);

	la::Scale(factor_1, beta_vec);
	la::AddExpert(factor_2, temp_beta_vec, beta_vec);
	la::Scale(d2, beta_vec);
	return;
      }

      // if drops
      //    for i in reverse(drop_indices)
      //        drop i in active set
      //        drop i from sign set
      //        downdateR(R, i)
      if (drops) {
	for (index_t i = drop_indices.size(); i > 0; i--) {
	  index_t id = drop_indices[i - 1];
	  DowndateR_(&r_mat, id);
	  --rank_r;
	}

	ArrayList<index_t> temp_active_set;
	temp_active_set.Init(active_set.size() - drop_indices.size());

	for (index_t i = 0, j = 0; i < active_set.size(); i++) {
	  if (j < drop_indices.size()) {
	    if (i == drop_indices[j]) {
	      beta_vec->ptr()[active_set[i]] = 0;
	      beta_sign[active_set[i]] = 0;
	      beta_var_state[active_set[i]] = inactive;
	      j++;
	    }
	    else {
	      temp_active_set[i - j] = active_set[i];
	    }
	  }
	  else {
	    temp_active_set[i - j] = active_set[i];
	  }
	}

	active_set.Destruct();
	active_set.Init(temp_active_set.size());

	for (index_t i = 0; i < active_set.size(); i++) {
	  active_set[i] = temp_active_set[i];
	}
      }// end if

    }// end while
  }// end else

  return;
}

void SparsePCA::UpdateR_(Vector& a_vec, Matrix& old_mat, 
			 double lambda, index_t *rank_r, Matrix *r_mat) {

  double a_trans_a = (la::Dot(a_vec, a_vec) + lambda) / (1 + lambda);
  double a_norm = sqrt(a_trans_a);
  double EPS = 1.0e-5;

  if (*rank_r == 0) {
    r_mat->Destruct();
    r_mat->Init(1, 1);
    r_mat->set(0, 0, a_norm);
    *rank_r = 1;
    return;
  }

  Vector a_old_vec;
  la::MulInit(a_vec, old_mat, &a_old_vec);

  Matrix r_trans_mat;
  la::TransposeInit(*r_mat, &r_trans_mat);

  Vector r_vec;
  success_t solve_op = la::SolveInit(r_trans_mat, a_old_vec, &r_vec);

  DEBUG_ASSERT_MSG(solve_op == SUCCESS_PASS, 
		   "UpdateR_():system of equation could not be solved\n");

  double rpp = a_trans_a - la::Dot(r_vec, r_vec);
  if (rpp < EPS) {
    return;
  }
  else {
    rpp = sqrt(rpp);
    *rank_r = *rank_r + 1;
  }

  Matrix temp_r_mat;
  temp_r_mat.Init(*rank_r, *rank_r);
  temp_r_mat.SetZero();

  for (index_t i = 0; i < *rank_r - 1; i++) {
    for(index_t j = 0; j < i + 1; j++) {
      temp_r_mat.set(j, i, r_mat->get(j, i));
    }
  }
  for (index_t i = 0; i < *rank_r - 1; i++) {
    temp_r_mat.set(i, *rank_r - 1, r_vec.get(i));
  }
  temp_r_mat.set(*rank_r - 1, *rank_r - 1, rpp);

  r_mat->Destruct();
  r_mat->Copy(temp_r_mat);

  return;
}

void SparsePCA::DowndateR_(Matrix *r_mat, index_t col_id) {

  index_t rank = r_mat->n_cols();
  if(rank == 1) {
    r_mat->Destruct();
    r_mat->Init(0, 0);
    return;
  }
  else {
    Matrix temp_r_mat;
    temp_r_mat.Init(rank - 1, rank - 1);
    temp_r_mat.SetZero();

    for (index_t i = 0; i < col_id; i++) {
      for (index_t j = 0; j < i + 1; j++) {
	temp_r_mat.set(j, i, r_mat->get(j, i));
      }
    }

    for (index_t i = col_id + 1; i < rank; i++) {
      double a = r_mat->get(i - 1, i);
      double b = r_mat->get(i, i);
      double x, y, z;

      if (fabs(b) > fabs(a)) {
	x = - a / b;
	y = 1 / sqrt(1 + x*x);
	z = x * y;
      }
      else {
	x = -b / a;
	z = 1 / sqrt(1 + x*x);
	y = z * x;
      }
      r_mat->set(i - 1, i, (z*a - y*b));
      r_mat->set(i, i, (y*a - z*b));

      for (index_t j = i + 1; j < rank; ++j) {
	a = r_mat->get(i - 1, j);
	b = r_mat->get(i, j);
	r_mat->set(i - 1, j, (z*a - y*b));
	r_mat->set(i, j, (y*a - z*b));
      }
    }

    for (index_t i = col_id; i < rank - 1; i++) {
      for (index_t j = 0; j < i + 1; j++) {
	temp_r_mat.set(j, i, r_mat->get(j, i+1));
      }
    }

    r_mat->Destruct();
    r_mat->Copy(temp_r_mat);

    return;
  }
}

double SparsePCA::MaxAbsValue_(Matrix& mat) {

  double max_val = 0.0;
  double *ptr_mat, *ptr_end;
    
  ptr_mat = mat.ptr();
  ptr_end = mat.ptr() + mat.n_elements();
    
  for (; ptr_mat < ptr_end; ptr_mat++) {
    if (max_val < fabs(*ptr_mat)) {
      max_val = fabs(*ptr_mat);
    }
  }

  return max_val;
}

double SparsePCA::MaxAbsValue_(Vector& vec) {

  double max_val = 0.0;
  double *ptr_vec, *ptr_end;
    
  ptr_vec = vec.ptr();
  ptr_end = vec.ptr() + vec.length();
    
  for (; ptr_vec < ptr_end; ptr_vec++) {
    if (max_val < fabs(*ptr_vec)) {
      max_val = fabs(*ptr_vec);
    }
  }

  return max_val;
}

double SparsePCA::MaxAbsValue_(Vector& vec, index_t *max_index) {

  double max_val = 0.0;
  double *ptr_vec, *ptr_end;
    
  ptr_vec = vec.ptr();
  ptr_end = vec.ptr() + vec.length();
    
  for (index_t i = 0; ptr_vec < ptr_end; ptr_vec++, i++) {
    if (max_val < fabs(*ptr_vec)) {
      max_val = fabs(*ptr_vec);
      *max_index = i;
    }
  }

  return max_val;
}

index_t SparsePCA::SubsetLength_(index_t *set, index_t set_size,
				 index_t subset_marker) {

  index_t length = 0;

  for (index_t i = 0; i < set_size; i++) {
    if (set[i] == subset_marker) {
      ++length;
    }
  }

  return length;
}

void SparsePCA::MakeSubvector_(Vector& vec, index_t *set, 
			       index_t subset_marker, Vector *subvec) {

  index_t set_length = vec.length();
  double *subarray;

  subarray = (double*)malloc(set_length * sizeof(double));

  index_t j = 0;
  for (index_t i = 0; i < set_length; i++) {
    if (set[i] == subset_marker) {
      subarray[j] = vec.get(i);
      ++j;
    }
  }

  subvec->Destruct();
  subvec->Copy(subarray, j);

  return;
}

void SparsePCA::MakeSubvector_(Vector& vec, 
			       ArrayList<index_t>& set, Vector *subvec) {

  if (set.size() == 0) {
    subvec->Destruct();
    subvec->Init(0);
    return;
  }
  else {

    double *subarray;
      
    subarray = (double*)malloc(set.size() * sizeof(double));
      
    for (index_t i = 0; i < set.size(); i++) {
      subarray[i] = vec.get(set[i]);
    }
      
    subvec->Destruct();
    subvec->Copy(subarray, set.size());
      
    return;
  }
}

void SparsePCA::MakeSubmatrix_(Matrix& mat, index_t *set, 
		    index_t subset_marker, Matrix *submatrix) {

  index_t set_length = mat.n_cols(), col_length = mat.n_rows(), subset_length;
  double *submatrix_array;

  subset_length = SubsetLength_(set, set_length, subset_marker);
  submatrix_array = 
    (double*)malloc(col_length * subset_length * sizeof(double));

  index_t j = 0;
  for (index_t i = 0; i < set_length; i++) {
    if (set[i] == subset_marker) {
      double *col_ptr;
      col_ptr = mat.GetColumnPtr(i);

      index_t start = j * col_length;
      for (index_t k = 0; k < col_length; k++) { 
	submatrix_array[k + start] = col_ptr[k];
      }

      ++j;
    }
  }

  DEBUG_ASSERT_MSG(j == subset_length, "MakeSubmatrix_(): X_A.n_cols() != |A|\n");

  submatrix->Destruct();
  submatrix->Copy(submatrix_array, col_length, subset_length);

  return;
}

void SparsePCA::MakeSubmatrix_(Matrix& mat, 
			       ArrayList<index_t>& set, Matrix *submatrix) {

  if (set.size() == 0) {
    submatrix->Destruct();
    submatrix->Init(0, 0);
    return;
  }
  else {    
    index_t set_length = set.size(), col_length = mat.n_rows();
    double *submatrix_array;
      
    submatrix_array = 
      (double*)malloc(col_length * set_length * sizeof(double));

    for (index_t i = 0; i < set_length; i++) {

      double *col_ptr;
      col_ptr = mat.GetColumnPtr(set[i]);
	  
      index_t start = i * col_length;
      for (index_t k = 0; k < col_length; k++) { 
	submatrix_array[k + start] = col_ptr[k];
      }
    }
      
    submatrix->Destruct();
    submatrix->Copy(submatrix_array, col_length, set_length);
      
    return;
  }
}
