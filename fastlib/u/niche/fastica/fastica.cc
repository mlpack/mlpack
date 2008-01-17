#include "fastlib/fastlib.h"
#include "lin_alg.h"

#define LOGCOSH 0
#define GAUSS 10
#define KURTOSIS 20
#define SKEW 30

#define SYMMETRIC 0
#define DEFLATION 1

// n indicates number of points
// d indicates number of dimensions (number of components or variables)


namespace {


  void SaveCorrectly(const char *filename, Matrix a) {
    Matrix a_transpose;
    la::TransposeInit(a, &a_transpose);
    data::Save(filename, a_transpose);
  }
  

  void RandVector(Vector &v) {

    index_t d = v.length();
    v.SetZero();
  
    for(index_t i = 0; i+1 < d; i+=2) {
      double a = drand48();
      double b = drand48();
      double first_term = sqrt(-2 * log(a));
      double second_term = 2 * M_PI * b;
      v[i] =   first_term * cos(second_term);
      v[i+1] = first_term * sin(second_term);
    }
  
    if((d % 2) == 1) {
      v[d - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
    }

    la::Scale(1/sqrt(la::Dot(v, v)), &v);

  }


  void Center(Matrix X, Matrix &X_centered) {
    Vector col_vector_sum;
    col_vector_sum.Init(X.n_rows());
    col_vector_sum.SetZero();
  
    index_t n = X.n_cols();
 
    for(index_t i = 0; i < n; i++) {
      Vector cur_col_vector;
      X.MakeColumnVector(i, &cur_col_vector);
      la::AddTo(cur_col_vector, &col_vector_sum);
    }

    la::Scale(1/(double) n, &col_vector_sum);

    X_centered.CopyValues(X);

    for(index_t i = 0; i < n; i++) {
      Vector cur_col_vector;
      X_centered.MakeColumnVector(i, &cur_col_vector);
      la::SubFrom(col_vector_sum, &cur_col_vector);
    }

  }


  void Whiten(Matrix X, Matrix &X_whitened, Matrix &whitening_matrix, Matrix &dewhitening_matrix) {
    Matrix X_cov, D, D_inv, E;
    Vector D_vector;

    Scale(1 / (double) (X.n_cols() - 1),
	  MulTransBInit(&X, &X, &X_cov));
    

    la::EigenvectorsInit(X_cov, &D_vector, &E);

    E.set(0, 1, -E.get(0, 1));
    E.set(1, 1, -E.get(1, 1));

    

    index_t d = D_vector.length();
    D.Init(d, d);
    D.SetZero();
    D_inv.Init(d, d);
    D_inv.SetZero();
    for(index_t i = 0; i < d; i++) {
      double sqrt_val = sqrt(D_vector[i]);
      D.set(i, i, sqrt_val);
      D_inv.set(i, i, 1 / sqrt_val);
    }

    la::MulTransBInit(D_inv, E, &whitening_matrix);
    la::MulInit(E, D, &dewhitening_matrix);
    la::MulInit(whitening_matrix, X, &X_whitened);
  }


  void Orthogonalize(Matrix W_old, Matrix &W) {

    Matrix W_squared, W_squared_inv_sqrt;
    
    la::MulTransAInit(W, W, &W_squared);
    
    Matrix D, E, E_times_D;
    Vector D_vector;
    
    la::EigenvectorsInit(W_squared, &D_vector, &E);
    D.InitDiagonal(D_vector);
    
    index_t d = D.n_rows();
    for(index_t i = 0; i < d; i++) {
      D.set(i, i, 1 / sqrt(D.get(i, i)));
    }
    
    la::MulInit(E, D, &E_times_D);
    la::MulTransBInit(E_times_D, E, &W_squared_inv_sqrt);
	
    // note that up until this point, W == W_old
    la::MulOverwrite(W_old, W_squared_inv_sqrt, &W);
  }


  index_t GetSamples(int max, double percentage, Vector *selected_indices) {   
    
    index_t num_selected = 0;
    Vector rand_nums;
    rand_nums.Init(max);
    for(index_t i = 0; i < max; i++) {
      double rand_num = drand48();
      rand_nums[i] = rand_num;
      if(rand_num <= percentage) {
	num_selected++;
      }
    }

    selected_indices -> Init(num_selected);
    
    int j = 0;
    for(index_t i = 0; i < max; i++) {
      if(rand_nums[i] <= percentage) {
	(*selected_indices)[j] = i;
	j++;
      }
    }

    return num_selected;
  }

  
  int FixedPointICA(Matrix X, Matrix whitening_matrix, Matrix dewhitening_matrix,
		   int approach, index_t num_of_IC, int g, int fine_tune,
		   double a1, double a2, double mu, bool stabilization,
		   double epsilon, index_t max_num_iterations,
		   index_t max_fine_tune, double sample_size,
		   Matrix* A, Matrix* W) {

    // ensure default values are passed into this function if the user doesn't care about certain parameters

    Matrix B;

    index_t d = X.n_rows();
    index_t n = X.n_cols();

    if(d < num_of_IC) {
      printf("error: must have num_of_IC <= Dimension!\n");
      return SUCCESS_FAIL;
    }

    W -> Init(d, num_of_IC);
    A -> Init(num_of_IC, d);

    if((sample_size > 1) || (sample_size < 0)) {
      sample_size = 1;
      printf("Setting sample_size to 1\n");
    }
    else if(sample_size < 1) {
      if((sample_size * n) < 1000) {
	sample_size = min(1000 / (double) n, (double) 1);
	printf("Warning: Setting sample_size to %0.3f (%d samples).\n", sample_size, (int) floor(sample_size * n));
      }
    }

    int g_orig;

    if((g == LOGCOSH) || (g == GAUSS) || (g == KURTOSIS) || (g == SKEW)) {
      g_orig = g;
    }
    else {
      printf("Illegal value [%d] for parameter: g\n", g);
      return SUCCESS_FAIL;
    }

    if(sample_size != 1) {
      g_orig += 2;
    }
    
    if(mu != 1) {
      g_orig += 1;
    }

    bool fine_tuning_enabled = true;
    int g_fine;

    if(fine_tune) {
      g_fine = g + 1;
    }
    else {
      if(mu != 1) {
	g_fine = g_orig;
      }
      else {
	g_fine = g_orig + 1;
      }

      fine_tuning_enabled = false;
    }

    bool stabilization_enabled;
    if(stabilization) {
      stabilization_enabled = true;
    }
    else {
      if(mu != 1) {
	stabilization_enabled = true;
      }
      else {
	stabilization_enabled = false;
      }
    }

    double mu_orig = mu;
    double mu_k = 0.01;
    index_t failure_limit = 5;
      
    int used_nonlinearity = g_orig;
    double stroke = 0;
    bool not_fine = true;
    bool taking_long = false;

    // currently we don't allow for guesses for the initial unmixing matrix B
    int initial_state_mode = 0;


    if(approach == SYMMETRIC) {
      printf("using Symmetric approach\n");

      if(initial_state_mode == 0) {
	//generate random B
	B.Init(d, num_of_IC);

	/*
	for(index_t i = 0; i < num_of_IC; i++) {
	  Vector b;
	  B.MakeColumnVector(i, &b);
	  RandVector(b);
	}
	*/
	B.SetZero();
	for(index_t i = 0; i < d; i++) {
	  B.set(i, i, 1);
	}
	  
      }
      
      

      Matrix B_old, B_old2;

      B_old.Init(d, num_of_IC);
      B_old2.Init(d, num_of_IC);

      B_old.SetZero();
      B_old2.SetZero();


      for(index_t round = 1; round <= (max_num_iterations + 1); round++) {
	if(round == (max_num_iterations + 1)) {
	  printf("No convergence after %d steps\n", max_num_iterations);
	
	
	  // orthogonalize B via: newB = B * (B' * B) ^ -.5;
	  Matrix temp;
	  temp.Copy(B);
	  Orthogonalize(temp, B);

	  MulTransAOverwrite(&B, &whitening_matrix, W);
	  MulOverwrite(&dewhitening_matrix, &B, A);
	  return SUCCESS_PASS;
	}
	
	{
	  Matrix temp;
	  temp.Copy(B);
	  Orthogonalize(temp, B);
	  B.PrintDebug("B");
	}

	Matrix B_delta_cov;
	MulTransAInit(&B, &B_old, &B_delta_cov);
	double min_abs_cos = DBL_MAX;
	for(index_t i = 0; i < d; i++) {
	  double current_cos = fabs(B_delta_cov.get(i, i));
	  if(current_cos < min_abs_cos) {
	    min_abs_cos = current_cos;
	  }
	}
      
	printf("min_abs_cos = %f\n", min_abs_cos);

	if(1 - min_abs_cos < epsilon) {
	  if(fine_tuning_enabled && not_fine) {
	    not_fine = false;
	    used_nonlinearity = g_fine;
	    mu = mu_k * mu_orig;
	    B_old.SetZero();
	    B_old2.SetZero();
	  }
	  else {
	    MulOverwrite(&dewhitening_matrix, &B, A);
	    MulTransAOverwrite(&B, &whitening_matrix, W);
	    return SUCCESS_PASS;
	  }
	}
	else if(stabilization_enabled) {

	  Matrix B_delta_cov2;
	  MulTransAInit(&B, &B_old2, &B_delta_cov2);
	  double min_abs_cos2 = DBL_MAX;
	  for(index_t i = 0; i < d; i++) {
	    double current_cos2 = fabs(B_delta_cov2.get(i, i));
	    if(current_cos2 < min_abs_cos2) {
	      min_abs_cos2 = current_cos2;
	    }
	  }

	  if((stroke == 0) && (1 - min_abs_cos2 < epsilon)) {
	    stroke = mu;
	    mu *= .5;
	    if((used_nonlinearity % 2) == 0) {
	      used_nonlinearity += 1;
	    }
	  }
	  else if(stroke > 0) {
	    mu = stroke;
	    stroke = 0;

	    if((mu == 1) && ((used_nonlinearity % 2) != 0)) {
	      used_nonlinearity -= 1;
	    }
	  }
	  else if((!taking_long) &&
		  (round > ((double) max_num_iterations / 2))) {
	    taking_long = true;
	    mu *= .5;
	    if((used_nonlinearity % 2) == 0) {
	      used_nonlinearity += 1;
	    }
	  }
	}

	B_old2.CopyValues(B_old);
	B_old.CopyValues(B);

	// show progress here, (the lack of code means no progress shown for now)



	// use Newton-Raphson to update B

	printf("used_nonlinearity = %d\n", used_nonlinearity);

	switch(used_nonlinearity) {
	  
	case LOGCOSH: {
	
	  Matrix hyp_tan, col_vector, sum, temp1, temp2;
	
	  MapOverwrite(&TanhArg,
		       a1,
		       MulTransAInit(&X, &B, &hyp_tan));

	
	  ColVector(d, a1, &col_vector);

	  Scale(1 / (double) n,
		AddTo(MulInit(&X, &hyp_tan, &temp1),
			     DotMultiplyOverwrite(MulInit(&col_vector,
							  MapOverwrite(&MinusArg,
								       n,
								       MatrixMapSum(&Square, 0, &hyp_tan, &sum)),
							  &temp2),
						  &B)));
	
	
	  break;
	}
	
	case LOGCOSH + 1: {
	  Matrix Y, hyp_tan, Beta, Beta_Diag, D, sum, temp1, temp2, temp3;
	
	  MulTransAInit(&X, &B, &Y);
	  MapInit(&TanhArg, a1, &Y, &hyp_tan);
	  DotMultiplySum(&Y, &hyp_tan, &Beta);
	  VectorToDiag(MapOverwrite(&Inv,
				    0,
				    AddTo(&Beta,
					  Scale(a1,
						MapOverwrite(&MinusArg,
							     n,
							     MatrixMapSum(&Square, 0, &hyp_tan, &sum))))),
		       &D);
	  
	  AddExpert(mu, 
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta, &Beta_Diag),
					    MulTransAInit(&Y, &hyp_tan, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);

	  break;	
	}
	
	case LOGCOSH + 2: {
	  /*
	    B = Xsub * hypTan / size(Xsub, 2) - ...
	    ones(size(B,1),1) * sum(1 - hypTan .^ 2) .* B / size(Xsub, 2) * a1;
	  */

	  Vector selected_indices;
	  index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);
	

	  Matrix hyp_tan, col_vector, sum, temp1, temp2;
	
	  MapOverwrite(&TanhArg,
		       a1,
		       MulTransAInit(&X_sub, &B, &hyp_tan));
	
	  ColVector(d, a1, &col_vector);
	
	  Scale(1 / (double) num_selected,
		AddTo(MulInit(&X_sub, &hyp_tan, &temp1),
			     DotMultiplyOverwrite(MulInit(&col_vector,
							  MapOverwrite(&MinusArg,
								       num_selected,
								       MatrixMapSum(&Square, 0, &hyp_tan, &sum)),
							  &temp2),
						  &B)));


	  break;
	}
	
	case LOGCOSH + 3: {
	  Vector selected_indices;
	  index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);
	

	  Matrix Y, hyp_tan, Beta, Beta_Diag, D, sum, temp1, temp2, temp3;
	
	  MulTransAInit(&X_sub, &B, &Y);
	  MapInit(&TanhArg, a1, &Y, &hyp_tan);
	  DotMultiplySum(&Y, &hyp_tan, &Beta);
	  VectorToDiag(MapOverwrite(&Inv,
				    0,
				    AddTo(&Beta,
					  Scale(a1,
						MapOverwrite(&MinusArg,
							     num_selected,
							     MatrixMapSum(&Square, 0, &hyp_tan, &sum))))),
		       &D);

	  AddExpert(mu, 
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta, &Beta_Diag),
					    MulTransAInit(&Y, &hyp_tan, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);

	
	  break;
	}
	
	case GAUSS: {
	  Matrix U, U_squared, ex, col_vector, sum, temp1, temp2;
	
	  MulTransAInit(&X, &B, &U);

	  MapInit(&Square, 0, &U, &U_squared);
	  MapInit(&ExpArg, -a2 / 2, &U_squared, &ex);
	  DotMultiplyOverwrite(&ex, &U);
	  //U is gauss
	  AddTo(DotMultiplyOverwrite(&ex,
				     Scale(-a2, &U_squared)),
		&ex);
	  //ex is dGauss

	  ColVector(d, a2, &col_vector);
	
	  Scale(1 / (double) n,
		SubOverwrite(MulInit(&X, &U, &temp1),
			     DotMultiplyOverwrite(&B,
						  MulInit(&col_vector,
							  Sum(&ex, &sum),
							  &temp2)),
			     &B));
	  
	  break;
	}
	
	case GAUSS + 1: {
	  Matrix Y, Y_squared_a2, ex, gauss, D, Beta, temp1, temp2, temp3;
	  Vector Beta_vector, sum_vector;
	
	  MulTransAInit(&X, &B, &Y);
	  MapInit(&SquareArg, a2, &Y, &Y_squared_a2);
	  MapInit(&ExpArg, -.5, &Y_squared_a2, &ex);
	  DotMultiplyInit(&Y, &ex, &gauss);
	
	  Beta_vector.Init(d);
	  double *Y_col_j;
	  double *gauss_col_j;
	  for(index_t j = 0; j < d; j++) {
	    Y_col_j = Y.GetColumnPtr(j);
	    gauss_col_j = gauss.GetColumnPtr(j);
	    double sum = 0;
	    for(index_t i = 0; i < n; i++) {
	      sum += Y_col_j[i] * gauss_col_j[i];
	    }
	    Beta_vector[j] = sum;
	  }

	
	  sum_vector.Init(d);
	  double *Y_squared_a2_col_j;
	  double *ex_col_j;
	  for(index_t j = 0; j < d; j++) {
	    Y_squared_a2_col_j = Y_squared_a2.GetColumnPtr(j);
	    ex_col_j = ex.GetColumnPtr(j);
	    double sum = 0;
	    for(index_t i = 0; i < n; i++) {
	      sum += (Y_squared_a2_col_j[i] - 1) * ex_col_j[i];
	    }
	    sum_vector[j] = sum;
	  }


	  //D = diag(1 ./ (Beta + sum((Y_squared_a2 - 1) .* ex)))
	  VectorToDiag(MapOverwrite(&Inv,
				    0,
				    AddTo(&Beta_vector, &sum_vector)),
		       &D);
	  
	  //B = B + myy * B * (Y' * gauss - diag(Beta)) * D;
	  AddExpert(mu,
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta_vector, &Beta),
					    MulTransAInit(&Y, &gauss, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);
	  
	  break;
	}

	case GAUSS + 2: {
	  Vector selected_indices;
	  index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	  Matrix U, U_squared, ex, col_vector, sum, temp1, temp2;
	
	  MulTransAInit(&X_sub, &B, &U);

	  MapInit(&Square, 0, &U, &U_squared);
	  MapInit(&ExpArg, -.5 * a2, &U_squared, &ex);
	  DotMultiplyOverwrite(&ex, &U);
	  //U is gauss
	  AddTo(DotMultiplyOverwrite(&ex,
				     Scale(-a2, &U_squared)),
		&ex);
	  //ex is dGauss

	
	
	  ColVector(d, a1, &col_vector);
	
	  Scale(1 / (double) num_selected,
		SubOverwrite(MulInit(&X_sub, &U, &temp1),
			DotMultiplyOverwrite(&B,
					     MulInit(&col_vector,
						     Sum(&ex, &sum),
						     &temp2)),
			&B));
	  
	  break;

	}
	case GAUSS + 3: {
	  Vector selected_indices;
	  index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	  Matrix Y, Y_squared_a2, ex, gauss, D, Beta, temp1, temp2, temp3;
	  Vector Beta_vector, sum_vector;
	
	  MulTransAInit(&X_sub, &B, &Y);
	  MapInit(&SquareArg, a2, &Y, &Y_squared_a2);
	  MapInit(&ExpArg, -.5, &Y_squared_a2, &ex);
	  DotMultiplyInit(&Y, &ex, &gauss);
	
	  Beta_vector.Init(d);
	  double *Y_col_j;
	  double *gauss_col_j;
	  for(index_t j = 0; j < d; j++) {
	    Y_col_j = Y.GetColumnPtr(j);
	    gauss_col_j = gauss.GetColumnPtr(j);
	    double sum = 0;
	    for(index_t i = 0; i < num_selected; i++) {
	      sum += Y_col_j[i] * gauss_col_j[i];
	    }
	    Beta_vector[j] = sum;
	  }

	
	  sum_vector.Init(d);
	  double *Y_squared_a2_col_j;
	  double *ex_col_j;
	  for(index_t j = 0; j < d; j++) {
	    Y_squared_a2_col_j = Y_squared_a2.GetColumnPtr(j);
	    ex_col_j = ex.GetColumnPtr(j);
	    double sum = 0;
	    for(index_t i = 0; i < num_selected; i++) {
	      sum += (Y_squared_a2_col_j[i] - 1) * ex_col_j[i];
	    }
	    sum_vector[j] = sum;
	  }

	  //D = diag(1 ./ (Beta + sum((Y_squared_a2 - 1) .* ex)))
	  VectorToDiag(MapOverwrite(&Inv,
			   0,
			   AddTo(&Beta_vector, &sum_vector)),
		       &D);
		
	  //B = B + myy * B * (Y' * gauss - diag(Beta)) * D;
	  AddExpert(mu,
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta_vector, &Beta),
					    MulTransAInit(&Y, &gauss, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);


	  break;
	}

	case KURTOSIS: {
	  Matrix temp1, temp2;

	  Scale(1 / (double) n,
		MulInit(&X,
			MapOverwrite(&pow,
				     3,
				     MulTransAInit(&X, &B, &temp1)),
			&temp2));
	
	  AddTo(&temp2,
		Scale(-3, &B));

	  break;
	}

	case KURTOSIS + 1: {
	  Matrix Y, G_pow_3, Beta, Beta_Diag, D_vector, D, temp1, temp2, temp3;

	  MulTransAInit(&X, &B, &Y);
	  MapInit(&pow, 3, &Y, &G_pow_3);

	  DotMultiplySum(&Y, &G_pow_3, &Beta);

	  VectorToDiag(MapOverwrite(&Inv,
				    0,
				    MapInit(&Plus,
					    -3 * n,
					    &Beta,
					    &D_vector)),
		       &D);
	  D.PrintDebug("this D");
	  AddExpert(mu, 
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta, &Beta_Diag),
					    MulTransAInit(&Y, &G_pow_3, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);
	  
	  break;
	}

	case KURTOSIS + 2: {
	  Vector selected_indices;
	  index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);
	
	
	  Matrix temp1, temp2;
	
	  Scale(1 / (double) num_selected,
		MulInit(&X_sub,
			MapOverwrite(&pow,
			    3,
			    MulTransAInit(&X_sub, &B, &temp1)),
			&temp2));
	
	  AddTo(&temp2,
		Scale(-3, &B));

	  break;
	}

	case KURTOSIS + 3: {
	  Vector selected_indices;
	  index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	  Matrix Y, G_pow_3, Beta, Beta_Diag, D_vector, D, temp1, temp2, temp3;
	
	  MulTransAInit(&X_sub, &B, &Y);
	  MapInit(&pow, 3, &Y, &G_pow_3);

	  DotMultiplySum(&Y, &G_pow_3, &Beta);

	  VectorToDiag(MapOverwrite(&Inv,
				    0,
				    MapInit(&Plus,
					    -3 * num_selected,
					    &Beta,
					    &D_vector)),
		       &D);

	  AddExpert(mu, 
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta, &Beta_Diag),
					    MulTransAInit(&Y, &G_pow_3, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);


	  break;
	}

	case SKEW: {
	  Matrix temp1;

	  Scale(1 / (double) n,
		MulOverwrite(&X,
			MapOverwrite(&Square,
			    0,
			    MulTransAInit(&X, &B, &temp1)),
			&B));
	
	  break;
	}

	case SKEW + 1: {
	  Matrix Y, G_skew, Beta, Beta_Diag, D_vector, D, temp1, temp2, temp3;
	
	  MulTransAInit(&X, &B, &Y);
	  MapInit(&Square, 0, &Y, &G_skew);
	  DotMultiplySum(&Y, &G_skew, &Beta);
	  VectorToDiag(MapInit(&Inv, 0, &Beta, &D_vector),
		       &D);
	
	  AddExpert(mu, 
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta, &Beta_Diag),
					    MulTransAInit(&Y, &G_skew, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);

	  break;
	}

	case SKEW + 2: {
	  Vector selected_indices;
	  index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	  Matrix temp1;
	
	  Scale(1 / (double) num_selected,
		MulOverwrite(&X_sub,
			MapOverwrite(&Square,
				     0,
				     MulTransAInit(&X_sub, &B, &temp1)),
			&B));
	
	  break;
	}
	
	case SKEW + 3: {
	  Vector selected_indices;
	  GetSamples(n, sample_size, &selected_indices);
	  Matrix X_sub;
	  MakeSubMatrixByColumns(selected_indices, X, &X_sub);
	
	
	  Matrix Y, G_skew, Beta, Beta_Diag, D_vector, D, temp1, temp2, temp3;
	
	  MulTransAInit(&X_sub, &B, &Y);
	  MapInit(&Square, 0, &Y, &G_skew);
	  DotMultiplySum(&Y, &G_skew, &Beta);
	  VectorToDiag(MapInit(&Inv, 0, &Beta, &D_vector),
		       &D);
	
	  AddExpert(mu, 
		    MulInit(MulInit(&B,
				    SubFrom(VectorToDiag(&Beta, &Beta_Diag),
					    MulTransAInit(&Y, &G_skew, &temp1)),
				    &temp2),
			    &D,
			    &temp3),
		    &B);


	  break;
	}
	  
	default: 
	  printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
		 used_nonlinearity);
	  exit(SUCCESS_FAIL);
	  
	}
      }
    }
    else if(approach == DEFLATION) {
      printf("using Deflation approach\n");
      B.Init(d, d);
      B.SetZero();

      index_t round = 0;

      index_t num_failures = 0;

      while(round < num_of_IC) {
	mu = mu_orig;
	used_nonlinearity = g_orig;
	stroke = 0;
	not_fine = true;
	taking_long = false;
	int end_fine_tuning = 0;
	
	Vector w;
	if(initial_state_mode == 0) {
	  w.Init(d);
	  RandVector(w);
	}

	for(index_t i = 0; i < round; i++) {
	  Vector b_i;
	  B.MakeColumnVector(i, &b_i);
	  la::AddExpert(-la::Dot(b_i, w), b_i, &w);
	}
	la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize

	Vector w_old, w_old2;
	w_old.Init(d);
	w_old.SetZero();
	w_old2.Init(d);
	w_old2.SetZero();

	index_t i = 1;
	index_t gabba = 1;
	while(i <= max_num_iterations + gabba) {

	  for(index_t j = 0; j < round; j++) {
	    Vector b_j;
	    B.MakeColumnVector(j, &b_j);
	    la::AddExpert(-la::Dot(b_j, w), b_j, &w);
	  }
	  la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize
	  
	  if(not_fine) {
	    if(i == (max_num_iterations + 1)) {
	      round++;
	      num_failures++;
	      if(num_failures > failure_limit) {
		printf("Too many failures to converge (%d). Giving up.\n", num_failures);
		return SUCCESS_FAIL;
	      }
	      break;
	    }
	  }
	  else {
	    if(i >= end_fine_tuning) {
	      w_old.Copy(w);
	    }
	  }

	  // check for convergence
	  bool converged = false;
	  Vector w_diff;
	  la::SubInit(w_old, w, &w_diff);
	  
	  if(la::Dot(w_diff, w_diff) < epsilon) {
	    converged = true;
	  }
	  else {
	    la::AddOverwrite(w_old, w, &w_diff);
	    
	    if(la::Dot(w_diff, w_diff) < epsilon) {
	      converged = true;
	    }
	  }

	  if(converged) {
	    if(fine_tuning_enabled & not_fine) {
	      not_fine = false;
	      gabba = max_fine_tune;
	      w_old.SetZero();
	      w_old2.SetZero();
	      used_nonlinearity = g_fine;
	      mu = mu_k * mu_orig;

	      end_fine_tuning = max_fine_tune + i;
	    }
	    else {
	      num_failures = 0;
	      Vector B_col_round, A_col_round, W_col_round;

	      B.MakeColumnVector(round, &B_col_round);
	      A -> MakeColumnVector(round, &A_col_round);
	      W -> MakeColumnVector(round, &W_col_round);

	      B_col_round.CopyValues(w);
	      la::MulOverwrite(dewhitening_matrix, w, &A_col_round);
	      la::MulOverwrite(w, whitening_matrix, &W_col_round);

	      break; // this line is intended to take us to the next IC
	    }
	  }
	  else if(stabilization_enabled) {
	    converged = false;
	    la::SubInit(w_old2, w, &w_diff);
	    
	    if(la::Dot(w_diff, w_diff) < epsilon) {
	      converged = true;
	    }
	    else {
	      la::AddOverwrite(w_old2, w, &w_diff);
	      
	      if(la::Dot(w_diff, w_diff) < epsilon) {
		converged = true;
	      }
	    }
	    
	    if((stroke == 0) && converged) {
	      stroke = mu;
	      mu *= .5;
	      if((used_nonlinearity % 2) == 0) {
		used_nonlinearity++;
	      }
	    }
	    else if(stroke != 0) {
	      mu = stroke;
	      stroke = 0;
	      if((mu == 1) && ((used_nonlinearity % 2) != 0)) {
		used_nonlinearity--;
	      }
	    }
	    else if(not_fine && (!taking_long) && (i > ((double) max_num_iterations / 2))) {
	      taking_long = true;
	      mu *= .5;
	      if((used_nonlinearity % 2) == 0) {
		used_nonlinearity++;
	      }
	    }
	  }

	  w_old2.CopyValues(w_old);
	  w_old.CopyValues(w);
	  
	  printf("used_nonlinearity = %d\n", used_nonlinearity);
	
	  switch(used_nonlinearity) {
	  case LOGCOSH: {
	    Vector hyp_tan, temp1;
	    
	    MapOverwrite(&TanhArg,
			 a1,
			 MulInit(&w, &X, &hyp_tan));

	    
	    Scale(1 / (double) n,
		  AddTo(MulInit(&X, &hyp_tan, &temp1),
			Scale(a1 * (VectorMapSum(&Square, 0, &hyp_tan) - n),
			      &w)));

	    break;
	  }


	  case LOGCOSH + 1: {
	    Vector hyp_tan, X_hyp_tan, Beta_w, temp1;
	    
	    MapOverwrite(&TanhArg,
			 a1,
			 MulInit(&w, &X, &hyp_tan));

	    MulInit(&X, &hyp_tan, &X_hyp_tan);
	    double Beta = la::Dot(X_hyp_tan, w);

	    AddExpert(mu,
		      Scale(1 / (a1 * (VectorMapSum(&Square, 0, &hyp_tan) - n) + Beta),
			    SubInit(&X_hyp_tan,
				    ScaleInit(Beta, &w, &Beta_w),
				    &temp1)),
		      &w);

	    break;
	  }


	  case LOGCOSH + 2: {
	    Vector selected_indices;
	    index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);
	    
	    
	    Vector hyp_tan, temp1;
	    
	    MapOverwrite(&TanhArg,
			 a1,
			 MulInit(&w, &X_sub, &hyp_tan));
	    
	    Scale(1 / (double) num_selected,
		  AddTo(MulInit(&X_sub, &hyp_tan, &temp1),
			Scale(a1 * (VectorMapSum(&Square, 0, &hyp_tan) - n),
			      &w)));
	    
	    break;
	  }
	  case LOGCOSH + 3: {
	    Vector selected_indices;
	    index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	    Vector hyp_tan, X_sub_hyp_tan, Beta_w, temp1;
	    
	    MapOverwrite(&TanhArg,
			 a1,
			 MulInit(&w, &X_sub, &hyp_tan));

	    MulInit(&X_sub, &hyp_tan, &X_sub_hyp_tan);
	    double Beta = la::Dot(X_sub_hyp_tan, w);

	    AddExpert(mu,
		      Scale(1 / (a1 * (VectorMapSum(&Square, 0, &hyp_tan) - num_selected) + Beta),
			    SubInit(&X_sub_hyp_tan,
				    ScaleInit(Beta, &w, &Beta_w),
				    &temp1)),
		      &w);

	    break;
	  }
	    
	    
	  case GAUSS: {
	    Vector u, u_squared, ex, temp1;

	    MulInit(&w, &X, &u);
	    MapInit(&Square, 0, &u, &u_squared);
	    MapInit(&ExpArg, -.5 * a2, &u_squared, &ex);
	    DotMultiplyOverwrite(&ex, &u);
	    //u is gauss
	    AddTo(DotMultiplyOverwrite(&ex,
				       Scale(-a2, &u_squared)),
		  &ex);
	    //ex is dGauss
	    Scale(1 / (double) n,
		  AddTo(MulInit(&X, &u, &temp1),
			Scale(-1 * Sum(&ex), &w)));

	    break;
	  }
	    

	  case GAUSS + 1: {
	    Vector u, u_squared, ex, X_gauss, Beta_w, temp1;

	    MulInit(&w, &X, &u);
	    MapInit(&Square, 0, &u, &u_squared);
	    MapInit(&ExpArg, -.5 * a2, &u_squared, &ex);
	    DotMultiplyOverwrite(&ex, &u);
	    //u is gauss
	    AddTo(DotMultiplyOverwrite(&ex,
				       Scale(-a2, &u_squared)),
		  &ex);
	    //ex is dGauss

	    MulInit(&X, &u, &X_gauss);
	    double Beta = la::Dot(X_gauss, w);

	    AddExpert(mu,
		      Scale(1 / (Beta - Sum(&ex)),
			    SubInit(&X_gauss,
				    ScaleInit(Beta, &w, &Beta_w),
				    &temp1)),
		      &w);
	    
	    break;
	  }

	    
	  case GAUSS + 2: {
	    Vector selected_indices;
	    index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	    Vector u, u_squared, ex, temp1;

	    MulInit(&w, &X_sub, &u);
	    MapInit(&Square, 0, &u, &u_squared);
	    MapInit(&ExpArg, -.5 * a2, &u_squared, &ex);
	    DotMultiplyOverwrite(&ex, &u);
	    //u is gauss
	    AddTo(DotMultiplyOverwrite(&ex,
				       Scale(-a2, &u_squared)),
		  &ex);
	    //ex is dGauss
	    Scale(1 / (double) num_selected,
		  AddTo(MulInit(&X_sub, &u, &temp1),
			Scale(-1 * Sum(&ex), &w)));


	    
	    
	    break;
	  }
	  

	  case GAUSS + 3: {
	    Vector selected_indices;
	    GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	    Vector u, u_squared, ex, X_sub_gauss, Beta_w, temp1;

	    MulInit(&w, &X_sub, &u);
	    MapInit(&Square, 0, &u, &u_squared);
	    MapInit(&ExpArg, -.5 * a2, &u_squared, &ex);
	    DotMultiplyOverwrite(&ex, &u);
	    //u is gauss
	    AddTo(DotMultiplyOverwrite(&ex,
				       Scale(-a2, &u_squared)),
		  &ex);
	    //ex is dGauss

	    MulInit(&X_sub, &u, &X_sub_gauss);
	    double Beta = la::Dot(X_sub_gauss, w);

	    AddExpert(mu,
		      Scale(1 / (Beta - Sum(&ex)),
			    SubInit(&X_sub_gauss,
				    ScaleInit(Beta, &w, &Beta_w),
				    &temp1)),
		      &w);

	    break;
	  }

	    
	  case KURTOSIS: {
	    Vector temp1, temp2;

	    Scale(1 / (double) n,
		  MulInit(&X,
			  MapOverwrite(&pow,
				       3,
				       MulInit(&w, &X, &temp1)),
			  &temp2));
	    
	    AddTo(&temp2,
		  Scale(-3, &w));

	    break;
	  }


	  case KURTOSIS + 1: {
	    Vector EXG_pow_3, Beta_w, temp1;
	    
	    Scale(1 / (double) n,
		  MulInit(&X,
			  MapOverwrite(&pow,
				       3,
				       MulInit(&w, &X, &temp1)),
			  &EXG_pow_3));

	    double Beta = la::Dot(w, EXG_pow_3);
	    
	    AddExpert(mu / (Beta - 3),
		      SubFrom(ScaleInit(Beta, &w, &Beta_w),
			      &EXG_pow_3),
		      &w);
	    
	    break;
	  }


	  case KURTOSIS + 2: {
	    Vector selected_indices;
	    index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	    Vector temp1, temp2;

	    Scale(1 / (double) num_selected,
		  MulInit(&X_sub,
			  MapOverwrite(&pow,
				       3,
				       MulInit(&w, &X_sub, &temp1)),
			  &temp2));
	    
	    AddTo(&temp2,
		  Scale(-3, &w));

	    break;
	  }
	    

	  case KURTOSIS + 3: {
	    Vector selected_indices;
	    index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);

	    Vector EXG_pow_3, Beta_w, temp1;
	    
	    Scale(1 / (double) num_selected,
		  MulInit(&X_sub,
			  MapOverwrite(&pow,
				       3,
				       MulInit(&w, &X_sub, &temp1)),
			  &EXG_pow_3));

	    double Beta = la::Dot(w, EXG_pow_3);
	    
	    AddExpert(mu / (Beta - 3),
		      SubFrom(ScaleInit(Beta, &w, &Beta_w),
			      &EXG_pow_3),
		      &w);
	    
	    break;
	  }


	  case SKEW: {
	    Vector temp1;
	    
	    Scale(1 / (double) n,
		  MulInit(&X,
			  MapOverwrite(&Square,
				       0,
				       MulInit(&w, &X, &temp1)),
			  &w));

	    break;
	  }


	  case SKEW + 1: {
	    Vector EXG_skew, Beta_w, temp1;
	    
	    Scale(1 / (double) n,
		  MulInit(&X,
			  MapOverwrite(&Square,
				       0,
				       MulInit(&w, &X, &temp1)),
			  &EXG_skew));

	    double Beta = la::Dot(w, EXG_skew);
	    
	    AddExpert(mu / Beta,
		      SubFrom(ScaleInit(Beta, &w, &Beta_w),
			      &EXG_skew),
		      &w);
	    
	    break;
	  }


	  case SKEW + 2: {
	    Vector selected_indices;
	    index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);


	    Vector temp1;
	    
	    Scale(1 / (double) num_selected,
		  MulInit(&X_sub,
			  MapOverwrite(&Square,
				       0,
				       MulInit(&w, &X_sub, &temp1)),
			  &w));

	    break;
	  }


	  case SKEW + 3: {
	    Vector selected_indices;
	    index_t num_selected = GetSamples(n, sample_size, &selected_indices);
	    Matrix X_sub;
	    MakeSubMatrixByColumns(selected_indices, X, &X_sub);

	    Vector EXG_skew, Beta_w, temp1;
	    
	    Scale(1 / (double) num_selected,
		  MulInit(&X_sub,
			  MapOverwrite(&Square,
				       0,
				       MulInit(&w, &X_sub, &temp1)),
			  &EXG_skew));

	    double Beta = la::Dot(w, EXG_skew);
	    
	    AddExpert(mu / Beta,
		      SubFrom(ScaleInit(Beta, &w, &Beta_w),
			      &EXG_skew),
		      &w);

	    break;
	  }

	    
	  default: 
	    printf("ERROR: invalid contrast function: used_nonlinearity = %d\n",
		   used_nonlinearity);
	    exit(SUCCESS_FAIL);
	    
	  }
	
	  la::Scale(1/sqrt(la::Dot(w, w)), &w); // normalize
	  i++;
	}
	round++;
      }
    }
    else {
      printf("no other approaches implemented\n!");
    }
    
    MulTransAOverwrite(&B, &whitening_matrix, W);

    return SUCCESS_FAIL;
  
  }
}


int FastICAMain(datanode *module) {

  const char *data = fx_param_str_req(NULL, "data");
  
  Matrix X, X_centered, X_whitened, whitening_matrix, dewhitening_matrix, A, W;
  data::Load(data, &X);
  
  index_t d = X.n_rows(); // number of dimensions
  index_t n = X.n_cols(); // number of points


  const char *string_approach =
    fx_param_str(NULL, "approach", "deflation");
  int approach;
  if(strcasecmp(string_approach, "deflation") == 0) {
    approach = DEFLATION;
  }
  else if(strcasecmp(string_approach, "symmetric")) {
    approach = SYMMETRIC;
  }
  else {
    printf("Error: approach must be 'deflation' or 'symmetric'\n");
    return SUCCESS_FAIL;
  }
    
  const char *string_nonlinearity =
    fx_param_str(NULL, "nonlinearity", "logcosh");
  int nonlinearity;
  if(strcasecmp(string_nonlinearity, "logcosh") == 0) {
    nonlinearity = LOGCOSH;
  }
  else if(strcasecmp(string_nonlinearity, "gauss") == 0) {
    nonlinearity = GAUSS;
  }
  else if(strcasecmp(string_nonlinearity, "kurtosis") == 0) {
    nonlinearity = KURTOSIS;
  }
  else if(strcasecmp(string_nonlinearity, "skew") == 0) {
    nonlinearity = SKEW;
  }
  else {
    printf("Error: nonlinearity must be 'logcosh', 'gauss', 'kurtosis', or 'skew'\n");
    return SUCCESS_FAIL;
  }

  //const index_t first_eig = fx_param_int(NULL, "first_eig", 1);
  // for now, the last eig must be d, and num_of IC must be d, until I have time to incorporate PCA into this code
  //const index_t last_eig = fx_param_int(NULL, "last_eig", d);
  index_t num_of_IC = d; //fx_param_int(NULL, "num_of_IC", d);

  bool fine_tune = fx_param_bool(NULL, "fine_tune", 0);
  double a1 = fx_param_double(NULL, "a1", 1);
  double a2 = fx_param_double(NULL, "a2", 1);
  double mu = fx_param_double(NULL, "mu", 1);
  bool stabilization = fx_param_bool(NULL, "stabilization", 0);
  double epsilon = fx_param_double(NULL, "epsilon", 0.0001);
  
  int int_max_num_iterations = fx_param_int(NULL, "max_num_iterations", 1000);
  if(int_max_num_iterations < 0) {
    printf("Error: max_num_iterations = %d must be >= 0\n",
	   int_max_num_iterations);
    return SUCCESS_FAIL;
  }
  index_t max_num_iterations = (index_t) int_max_num_iterations;

  int int_max_fine_tune = fx_param_int(NULL, "max_fine_tune", 5);
  if(int_max_fine_tune < 0) {
    printf("Error: max_fine_tune = %d must be >= 0\n",
	   int_max_fine_tune);
    return SUCCESS_FAIL;
  }
  index_t max_fine_tune = (index_t) int_max_fine_tune;

  double sample_size = fx_param_double(NULL, "sample_size", 1);
  if((sample_size < 0) || (sample_size > 1)) {
    printf("Error: sample_size = %f must be an element in [0,1]\n",
	   sample_size);
    return SUCCESS_FAIL;
  }


  printf("X: %d x %d\n", d, n);

  fx_timer_start(NULL, "FastICA");

  X_centered.Init(d, n);
  Center(X, X_centered);

 

  Whiten(X_centered, X_whitened, whitening_matrix, dewhitening_matrix);

  
  FixedPointICA(X_whitened, whitening_matrix, dewhitening_matrix, approach, num_of_IC, nonlinearity, fine_tune, a1, a2, mu, stabilization, epsilon, max_num_iterations, max_fine_tune, sample_size, &A, &W);

  W.PrintDebug("W");

  Matrix Y;
  la::MulInit(W, X, &Y);

  fx_timer_stop(NULL, "FastICA");
  

  SaveCorrectly("unmixing_matrix.dat", W);
  SaveCorrectly("indep_comps.dat", Y);

  return SUCCESS_PASS;
}


int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  int ret_val = FastICAMain(fx_root);

  fx_done();

  return ret_val;
}

