#include "fastlib/fastlib.h"

/**
 * Save the matrix to a file so that rows in the matrix correspond to rows in
 * the file: This just means call data::Save() on the transpose of the matrix
 */ 
void SaveCorrectly(const char *filename, Matrix a) {
  Matrix a_transpose;
  la::TransposeInit(a, &a_transpose);
  data::Save(filename, a_transpose);
}

/**
 * Returns \f$ x^{arg} \f$
 */
double ExpArg(double x, double arg) {
  return exp(x * arg);
}

/**
 * Returns \f$ 1 / x \f$
 */
double Inv(double x, double arg) {
  return 1 / x;
}

/**
 * Returns \f$ x^2 \f$
 */
double Square(double x, double arg) {
  return x * x;
}

/**
 * Returns \f$ arg (x^2) \f$
 */
double SquareArg(double x, double arg) {
  return arg * x * x;
}

/**
 * Returns \f$ \tanh(arg x) \f$
 */
double TanhArg(double x, double arg) {
  return tanh(arg * x);
}

/**
 * Returns \f$ arg x \f$
 */
double Times(double x, double arg) {
  return arg * x;
}

/**
 * Returns \f$ x + arg \f$
 */
double Plus(double x, double arg) {
  return x + arg;
}

/**
 * Returns \f$ x - arg \f$
 */
double MinusArg(double x, double arg) {
  return x - arg;
}

/**
 * Returns \f$ arg - x \f$
 */
double ArgMinus(double x, double arg) {
  return arg - x;
}

/**
 * Inits a n by n diagonal matrix and sets the diagonal entries to value
 */
Matrix* DiagMatrixInit(index_t n, double value, Matrix *diag_matrix) {
  diag_matrix -> Init(n, n);
  diag_matrix -> SetZero();
  for(index_t i = 0; i < n; i++) {
    diag_matrix -> set(i, i, value);
  }

  return diag_matrix;
}

/**
 * Inits a n-dimensional column vector and sets all entries to value
 */
Matrix* ColVector(index_t n, double value, Matrix *col_vector) {
  col_vector -> Init(n, 1);
  col_vector -> SetAll(value);
  
  return col_vector;
}

/**
 * Sums over the rows of a M by N matrix and Inits a 1 by N matrix storing
 * the sum
 * (\f$ sum\_vector \gets \sum_i \vec{A_{row i}} \f$)
 */
Matrix* Sum(const Matrix* const A, Matrix *sum_vector) {
  index_t n_rows = A -> n_rows();
  index_t n_cols = A -> n_cols();

  sum_vector -> Init(1, n_cols);

  const double *A_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    A_col_j = A -> GetColumnPtr(j);
    double sum = 0;
    for(index_t i = 0; i < n_rows; i++) {
      sum += A_col_j[i];
    }
    (*sum_vector).set(0, j, sum);
  }

  return sum_vector;
}

/**
 * Returns the sum of the components of vector v
 * (returns \sum v_i \f$)
 */
double Sum(Vector *v) {
  index_t n = v -> length();

  double sum = 0;
  for(index_t i = 0; i < n; i++) {
    sum += (*v)[i];
  }

  return sum;
}

/**
 * Applies function with argument arg to a M by N Matrix and Inits a 1 by N
 * matrix to the sum over the transformed rows
 * (\f[
 * \tilde{A}_{i,j} \gets function(A_{i,j}, arg),
 * sum\_vector \gets \sum_i \vec{\tilde{A}_{row i}}
 * \f]
 */
Matrix* MatrixMapSum(double (*function)(double,double),
		     double arg,
		     const Matrix* const A,
		     Matrix *sum_vector) {
  index_t n_rows = A -> n_rows();
  index_t n_cols = A -> n_cols();

  sum_vector -> Init(1, n_cols);

  const double *A_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    A_col_j = A -> GetColumnPtr(j);
    double sum = 0;
    for(index_t i = 0; i < n_rows; i++) {
      sum += function(A_col_j[i], arg);
    }
    (*sum_vector).set(0, j, sum);
  }

  return sum_vector;
}

/**
 * Applies function with argument arg to vector v and returns the sum of the
 * transformed components
 * (sum \gets \f$ \sum_i function(v_i, arg) \f$)
 */
double VectorMapSum(double (*function)(double,double),
		    double arg,
		    const Vector* const v) {
  index_t n = v -> length();

  double sum = 0;
  for(index_t i = 0; i < n; i++) {
    sum += function((*v)[i], arg);
  }
  
  return sum;
}

/**
 * Multiplies A and B entry-wise and Inits a matrix to the result
 * @pre{ A and B are of equal dimensions }
 * (\f$ C \gets A .* B \f$)
 */
Matrix* DotMultiplyInit(const Matrix* const A, const Matrix* const B,
			Matrix* C) {
  index_t n_rows = A -> n_rows();
  index_t n_cols = A -> n_cols();

  C -> Init(n_rows, n_cols);

  const double *A_col_j;
  const double *B_col_j;
  double *C_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    A_col_j = A -> GetColumnPtr(j);
    B_col_j = B -> GetColumnPtr(j);
    C_col_j = C -> GetColumnPtr(j);
    for(index_t i = 0; i < n_rows; i++) {
      C_col_j[i] = A_col_j[i] * B_col_j[i];
    }
  }

  return C;
}

/**
 * Multiplies A and B entry-wise and overwrites matrix B with the result
 * @pre{ A and B are of equal dimensions }
 * (\f$ B \gets A \bullet B \f$)
 */
Matrix* DotMultiplyOverwrite(const Matrix* const A, Matrix* const B) {
  index_t n_rows = A -> n_rows();
  index_t n_cols = A -> n_cols();

  const double *A_col_j;
  double *B_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    A_col_j = A -> GetColumnPtr(j);
    B_col_j = B -> GetColumnPtr(j);
    for(index_t i = 0; i < n_rows; i++) {
      B_col_j[i] *= A_col_j[i];
    }
  }

  return B;
}

/**
 * Multiplies u and v entry-wise and Inits a vector to the result
 * @pre{ u and v are of equal dimensions }
 * (\f$ \vec{w} \gets \vec{u} \bullet \vec{v} \f$)
 */
Vector* DotMultiplyInit(const Vector* const u, const Vector* const v,
			Vector* w) {
  index_t n = u -> length();
  
  (*w).Init(n);

  for(index_t i = 0; i < n; i++) {
    (*w)[i] = (*u)[i] * (*v)[i];
  }

  return w;
}

/**
 * Multiplies u and v entry-wise and overwrites vector v with the result
 * @pre{ u and v are of equal dimensions }
 * (\f$ \vec{v} \gets \vec{u} \bullet \vec{v} \f$)
 */
Vector* DotMultiplyOverwrite(const Vector* const u, Vector* const v) {
  index_t n = u -> length();
  
  for(index_t i = 0; i < n; i++) {
    (*v)[i] *= (*u)[i];
  }

  return v;
}

/**
 * Multiplies A and B (N by M matrices) entry-wise and Inits a 1 by N matrix
 * to the sum over the transformed rows of the result
 * (\f$ C = A \bullet B, sum\_vector = \sum_i \vec{C_{row i}} \f$)
 */
Matrix* DotMultiplySum(const Matrix* const A, const Matrix* const B,
		       Matrix* sum_vector) {
  index_t n_rows = A -> n_rows();
  index_t n_cols = A -> n_cols();

  sum_vector -> Init(1, n_cols);

  const double *A_col_j;
  const double *B_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    A_col_j = A -> GetColumnPtr(j);
    B_col_j = B -> GetColumnPtr(j);
    double sum = 0;
    for(index_t i = 0; i < n_rows; i++) {
      sum += A_col_j[i] * B_col_j[i];
    }
    (*sum_vector).set(0, j, sum);
  }

  return sum_vector;
}
  
/**
 * Inits the diagonal entries of a N by N diagonal matrix to the
 * entries of a 1 by N matrix
 */
Matrix* VectorToDiag(const Matrix* const diag_vector, Matrix* diag_matrix) {

  index_t n = diag_vector -> n_cols();
  diag_matrix -> Init(n, n);
  diag_matrix -> SetZero();

  for(index_t i = 0; i < n; i++) {
    diag_matrix -> set(i, i, diag_vector -> get(0, i));
  }

  return diag_matrix;
}

/**
 * Inits the diagonal entries of a N by N diagonal matrix to the
 * entries of a N-dimensional vector
 */
Matrix* VectorToDiag(const Vector* const diag_vector, Matrix* diag_matrix) {

  diag_matrix -> InitDiagonal(*diag_vector);

  return diag_matrix;
}

/**
 * Inits the components of a vector to the diagonal entries of a square matrix
 * @pre{ diag_matrix is square }
 */
Vector* DiagToVector(const Matrix* const diag_matrix, Vector* diag_vector) {

  index_t n = diag_matrix -> n_rows();

  diag_vector -> Init(n);

  for(index_t i = 0; i < n; i++) {
    (*diag_vector)[i] = diag_matrix -> get(i, i);
  }

  return diag_vector;
}

/**
 * Sets A to alpha * A
 * (\f$ A \gets \alpha * A \f$)
 */
Matrix* Scale(double alpha, Matrix *A) {
  la::Scale(alpha, A);
  return A;
}

/**
 * Sets v to alpha * v
 * (\f$ \vec{v} \gets \alpha \vec{v} \f$)
 */
Vector* Scale(double alpha, Vector* v) {
  la::Scale(alpha, v);
  return v;
}

/**
 * Inits a matrix to alpha * A
 * (\f$ B \gets \alpha A \f$)
 */
Matrix* ScaleInit(double alpha, const Matrix* const A, Matrix* B) {
  la::ScaleInit(alpha, *A, B);
  return B;
}

/**
 * Inits a vector to alpha * u
 * (\f$ \vec{v} \gets \alpha \vec{u} \f$)
 */
Vector* ScaleInit(double alpha, const Vector* const u, Vector* v) {
  la::ScaleInit(alpha, *u, v);
  return v;
}

/**
 * Inits a matrix to A * B
 * (\f$ C \gets A B \f$)
 */
Matrix* MulInit(const Matrix* const A, const Matrix* const B,
		Matrix* const C) {
  la::MulInit(*A, *B, C);
  return C;
}

/**
 * Inits a vector to A * u
 * (\f$ \vec{v} \gets A \vec{u} \f$)
 */
Vector* MulInit(const Matrix* const A, const Vector* const u,
		Vector* const v) {
  la::MulInit(*A, *u, v);
  return v;
}

/**
 * Inits a vector to u A or A' u
 * (\f$ \vec{u} \gets \vec{u} A \f$ or \f$ \vec{u} \gets A^T \vec{u} \f$)
 */
Vector* MulInit(const Vector* const u, const Matrix* const A,
		Vector* const v) {
  la::MulInit(*u, *A, v);
  return v;
}

/**
 * Overwrites a matrix with A * B
 * (\f$ C \gets A B \f$)
 */
Matrix* MulOverwrite(const Matrix* const A, const Matrix* const B,
		     Matrix* const C) {
  la::MulOverwrite(*A, *B, C);
  return C;
}

/**
 * Inits a matrix to A' * B
 * (\f$ C \gets A^T B \f$)
 */
Matrix* MulTransAInit(const Matrix* const A, const Matrix* const B,
		      Matrix* C) {
  la::MulTransAInit(*A, *B, C);
  return C;
}

/**
 * Overwrites a matrix with A' * B
 * (\f$ C \gets A^T B \f$)
 */
Matrix* MulTransAOverwrite(const Matrix* const A, const Matrix* const B,
			   Matrix* const C) {
  la::MulTransAOverwrite(*A, *B, C);
  return C;
}

/**
 * Inits a matrix to A * B'
 * (\f$ C \gets A B^T \f$)
 */
Matrix* MulTransBInit(const Matrix* const A, const Matrix* const B,
		      Matrix* C) {
  la::MulTransBInit(*A, *B, C);
  return C;
}

/** 
 * Overwrites a matrix with A * B'
 * (\f$ C \gets A B^T \f$)
 */
Matrix* MulTransBOverwrite(const Matrix* const A, Matrix* const B,
			   Matrix* const C) {
  la::MulTransBOverwrite(*A, *B, C);
  return C;
}

/**
 * Inits a matrix to A - B
 * (\f$ C \gets A - B \f$)
 */
Matrix* SubInit(const Matrix* const A, const Matrix* const B, Matrix* C) {
  la::SubInit(*B, *A, C);
  return C;
}

/**
 * Inits a vector to u - v
 * (\f$ \vec{w} \gets \vec{u} - \vec{v} \f$)
 */
Vector* SubInit(const Vector* const u, const Vector* const v, Vector* w) {
  la::SubInit(*v, *u, w);
  return w;
}

/**
 * Overwrites a matrix with A - B
 * (\f$ C \gets A - B \f$)
 */
Matrix* SubOverwrite(const Matrix* const A, const Matrix* const B,
		     Matrix* const C) {
  la::SubOverwrite(*B, *A, C);
  return C;
}

/**
 * Sets matrix B to B - A
 * (\f$ B \gets B - A \f$)
 */
Matrix* SubFrom(const Matrix* const A, Matrix* const B) {
  la::SubFrom(*A, B);
  return B;
}

/**
 * Sets vector v to v - u
 * (\f$ \vec{v} \gets \vec{v} - vec{u} \f$)
 */
Vector* SubFrom(const Vector* const u, Vector* const v) {
  la::SubFrom(*u, v);
  return v;
}

/**
 * Sets matrix B to B + A
 * (\f$ B \gets B + A \f$)
 */
Matrix* AddTo(const Matrix* const A, Matrix* const B) {
  la::AddTo(*A, B);
  return B;
}

/**
 * Sets vector v to v + u
 * (\f$ \vec{v} \gets \vec{v} + \vec{u} \f$)
 */
Vector* AddTo(const Vector* const u, Vector* const v) {
  la::AddTo(*u, v);
  return v;
}

/**
 * Sets matrix B to B + alpha * A
 * (\f$ B \gets B + \alpha A \f$)
 */
Matrix* AddExpert(double alpha, const Matrix* const A, Matrix* const B) {
  la::AddExpert(alpha, *A, B);

  return B;
}

/**
 * Sets vector v to v + alpha * u
 * (\f$ \vec{v} \gets \vec{v} + \alpha \vec{u} \f$)
 */
Vector* AddExpert(double alpha, const Vector* const u, Vector* const v) {
  la::AddExpert(alpha, *u, v);

  return v;
}

/**
 * Applies function with argument arg to matrix A and
 * overwrites A with the result
 * (\f$ A_{i,j} \gets function(A_{i,j}, arg) \f$)
 */
Matrix* MapOverwrite(double (*function)(double,double),
		     double arg,
		     Matrix *A) {
  index_t n_rows = A -> n_rows();
  index_t n_cols = A -> n_cols();

  double *A_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    A_col_j = A -> GetColumnPtr(j);
    for(index_t i = 0; i < n_rows; i++) {
      A_col_j[i] = function(A_col_j[i], arg);
    }
  }

  return A;
}

/**
 * Applies function with argument arg to vector v and
 * overwrites v with the result
 * (\f$ v_i \gets function(v_i, arg) \f$)
 */
Vector* MapOverwrite(double (*function)(double,double),
		     double arg,
		     Vector* const v) {
  index_t n = v -> length();

  for(index_t i = 0; i < n; i++) {
    (*v)[i] = function((*v)[i], arg);
  }

  return v;
}

/**
 * Inits a matrix to the result of applying function with argument arg to
 * a matrix
 * (\f$ B_{i,j} \gets function(A_{i,j}, arg) \f$)
 */
Matrix* MapInit(double (*function)(double,double),
		double arg,
		const Matrix* const A,
		Matrix *B) {
  index_t n_rows = A -> n_rows();
  index_t n_cols = A -> n_cols();

  B -> Init(n_rows, n_cols);

  const double *A_col_j;
  double *B_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    A_col_j = A -> GetColumnPtr(j);
    B_col_j = B -> GetColumnPtr(j);
    for(index_t i = 0; i < n_rows; i++) {
      B_col_j[i] = function(A_col_j[i], arg);
    }
  }

  return B;
}

/**
 * Inits a vector to the result of applying function with argument arg to
 * a vector
 * (\f$ v_i \gets function(u_i, arg) \f$)
 */
Vector* MapInit(double (*function)(double,double),
		double arg,
		const Vector* const u,
		Vector *v) {
  index_t n = u -> length();
  v -> Init(n);
  
  for(index_t i = 0; i < n; i++) {
    (*v)[i] = function((*u)[i], arg);
  }

  return v;
}

/**
 * Inits a matrix to uniform random entries in [0,1]
 */
void RandMatrix(index_t n_rows, index_t n_cols, Matrix *A) {
  A -> Init(n_rows, n_cols);

  for(index_t j = 0; j < n_cols; j++) {
    for(index_t i = 0; i < n_rows; i++) {
      A -> set(i, j, drand48());
    }
  }
}

/**
 * Inits a matrix to the columns of A specified in column_indices
 */
void MakeSubMatrixByColumns(Vector column_indices, Matrix A, Matrix *A_sub) {
  
  index_t num_selected = column_indices.length();

  A_sub -> Init(A.n_rows(), num_selected);
  
  for(index_t i = 0; i < num_selected; i++) {
    index_t index = (index_t) column_indices[i];
    Vector A_col_index_i, A_sub_col_i;
    A.MakeColumnVector(index, &A_col_index_i);
    A_sub -> MakeColumnVector(i, &A_sub_col_i);
    A_sub_col_i.CopyValues(A_col_index_i);
  }
}

/**
 * Sets a matrix to a centered matrix, where centering is done by subtracting
 * the sum over the columns (a column vector) from each column of the matrix
 */
void Center(Matrix X, Matrix* X_centered) {
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

  X_centered -> Copy(X);

  for(index_t i = 0; i < n; i++) {
    Vector cur_col_vector;
    X_centered -> MakeColumnVector(i, &cur_col_vector);
    la::SubFrom(col_vector_sum, &cur_col_vector);
  }

}

/**
 * Whitens a matrix using the singular value decomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is
 * the identity matrix
 */
void WhitenUsingSVD(Matrix X, Matrix* X_whitened, Matrix* whitening_matrix) {
  
  Matrix cov_X, U, VT, inv_S_matrix, temp1;
  Vector S_vector;
  
  Scale(1 / (double) (X.n_cols() - 1),
	MulTransBInit(&X, &X, &cov_X));
  
  la::SVDInit(cov_X, &S_vector, &U, &VT);
  
  index_t d = S_vector.length();
  inv_S_matrix.Init(d, d);
  inv_S_matrix.SetZero();
  for(index_t i = 0; i < d; i++) {
    double inv_sqrt_val = 1 / sqrt(S_vector[i]);
    inv_S_matrix.set(i, i, inv_sqrt_val);
  }

  cov_X.PrintDebug("cov(X')");
  U.PrintDebug("U");
  VT.PrintDebug("VT");
  inv_S_matrix.PrintDebug("S^-.5");
  
  MulTransBInit(MulTransAInit(&VT, &inv_S_matrix, &temp1),
		&U,
		whitening_matrix);
  
  MulInit(whitening_matrix, &X, X_whitened);
  
}

/**
 * Whitens a matrix using the eigen decomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is
 * the identity matrix
 */
void WhitenUsingEig(Matrix X, Matrix* X_whitened, Matrix* whitening_matrix, Matrix* dewhitening_matrix) {
  Matrix cov_X, D, D_inv, E;
  Vector D_vector;

  Scale(1 / (double) (X.n_cols() - 1),
	MulTransBInit(&X, &X, &cov_X));
    

  la::EigenvectorsInit(cov_X, &D_vector, &E);

  //E.set(0, 1, -E.get(0, 1));
  //E.set(1, 1, -E.get(1, 1));

    

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

  la::MulTransBInit(D_inv, E, whitening_matrix);
  la::MulInit(E, D, dewhitening_matrix);
  la::MulInit(*whitening_matrix, X, X_whitened);
}

/**
 * Overwrites a dimension-N vector to a random vector on the unit sphere in R^N
 */
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

/**
 * Inits a matrix to random normally distributed entries from N(0,1)
 */
Matrix* RandNormalInit(index_t d, index_t n, Matrix* A) {

  double* A_elements = A -> ptr();

  index_t num_elements = d * n;

  for(index_t i = 0; i+1 < num_elements; i+=2) {
    double a = drand48();
    double b = drand48();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    A_elements[i] =   first_term * cos(second_term);
    A_elements[i+1] = first_term * sin(second_term);
  }
  
  if((d % 2) == 1) {
    A_elements[d - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
  }

  return A;
}

/**
 * Inits a matrix to a num_row_reps by num_col_reps block matrix where
 * each block is base_matrix
 */
Matrix* RepeatMatrix(index_t num_row_reps, index_t num_col_reps,
		     Matrix base_matrix, Matrix* new_matrix) {

  index_t num_rows = base_matrix.n_rows();
  index_t num_cols = base_matrix.n_cols();

  new_matrix -> Init(num_rows * num_row_reps, num_cols * num_col_reps);
  
  double* base_elements;
  double* new_elements = new_matrix -> ptr();
    
  for(index_t col_rep = 0; col_rep < num_col_reps; col_rep++) {
    base_elements = base_matrix.ptr();
    for(index_t col_num = 0; col_num < num_cols; col_num++) {
      for(index_t row_rep = 0; row_rep < num_row_reps; row_rep++) {
	memcpy(new_elements, base_elements, num_rows * sizeof(double));
	new_elements += num_rows;
      }
      base_elements += num_rows;
    }
  }

  return new_matrix;
}

