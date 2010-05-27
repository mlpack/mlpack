/**
 * @file infomax_ica.cc
 * @author Chip Mappus
 *
 * Methods for InfomaxICA.
 *
 * @see infomax_ica.h
 */

#include "infomax_ica.h"

/** 
 * Dummy constructor 
 */
InfomaxICA::InfomaxICA(){

}

InfomaxICA::InfomaxICA(double lambda, int b, double epsilon):
  lambda_(lambda),
  b_(b),
  epsilon_(epsilon){

}

/**
 * Sphere the data, apply ica. This is the first function to call
 * after initializing the variables.
 */
void InfomaxICA::applyICA(const Matrix &dataset){
  double current_cos=DBL_MAX;
  w_.Init(b_,b_);
  data_.Init(dataset.n_rows(),dataset.n_cols());
  la::ScaleOverwrite(1.0,dataset,&data_);
  if (b_<data_.n_cols()){
    sphere(data_);
    // initial estimate for w is Id
    Matrix i1 = eye(data_.n_rows(),double(1));
    la::ScaleOverwrite(1.0,i1,&w_);    
    while (epsilon_<=current_cos){
      Matrix w_prev;
      w_prev.Copy(w_);
      evaluateICA();
      current_cos = w_delta(w_prev,w_);
    }
  }
  else
    fprintf(stdout,"Window size must be less than number of instances.");
}

/**
 * Run infomax. Call this after initialization and applyICA.
 */
void InfomaxICA::evaluateICA(){
  Matrix BI = eye(w_.n_rows(),double(b_));
  // intermediate calculation variables
  Matrix icv = Matrix(w_.n_rows(),b_);
  Matrix icv2 = Matrix(w_.n_rows(),w_.n_cols());
  Matrix icv4 = Matrix(w_.n_rows(),w_.n_cols());
  for (index_t i=0;i<data_.n_cols();i+=b_){
    if (i+b_<data_.n_cols()){
      Matrix subm;
      data_.MakeColumnSlice(i,b_,&subm);
      printf("1\n");
      la::MulExpert(1,false,w_,false,subm,0.0,&icv);
      la::Scale(double(-1.0),&icv);
      expM(icv);  
      addOne(icv);
      invertVals(icv);
      la::Scale(-2.0,&icv);
      addOne(icv);
      printf("2\n");
      la::MulExpert(1,false,icv,true,subm,0.0,&icv2);
      la::AddOverwrite(icv2,BI,&icv4);
      la::Scale(lambda_,&icv4);
      printf("3\n");
      la::MulExpert(1,false,icv4,false,w_,0.0,&icv2);
      la::AddTo(w_.n_elements(),icv2.ptr(),w_.ptr());
    }
  }
}

// sphereing functions
/** 
 * Sphere the input data. 
 */
void InfomaxICA::sphere(Matrix &data){
  Matrix sample_covariance = sampleCovariance(data);
  Matrix wz = sqrtm(sample_covariance);
  Matrix wz_inverse = Matrix(wz.n_rows(),wz.n_cols());
  Matrix data_sub_means = subMeans(data);
  if (la::InverseOverwrite(wz,&wz_inverse)){
    la::Scale(wz.n_cols()*wz.n_rows(),2.0,wz_inverse.ptr());
    la::MulOverwrite(wz_inverse,data_sub_means,&data);
  }
}

// Covariance matrix.
Matrix InfomaxICA::sampleCovariance(const Matrix &m){
  Matrix ttm = subMeans(m);
  Matrix wm = Matrix(m.n_cols(),m.n_rows());  
  la::TransposeOverwrite(ttm,&wm);

  Matrix twm = Matrix(ttm.n_rows(),ttm.n_rows());
  Matrix output = Matrix(ttm.n_rows(),ttm.n_rows());
  output.SetZero();
  Matrix tttm = Matrix(wm);

  la::Scale(tttm.n_rows()*tttm.n_cols(),1/(double(ttm.n_cols())-1),tttm.ptr());
  la::MulTransAOverwrite(wm,tttm,&twm);
  la::AddOverwrite(twm.n_rows()*twm.n_cols(),twm.ptr(),twm.ptr(),output.ptr());
  la::Scale(output.n_rows()*output.n_cols(),double(0.5),output.ptr());
  return output;
}

Matrix InfomaxICA::subMeans(const Matrix &m){
  Matrix output = Matrix(m);
  Vector row_means = rowMean(output);
  la::Scale(row_means.length(),-1.0,row_means.ptr());
  for (index_t j=0;j<output.n_cols();j++){
    la::AddTo(row_means.length(),row_means.ptr(),output.GetColumnPtr(j));
  }  
  return output;
}

/** 
 * Compute the sample mean of a column 
 */
Vector InfomaxICA::rowMean(const Matrix &m){
  Vector row_means;
  row_means.Init(m.n_rows());
  row_means.SetZero();
  for (index_t j=0;j<m.n_cols();j++){
    la::AddTo(row_means.length(),m.GetColumnPtr(j),row_means.ptr());
  }
  la::Scale(row_means.length(),1/double(m.n_cols()),row_means.ptr());
  return row_means;
}

/** 
 *  Matrix square root using Cholesky decomposition method.  Assumes
 *  the input matrix is square.
 */
Matrix InfomaxICA::sqrtm(const Matrix &m){
  Matrix output = Matrix(m.n_rows(), m.n_cols());
  Matrix chol = Matrix(m);
  if (la::Cholesky(&chol)){
    Matrix u,vt;
    Vector s;
    la::TransposeSquare(&chol);
    if (la::SVDInit(chol,&s,&u,&vt)){
      Matrix S = Matrix(s.length(),s.length());
      Matrix tm1 = Matrix(u.n_rows(),S.n_cols());
      S.SetZero();
      S.SetDiagonal(s);
      printf("4\n");
      la::MulExpert(1,false,u,false,S,0.0,&tm1);
      printf("5\n");
      la::MulExpert(1,false,tm1,true,u,0.0,&output);
    }
    else
      fprintf(stderr,"infomaxICA sqrtm: SVD failed.\n");      
  }
  else
    fprintf(stderr,"infomaxICA sqrtm: Cholesky decomposition failed.\n");
  return output;
}

// Compare w estimates for convergence
double InfomaxICA::w_delta(const Matrix &w_prev, const Matrix &w_pres){
  Matrix temp;
  Vector delta_c;
  Vector delta_r;
  double delta_dot;
  la::SubInit(w_pres,w_prev,&temp);
  vectorize(temp,delta_r);
  vectorize(temp,delta_c);
  delta_dot = la::Dot(delta_r.length(),delta_r.ptr(),delta_c.ptr());
  fprintf(stderr,"w change=%f\n",delta_dot);
  return delta_dot;
}

// utility functions

/** 
 * Inplace exp for each matrix element 
 */
void InfomaxICA::expM(Matrix &m){
  for (index_t i=0;i<m.n_rows();i++){
    for (index_t j=0;j<m.n_cols();j++){
      m.set(i,j,exp(m.get(i,j)));
    }
  }
}

/** 
 * Inplace M+1 
 */
void InfomaxICA::addOne(Matrix &m){
  for (index_t i=0;i<m.n_rows();i++){
    for (index_t j=0;j<m.n_cols();j++){
      m.set(i,j,m.get(i,j)+1);
    }
  }
}

/** 
 * Inplace x_{ij}^-1 
 */
void InfomaxICA::invertVals(Matrix &m){
  for (index_t i=0;i<m.n_rows();i++){
    for (index_t j=0;j<m.n_cols();j++){
      m.set(i,j,1/m.get(i,j));
    }
  }
}

/** 
 * Identity matrix function 
 */
Matrix InfomaxICA::eye(index_t dim, double diagVal){
  Matrix output;
  Vector *diag = new Vector();
  diag->Init(dim);
  diag->SetAll(diagVal);
  output.InitDiagonal(*diag);
  return output;
}

/** 
 * Initialize the vector v to values of the matrix m. 
 */
void InfomaxICA::vectorize(const Matrix &m, Vector &v){
  index_t v_ind = 0;
  v.Init(m.n_rows()*m.n_cols());
  for (index_t i=0;i<m.n_rows();i++){
    for (index_t j=0;j<m.n_cols();j++){
      v[v_ind]=m.get(i,j);
      v_ind++;
    }
  }
  
}

/** 
 * Simple display matrix function 
 */
void InfomaxICA::displayMatrix(const Matrix &m){
  fprintf(stdout,"\n");
  for (index_t i=0;i<m.n_rows();i++){
    for (index_t j=0;j<m.n_cols();j++){
      fprintf(stdout,"%f ",m.get(i,j));
    }
    fprintf(stdout,"\n");
  }  
}

/** 
 * Simple display vector function 
 */
void InfomaxICA::displayVector(const Vector &m){
  fprintf(stdout,"\n");
  for (index_t i=0;i<m.length();i++){
    fprintf(stdout,"%f ",m[i]);
  }
  fprintf(stdout,"\n");
}

/** 
 * Return the current unmixing matrix estimate. Requires a reference
 * to an uninitialized matrix.
 */
void InfomaxICA::getUnmixing(Matrix &w){
  w.Copy(w_);
}

/** 
 * Return the source estimates, S. S is a reference to an
 * uninitialized matrix.
 */
void InfomaxICA::getSources(const Matrix &dataset, Matrix &s){
  s.Init(dataset.n_rows(),dataset.n_cols());
  la::MulExpert(1.0,false,w_,false,dataset,0.0,&s);
}

void InfomaxICA::setLambda(const double lambda){
  lambda_=lambda;
}

void InfomaxICA::setB(const int b){
  b_=b;
}

void InfomaxICA::setEpsilon(const double epsilon){
  epsilon_=epsilon;
}
