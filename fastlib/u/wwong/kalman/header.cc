#include <iostream>
#include "header.h"
#include "fastlib/fastlib.h"

using namespace std;

Vector propagate_one_step(const Matrix& a_mat, const Matrix& b_mat, const Vector& x, const Vector &u, const Vector& w)  {
  // returns a_mat*x + b_mat*u + w, where output_vec_dim = A.n_rows() = B.n_rows()= w.length();
  // Place a debug_assert_msg here later

  Vector temp_1, temp_2;
  temp_1.Init(a_mat.n_rows());  temp_2.Init(a_mat.n_rows());
  la::MulInit(a_mat,x,&temp_1); 
  la::MulInit(b_mat,u,&temp_2);
  la::AddTo(temp_2,&temp_1); //  a_mat*x+b_mat*u
  la::AddTo(w,&temp_1); // temp_1 = a_mat*x + b_mat*u + w
  return temp_1;
}; 


void SsmTimeInvariantSignalGenerator(const int& T, const ssm& LDS, const Matrix& u, Matrix* w, Matrix* v, Matrix* x, Matrix *y)
{

  int nx   = LDS.A.n_cols();  int ny   = LDS.C.n_rows(); 

  Vector u_t, w_t,  v_t, x_t, x_next, y_t; // all aliases therefore no need to .Init(length) 
  Vector w_v_t; w_v_t.Init(nx+ny); // w_v_t = [w_t; v_t];

  Matrix noise_matrix(nx+ny, nx+ny), Strans;
  la::TransposeInit(LDS.S,&Strans);

  set_portion_of_matrix(LDS.Q,0,nx-1,0,nx-1,&noise_matrix);
  set_portion_of_matrix(LDS.S,0,nx-1,nx,nx+ny-1, &noise_matrix);
  set_portion_of_matrix(Strans,nx,nx+ny-1,0,nx-1, &noise_matrix);
  set_portion_of_matrix(LDS.R,nx,nx+ny-1,nx,nx+ny-1,&noise_matrix); // noise_matrix = [Q S; S' R];
  
  for (int t =0; t<=T;t++)
    { 
      u.MakeColumnVector(t, &u_t);
      (*w).MakeColumnVector(t, &w_t);
      (*v).MakeColumnVector(t, &v_t);
      (*x).MakeColumnVector(t, &x_t);
      (*x).MakeColumnVector((t+1), &x_next);
      (*y).MakeColumnVector(t, &y_t) ;

      RandVector(w_v_t, noise_matrix);
      extract_sub_vector_of_vector_init(w_v_t,0,nx-1, &w_t);
      extract_sub_vector_of_vector_init(w_v_t,nx,nx+ny-1, &v_t);      

      x_next.CopyValues(propagate_one_step(LDS.A, LDS.B, x_t, u_t,w_t)); // x_{t+1} = Ax_t + Bu_t + w_t;
      y_t.CopyValues(propagate_one_step(LDS.C, LDS.D, x_t, u_t,v_t)); // y_t     = Cx_t + Du_t + v_t;

      if (t ==T)
	{
	  u_t.Destruct(); v_t.Destruct(); y_t.Destruct();           
	  (u).MakeColumnVector(t+1, &u_t);
	  (*v).MakeColumnVector(t+1, &v_t);
	  (*y).MakeColumnVector(t+1, &y_t);
	  y_t.CopyValues(propagate_one_step(LDS.C, LDS.D, x_next, u_t,v_t)); // y_{T+1}    = Cx_{T+1} + Du_{T+1} + v_{T+1};
	  break;	 
	}
      else
	{
	   u_t.Destruct(); w_t.Destruct(); v_t.Destruct(); x_t.Destruct(); x_next.Destruct(); y_t.Destruct();           
	};

      };

 };

Matrix schur(const Matrix& a_mat, const Matrix& b_mat, const Matrix& c_mat, const Matrix& d_mat)
{
//returns  a_mat - b_mat*inv(c_mat)*d_mat

Matrix temp_1, temp_2, temp_3; 
temp_1.Copy(c_mat); la::Inverse(&temp_1); // temp1 = inv(c_mat)
la::MulInit(b_mat,temp_1,&temp_2); // temp_2 = b_mat*inv(c_mat)
la::MulInit(temp_2,d_mat,&temp_3); // temp_3 = b_mat*inv(c_mat)*d_mat
Matrix result; 
result.Copy(a_mat);
la::AddExpert(-1,temp_3,&result); 
return result;
};


void print_matrix(const Matrix& a_mat, const char* name)
{
  cout<<endl<<"Printing Matrix.."<<name<<endl;;
  for (int r = 1; r<=(int)a_mat.n_rows();r++)
     {
       for (int c=1; c<=(int)a_mat.n_cols();c++)
	 {
	   std::cout<<" "<<a_mat.get(r-1,c-1)<<" ";
	 };
       std::cout<<std::endl<<std::endl;
     };
};


void matrix_concatenate_col_init(const Matrix& a_mat, const Matrix& b_mat, Matrix* x_mat)
{
  // Initializes x_mat such that x_mat <- [a_mat | b_mat]

  int n_rows = a_mat.n_rows(), n_cols = a_mat.n_cols() + b_mat.n_cols();
  (*x_mat).Init(n_rows, n_cols);
  
  for (int r1 = 0; r1<n_rows; r1++)
  {
    for (int c1 = 0; c1<a_mat.n_cols(); c1++)
   {
     (*x_mat).set(r1,c1,a_mat.get(r1,c1));
   };
  };

  for (int r2 = 0; r2<n_rows; r2++)
  {
    for (int c2 = a_mat.n_cols(); c2<n_cols; c2++)
   {
     (*x_mat).set(r2,c2,b_mat.get(r2,c2-a_mat.n_cols() ));
   };
  };

};


void matrix_concatenate_row_init(const Matrix& a_mat, const Matrix& b_mat, Matrix* x_mat)
{
  // Initializes x_mat such that x_mat <- [a_mat 
  //                                        --
  //                                       b_mat]

  int n_cols = a_mat.n_rows(), n_rows = a_mat.n_rows() + b_mat.n_rows();
  (*x_mat).Init(n_rows, n_cols);
  
  for (int r1 = 0; r1<a_mat.n_rows(); r1++)
  {
    for (int c1 = 0; c1<n_cols; c1++)
   {
     (*x_mat).set(r1,c1,a_mat.get(r1,c1));
   };
  };

  for (int r2 = a_mat.n_rows(); r2<n_rows; r2++)
  {
    for (int c2 = 0; c2<n_cols; c2++)
   {
     (*x_mat).set(r2,c2, b_mat.get(r2-a_mat.n_rows(),c2));
   };
  };

};

void extract_sub_matrix_init(const Matrix& a_mat, const int& r_in, const int& r_out, const int& c_in, const int& c_out, Matrix* x_mat)
{
  // Initialize x_mat with values the same as the appropriate sub-matrix of a_mat.
  // r_in = starting row_index; r_out, c_in and c_out sim. defined

  (*x_mat).Init(r_out-r_in+1,c_out-c_in +1);
  
  for (int r = 0; r< (*x_mat).n_rows(); r++)
    {
      for (int c = 0; c< (*x_mat).n_cols(); c++)
	  {
	    (*x_mat).set(r,c,a_mat.get(r_in +r, c_in +c));
	  };
    };
};


void extract_sub_vector_of_vector_init(const Vector& v, const int& r_in, const int& r_out, Vector* x)
{
  // Initialize vector x  with values the same as the appropriate sub-vector of v.
  // r_in = starting row_index; r_out

  (*x).Init(r_out-r_in+1);
  double temp [r_out-r_in+1];
  for (int r = 0; r< (*x).length(); r++)
    {
      temp[r] = v.get(r_in+r);
    };

  (*x).CopyValues(temp); // temp is already a ptr
};


void set_portion_of_matrix(const Matrix& a_mat, const int& r_in,const int& r_out, const int& c_in, const int& c_out,Matrix* x_mat)
{
  // sets value of sub-matrix of x_mat to those in matrix a_mat
  for (int r=r_in; r<=r_out; r++)
    {
      for (int c=c_in; c<=c_out; c++)
	{
	  (*x_mat).set(r,c, a_mat.get(r-r_in,c-c_in));
	};
    };

};

void set_portion_of_matrix(const Vector& a, const int& r_in,const int& r_out, const int& c, Matrix* x_mat)
{
  // sets value of sub-matrix of x_mat to those in vector a 
  for (int r=r_in; r<=r_out; r++)
    {
      	  (*x_mat).set(r,c, a.get(r-r_in));
    };

};


void RandVector(Vector &v, const Matrix& noise_mat) {

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

    Matrix noise_mat_sqrt, noise_mat_sqrt_trans; // noise_mat = noise_mat_sqrt*noise_mat_sqrt_trans

    Vector temp;

    la::CholeskyInit(noise_mat,&noise_mat_sqrt_trans);
    la::TransposeInit(noise_mat_sqrt_trans,&noise_mat_sqrt);
    la::MulInit(noise_mat_sqrt, v, &temp);
    v.CopyValues(temp);
};

