#include "fastlib/fastlib.h"



double TimesTen(double x) {
  return 10 * x;
}

double PlusOne(double x) {
  return x + 1;
}


void SaveCorrectly(const char *filename, Matrix a) {
  Matrix a_transpose;
  la::TransposeInit(a, &a_transpose);
  data::Save(filename, a_transpose);
}
  


Matrix Mul(Matrix A, Matrix B, Matrix *C) {
  la::MulInit(A, B, C);

  return *C;
}

Matrix Sub(Matrix A, Matrix B, Matrix *C) {
  la::SubInit(B, A, C);

  return *C;
}


Matrix Map1(double (*function)(double), Matrix *A) {
  index_t n_rows = A->n_rows();
  index_t n_cols = A->n_cols();

  for(index_t j = 0; j < n_cols; j++) {
    for(index_t i = 0; i < n_rows; i++) {
      A->set(i, j, function(A->get(i, j)));
    }
  }

  return *A;
}

Matrix Map2(double (*function)(double), Matrix *A) {
  index_t n_rows = A->n_rows();
  index_t n_cols = A->n_cols();

  Vector A_col_j;
  for(index_t j = 0; j < n_cols; j++) {
    Vector A_col_j;
    A->MakeColumnVector(j, &A_col_j);
    for(index_t i = 0; i < n_rows; i++) {
      A_col_j[i] = function(A_col_j[i]);
    }
  }

  return *A;
}



void RandMatrix(index_t n_rows, index_t n_cols, Matrix *A) {
  A->Init(n_rows, n_cols);

  for(index_t j = 0; j < n_cols; j++) {
    for(index_t i = 0; i < n_rows; i++) {
      A->set(i, j, drand48());
    }
  }
}

int main(int argc, char *argv[]) {
  //fx_init(argc, argv);
  /*
  Matrix A, B, C, D, E;

  RandMatrix(5, 2, &A);
  RandMatrix(2, 4, &B);
  RandMatrix(5, 4, &C);
  RandMatrix(7, 5, &D);
  RandMatrix(7, 4, &E);

  SaveCorrectly("A.dat", A);
  SaveCorrectly("B.dat", B);
  SaveCorrectly("C.dat", C);
  SaveCorrectly("D.dat", D);
  SaveCorrectly("E.dat", E);


  Matrix *temp1 = new Matrix;
  Matrix *temp2 = new Matrix;
  Matrix *temp3 = new Matrix;
  Matrix *temp4 = new Matrix;
  

  Sub(Mul(D, Sub(Mul(A, B, temp1), C, temp2), temp3), E, temp4);
						

  
  A.PrintDebug("A");
  B.PrintDebug("B");
  C.PrintDebug("C");
  D.PrintDebug("D");
  E.PrintDebug("E");


  temp4->PrintDebug("final result");

  
  delete temp1;
  delete temp2;
  delete temp3;
  delete temp4;

  */

  fx_init(argc, argv);




  Matrix G;
  RandMatrix(800, 30, &G);



  //G.PrintDebug("G");


  /*
    fx_timer_start(NULL, "map1");
    for(index_t i = 0; i < 1e5; i++) {
    Map1(&PlusOne, &G);
    }
    fx_timer_stop(NULL, "map1");
  */
  
  fx_timer_start(NULL, "map2");
  for(index_t i = 0; i < 1e5; i++) {
    Map2(&PlusOne, &G);
  }
  fx_timer_stop(NULL, "map2");
  




  //G.PrintDebug("Map(TimesTen, G)");

  fx_done();

  return 0;

}
