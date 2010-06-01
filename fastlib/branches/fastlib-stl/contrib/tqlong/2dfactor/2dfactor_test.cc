#include <fastlib/fastlib.h>
#include "2dfactor.h"
#include <iostream>

const fx_entry_doc __2dfactor_entries[] = {
  /*  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"fileTR", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM transition.\n"},
  {"fileE", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM emission.\n"},
  {"length", FX_PARAM, FX_INT, NULL,
   "  Sequence length, default = 10.\n"},
  {"lenmax", FX_PARAM, FX_INT, NULL,
   "  Maximum sequence length, default = length\n"},
  {"numseq", FX_PARAM, FX_INT, NULL,
   "  Number of sequance, default = 10.\n"},
  {"fileSEQ", FX_PARAM, FX_STR, NULL,
   "  Output file for the generated sequences.\n"},
  */
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc __2dfactor_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc __2dfactor_doc = {
  __2dfactor_entries, __2dfactor_submodules,
  "This is a program for 2d data factorization.\n"
};

#define N_SUB 40
#define N_INSTANCE 10

void readAllORLimages(ArrayList<Matrix>& imageList);
void projectAllORLimages(const ArrayList<Matrix>& imageList,
			 const Matrix& rowBasis, const Matrix& colBasis);

namespace data {
  void Append(const char* fname, const Vector& vector, bool append = true);
  void Append(const char* fname, const Matrix& matrix, bool append = true);
  void Append(const char* fname, const char* str, bool append = true);
  void Append(const char* fname, index_t num, bool append = true);
  void Append(const char* fname, double num, bool append = true);
  void Load(FILE* f, Vector* vector, index_t length);
  void Load(FILE* f, Matrix* matrix, index_t n_rows, index_t n_cols);
  void Load(FILE* f, const char* str); // check point
  void Load(FILE* f, index_t* num);
  void Load(FILE* f, double* num);
};

void SaveResult(const Vector& rowEigenValues, const Matrix& rowBasis, 
		const Vector& colEigenValues, const Matrix& colBasis, 
		const Matrix& mean) {
  char filename[] = "2dPCA40sub10inst.txt";
  data::Append(filename, "n_rows=", false); // create new file
  data::Append(filename, colEigenValues.length());
  data::Append(filename, "n_cols=");
  data::Append(filename, rowEigenValues.length());
  data::Append(filename, "rowEigenValues=");
  data::Append(filename, rowEigenValues);
  data::Append(filename, "rowBasis=");
  data::Append(filename, rowBasis);
  data::Append(filename, "colEigenValues=");
  data::Append(filename, colEigenValues);
  data::Append(filename, "colBasis=");
  data::Append(filename, colBasis);
  data::Append(filename, "mean=");
  data::Append(filename, mean);
}

void LoadResult(Vector& rowEigenValues, Matrix& rowBasis, 
		Vector& colEigenValues, Matrix& colBasis, 
		Matrix& mean) {
  char filename[] = "2dPCA40sub10inst.txt";
  index_t n_rows, n_cols;
  FILE *f = fopen(filename, "r");
  data::Load(f, "n_rows=");
  data::Load(f, &n_rows); //ot::Print(n_rows);
  data::Load(f, "n_cols=");
  data::Load(f, &n_cols); //ot::Print(n_cols);
  data::Load(f, "rowEigenValues=");
  data::Load(f, &rowEigenValues, n_cols); //ot::Print(rowEigenValues);
  data::Load(f, "rowBasis=");
  data::Load(f, &rowBasis, n_cols, n_cols);
  data::Load(f, "colEigenValues=");
  data::Load(f, &colEigenValues, n_rows); //ot::Print(colEigenValues);
  data::Load(f, "colBasis=");
  data::Load(f, &colBasis, n_rows, n_rows);
  data::Load(f, "mean=");
  data::Load(f, &mean, n_rows, n_cols);
  fclose(f);
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &__2dfactor_doc );

  ArrayList<Matrix> imageList;
  readAllORLimages(imageList);

  Vector rowEigenValues, colEigenValues;
  Matrix rowBasis, colBasis, mean;

  /* // Compute 2dPCA of 40 subjects (10 instances each)
  la::RowCol2dPCA(imageList, rowEigenValues, rowBasis, 
	      colEigenValues, colBasis, mean);
  SaveResult(rowEigenValues, rowBasis, colEigenValues, colBasis, mean);
  */

  LoadResult(rowEigenValues, rowBasis, colEigenValues, colBasis, mean);
  
  Matrix rowBasisMajor, colBasisMajor;
  la::Get2dBasisMajor(0.95, rowEigenValues, rowBasis,
		      0.95, colEigenValues, colBasis,
		      rowBasisMajor, colBasisMajor);

  std::cout << "rowMajor = " << rowBasisMajor.n_rows() << " x "
	    << rowBasisMajor.n_cols() << std::endl;
  std::cout << "colMajor = " << colBasisMajor.n_rows() << " x "
	    << colBasisMajor.n_cols() << std::endl;
  /*
  {
    Matrix tmp;
    la::MulTransAInit(rowBasisMajor, rowBasisMajor, &tmp);
    ot::Print(tmp);
  }
  */

  projectAllORLimages(imageList, rowBasisMajor, colBasisMajor);

  fx_done(NULL);
}

void readAllORLimages(ArrayList<Matrix>& imageList){
  imageList.Init();
  for (int i_sub = 1; i_sub <= N_SUB; i_sub++) {
    std::cout << i_sub << std::endl;
    for (int i_instance = 1; i_instance <= N_INSTANCE; i_instance++) {
      Matrix A;
      char filename[256];
      sprintf(filename, "./orl_faces/s%d/%d.csv", i_sub, i_instance);
      data::Load(filename, &A);
      DEBUG_ASSERT(A.n_rows() == 112 && A.n_cols() == 92);
      imageList.PushBackCopy(A);
    }
  }
}

void projectAllORLimages(const ArrayList<Matrix>& imageList,
			 const Matrix& rowBasis, const Matrix& colBasis) {
  int k = 0;
  for (int i_sub = 1; i_sub <= N_SUB; i_sub++) {
    std::cout << i_sub << std::endl;
    for (int i_instance = 1; i_instance <= N_INSTANCE; i_instance++) {
      Matrix A;
      char filename[256];
      sprintf(filename, "./orl_faces/s%d/%d-nmf.csv", i_sub, i_instance);
      la::Project2dBasis(imageList[k], rowBasis, colBasis, A);
      data::Save(filename, A);
      DEBUG_ASSERT(A.n_rows() == 112 && A.n_cols() == 92);
      k++;
    }
  }
  
}

namespace data {
  void Append(const char* fname, const Vector& vector, bool append) {    
    FILE* f = fopen(fname, append? "a":"w");
    for (index_t i = 0; i < vector.length(); i++)
      fprintf(f, "%e\n", vector[i]);
    fclose(f);
  }
  void Append(const char* fname, const Matrix& matrix, bool append) {
    FILE* f = fopen(fname, append? "a":"w");
    for (index_t j = 0; j < matrix.n_cols(); j++) { // column wise
      for (index_t i = 0; i < matrix.n_rows(); i++)
	fprintf(f, "%e ", matrix.get(i, j));
      fprintf(f, "\n");
    }
    fclose(f);    
  }
  void Append(const char* fname, const char* str, bool append) {
    FILE* f = fopen(fname, append? "a":"w");
    fprintf(f, "%s\n", str);
    fclose(f);    
  }
  void Append(const char* fname, index_t num, bool append) {
    FILE* f = fopen(fname, append? "a":"w");
    fprintf(f, "%d\n", num);
    fclose(f);    
  }
  void Append(const char* fname, double num, bool append) {
    FILE* f = fopen(fname, append? "a":"w");
    fprintf(f, "%e\n", num);
    fclose(f);    
  }

  void Load(FILE* f, Vector* vector, index_t length) {
    double tmp;
    vector->Init(length);
    for (index_t i = 0; i < length; i++) {
      fscanf(f, "%lg\n", &tmp);
      (*vector)[i] = tmp;
    }
  }
  void Load(FILE* f, Matrix* matrix, index_t n_rows, index_t n_cols) {
    double tmp;
    matrix->Init(n_rows, n_cols);
    for (index_t j = 0; j < n_cols; j++) { // column wise
      for (index_t i = 0; i < n_rows; i++) {
	fscanf(f, "%lg ", &tmp);
	matrix->ref(i, j) = tmp;
      }
      fscanf(f, "\n");
    }
  }
  void Load(FILE* f, const char* str) { // check point
    char s[256];
    fgets(s, 256, f);
    if (s[strlen(s)-1] == '\n') s[strlen(s)-1] = '\0';
    //printf("%s\n", s);
    //printf("%s\n", str);
    
    DEBUG_ASSERT(strcmp(s, str) == 0);
  }
  void Load(FILE* f, index_t* num) {
    fscanf(f, "%d\n", num);
  }
  void Load(FILE* f, double* num) {
    fscanf(f, "%lg\n", num);
  }

};

