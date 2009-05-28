#ifndef UTILS_H
#define UTILS_H


template<typename T>
void WriteOutOTObject(const char* filename,
		      const T &object) {
  FILE* file = fopen(filename, "wb");
  index_t size = ot::FrozenSize(object);
  char* buf = mem::Alloc<char>(size);
  ot::Freeze(buf, object);
  fwrite(&size, sizeof(size), 1, file);
  fwrite(buf, 1, size, file);
  mem::Free(buf);
  fclose(file);
}

template<typename T>
void ReadInOTObject(const char* filename,
		    T* p_object) {
  FILE* file = fopen(filename, "rb");
  index_t size;
  fread(&size, sizeof(size), 1, file);
  char* buf = mem::Alloc<char>(size);
  fread(buf, 1, size, file);
  ot::InitThaw(p_object, buf);
  mem::Free(buf);
  fclose(file);
}


template<typename T>
void PrintDebug(const char* name, GenMatrix<T> x, const char* disp_format,
		FILE* stream = stderr) {
  char printstring[80];
  sprintf(printstring, "%s ", disp_format);
  
  int n_rows = x.n_rows();
  int n_cols = x.n_cols();
  fprintf(stream, "----- GENMATRIX<T> %s ------\n", name);
  for(int i = 0; i < n_rows; i++) {
    for(int j = 0; j < n_cols; j++) {
      fprintf(stream, printstring, x.get(i, j));
    }
    fprintf(stream, "\n");
  }
  fprintf(stream, "\n");
}

template<typename T>
void PrintDebug(const char* name, GenVector<T> x, const char* disp_format,
		FILE* stream = stderr) {
  char printstring[80];
  sprintf(printstring, "%s ", disp_format);
  
  int n_dims = x.length();
  fprintf(stream, "----- GENVECTOR<T> %s ------\n", name);
  for(int i = 0; i < n_dims; i++) {
    fprintf(stream, printstring, x[i]);
  }
  fprintf(stream, "\n");
}

void LoadVaryingLengthData(const char* filename,
			   ArrayList<GenMatrix<int> >* p_data) {
  ArrayList<GenMatrix<int> > &data = *p_data;
  
  data.Init();

  FILE* file = fopen(filename, "r");

  char* buffer = (char*) malloc(sizeof(char) * 70000);
  size_t len = 70000;


  int n_elements = 0;
  int n_read;
  while((n_read = getline(&buffer, &len, file)) != -1) {
    int sequence_length = (int) ((n_read - 1) / 2);

    n_elements++;
    data.GrowTo(n_elements);
  
    GenMatrix<int> &sequence = data[n_elements - 1];
    sequence.Init(1, sequence_length);
    int* sequence_ptr = sequence.ptr();
    for(int i = 0; i < sequence_length; i++) {
      sscanf(buffer + (2 * i), "%d", sequence_ptr + i);
    }
  }

  free(buffer);
  fclose(file);
}

void NormalizeKernelMatrix(Matrix* p_kernel_matrix) {
  Matrix &kernel_matrix = *p_kernel_matrix;

  int n_points = kernel_matrix.n_rows();
  
  Vector sqrt_diag;
  sqrt_diag.Init(n_points);
  for(int i = 0; i < n_points; i++) {
    sqrt_diag[i] = sqrt(kernel_matrix.get(i, i));
  }
  for(int i = 0; i < n_points; i++) {
    for(int j = 0; j < n_points; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i) /
			(sqrt_diag[i] * sqrt_diag[j]));
    }
  }
}


#endif /* UTILS_H */
