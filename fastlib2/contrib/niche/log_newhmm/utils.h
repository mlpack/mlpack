#ifndef UTILS_H
#define UTILS_H

#define NEG_INFTY -std::numeric_limits<double>::infinity()

template <typename T>
void KillDuplicatePoints(const GenMatrix<T> &data,
			 GenMatrix<T> *p_new_data,
			 Matrix* p_weights) {
  GenMatrix<T> &new_data = *p_new_data;
  Matrix &weights = *p_weights;

  int n_points = data.n_cols();
  int n_dims = data.n_rows();

  ArrayList<int> weights_arraylist;
  weights_arraylist.Init(1);

  ArrayList<GenVector<T> > new_points_arraylist;
  new_points_arraylist.Init(1);

  new_points_arraylist[0].Copy(data.GetColumnPtr(0), n_dims);
  weights_arraylist[0] = 1;

  GenVector<T> last_point;
  last_point.Alias(new_points_arraylist[0]);

  int cur_point_num = 0;
  for(int i = 1; i < n_points; i++) {
    GenVector<T> cur_point;
    data.MakeColumnVector(i, &cur_point);
    
    if(VectorEquals(cur_point, last_point)) {
      weights_arraylist[cur_point_num]++;
    }
    else {
      new_points_arraylist.PushBackCopy(cur_point);
      cur_point_num++;
      weights_arraylist.Resize(cur_point_num + 1);
      weights_arraylist[cur_point_num] = 1;
      last_point.Destruct();
      last_point.Alias(cur_point);
    }
  }

  int n_distinct_points = cur_point_num + 1;

  weights.Init(1, n_distinct_points);
  for(int i = 0; i < n_distinct_points; i++) {
    weights.set(0, i, weights_arraylist[i]);
  }

  new_data.Init(n_dims, n_distinct_points);
  for(int i = 0; i < n_distinct_points; i++) {
    new_data.CopyVectorToColumn(i, new_points_arraylist[i]);
  }
}

template <typename T>
bool VectorEquals(const GenVector<T> &x,
		  const GenVector<T> &y) {
  int n_dims = x.length();
  for(int i = 0; i < n_dims; i++) {
    if(x[i] != y[i]) {
      return false;
    }
  }
  
  return true;
}

template<typename T>
void WriteOutOTObject(const char* filename,
		      const T &object) {
  size_t size = ot::FrozenSize(object);
  char* buf = mem::Alloc<char>(size);
  ot::Freeze(buf, object);

  FILE* file = fopen(filename, "wb");
  fwrite(&size, sizeof(size), 1, file);
  fwrite(buf, 1, size, file);
  fclose(file);

  mem::Free(buf);
}

template<typename T>
void ReadInOTObject(const char* filename,
		    T* p_object) {
  size_t size;

  FILE* file = fopen(filename, "rb");
  fread(&size, sizeof(size), 1, file);
  char* buf = mem::Alloc<char>(size);
  fread(buf, 1, size, file);
  fclose(file);

  ot::InitThaw(p_object, buf);
  mem::Free(buf);
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

// for time series observations in R^n for n >= 1
void LoadVaryingLengthData(const char* filename,
			   ArrayList<GenMatrix<double> >* p_data) {
  ArrayList<GenMatrix<double> > &data = *p_data;
  
  data.Init();

  FILE* file = fopen(filename, "r");

  char* buffer = (char*) malloc(sizeof(char) * 70000);
  size_t len = 70000;
  int n_read;

  while((n_read = getline(&buffer, &len, file)) != -1) {
    if(buffer[0] != '%') {
      break;
    }
  }
  fclose(file);
  
  bool on_number = false;
  int n_numbers = 0;
  for(int i = 0; i < n_read; i++) {
    int c = buffer[i];
    if((('0' <= c) && (c <= '9'))
       || (c == '-')
       || (c == '.')
       || (c == 'e')
       || (c == 'E')) {
      if(!on_number) {
	on_number = true;
	n_numbers++;
      }
    }
    else {
      on_number = false;
    }
  }
  
  int n_dims = n_numbers;
  printf("n_dims = %d\n", n_dims);


 
  file = fopen(filename, "r");
 

  int n_elements = 0;
  bool waiting = true;
  
  ArrayList<Vector> point_list;
  point_list.Init();

  int n_points_in_cur_point_list = 0;


  while((n_read = getline(&buffer, &len, file)) != -1) {
    if(buffer[0] == '%') {
      if(waiting == false) {
	//printf("n_points_in_cur_point_list = %d\n", n_points_in_cur_point_list);
	GenMatrix<double> &data_sequence = data[n_elements - 1];
	data_sequence.Init(n_dims, n_points_in_cur_point_list);
	for(int i = 0; i < n_points_in_cur_point_list; i++) {
	  data_sequence.CopyVectorToColumn(i, point_list[i]);
	}
      }
      waiting = true;
      continue;
    }
    else {
      if(waiting) {
	waiting = false;
	n_elements++;
	data.GrowTo(n_elements);
    
	n_points_in_cur_point_list = 0;
	point_list.Renew();
	point_list.Init();
      }

      n_points_in_cur_point_list++;
      point_list.GrowTo(n_points_in_cur_point_list);
      Vector &new_point = point_list[n_points_in_cur_point_list - 1];
      new_point.Init(n_dims);
      double* new_point_ptr = new_point.ptr();
      char* buffer_copy = strdup(buffer);
      char* token = strtok(buffer_copy, " \t,");
      sscanf(token, "%lf", new_point_ptr);

      int n_dims_read = 1;
      while((token = strtok(NULL, " \t,")) != NULL) {
	sscanf(token, "%lf", new_point_ptr + n_dims_read);
	n_dims_read++;
      }

      free(buffer_copy);
    }
  }

  if(waiting == false) {
    //printf("n_points_in_cur_point_list = %d\n", n_points_in_cur_point_list);
    GenMatrix<double> &data_sequence = data[n_elements - 1];
    data_sequence.Init(n_dims, n_points_in_cur_point_list);
    for(int i = 0; i < n_points_in_cur_point_list; i++) {
      data_sequence.CopyVectorToColumn(i, point_list[i]);
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

void NormalizeKernelMatrixLog(Matrix* p_kernel_matrix_log) {
  Matrix &kernel_matrix_log = *p_kernel_matrix_log;
  
  int n_points = kernel_matrix_log.n_rows();
  
  Vector half_diag;
  half_diag.Init(n_points);
  for(int i = 0; i < n_points; i++) {
    half_diag[i] = 0.5 * kernel_matrix_log.get(i, i);
  }
  for(int i = 0; i < n_points; i++) {
    for(int j = 0; j < n_points; j++) {
      kernel_matrix_log.set(j, i,
			    exp(kernel_matrix_log.get(j, i)
				- half_diag[i] - half_diag[j]));
    }
  }
}



double LogSumExp(double x, double y) {
  if(x > y) {
    return x + log(1 + exp(y - x));
  }
  else {
    if(y == NEG_INFTY) {
      return NEG_INFTY;
    }
    else {
      return y + log(1 + exp(x - y));
    }
  }
}


double LogSumExp(const Vector &x) {

  int n_dims = x.length();
  
  double max = -std::numeric_limits<double>::max();
  
  for(int i = 0; i < n_dims; i++) {
    if(unlikely(x[i] > max)) {
      max = x[i];
    }
  }

  if(max == NEG_INFTY) {
    return NEG_INFTY;
  }
  else {
    double sum = 0;
    for(int i = 0; i < n_dims; i++) {
      sum += exp(x[i] - max);
    }
    return max + log(sum);
  }
}

double LogSumExp(const double* x, int n_dims) {

  double max = -std::numeric_limits<double>::max();
  
  for(int i = 0; i < n_dims; i++) {
    if(unlikely(x[i] > max)) {
      max = x[i];
    }
  }

  if(max == NEG_INFTY) {
    return NEG_INFTY;
  }
  else {
    double sum = 0;
    for(int i = 0; i < n_dims; i++) {
      sum += exp(x[i] - max);
    }
    return max + log(sum);
  }
}

    

double LogSumMapExpVectors(const Vector &a, const Vector &b) {

  int n_dims = a.length();
  
  double max = -std::numeric_limits<double>::max();
  
  double sum;
  for(int i = 0; i < n_dims; i++) {
    sum = a[i] + b[i];
    if(unlikely(sum > max)) {
      max = sum;
    }
  }

  if(max == NEG_INFTY) {
    return NEG_INFTY;
  }
  else {
    sum = 0;
    for(int i = 0; i < n_dims; i++) {
      sum += exp(a[i] + b[i] - max);
    }
    return max + log(sum);
  }
}



void LogMatrixMultiplyOverwrite(const Matrix &A,
				const Matrix &B,
				Matrix* p_C) {
  Matrix& C = *p_C;

  Matrix At;
  la::TransposeInit(A, &At);

  index_t n_At_cols = At.n_cols();
  index_t n_B_cols = B.n_cols();

  for(int k = 0; k < n_B_cols; k++) {
    Vector B_k;
    B.MakeColumnVector(k, &B_k);
    
    for(int i = 0; i < n_At_cols; i++) {
      Vector A_i;
      At.MakeColumnVector(i, &A_i);
      
      C.set(i, k, LogSumMapExpVectors(A_i, B_k));
    }
  }
}

void LogMatrixMultiplyATransOverwrite(const Matrix &At,
				      const Matrix &B,
				      Matrix* p_C) {
  Matrix& C = *p_C;

  index_t n_At_cols = At.n_cols();
  index_t n_B_cols = B.n_cols();

  for(int k = 0; k < n_B_cols; k++) {
    Vector B_k;
    B.MakeColumnVector(k, &B_k);
    
    for(int i = 0; i < n_At_cols; i++) {
      Vector A_i;
      At.MakeColumnVector(i, &A_i);
      
      C.set(i, k, LogSumMapExpVectors(A_i, B_k));
    }
  }
}

void LogMatrixMultiplyATransOverwriteResultTrans(const Matrix &At,
						 const Matrix &B,
						 Matrix* p_Ct) {
  Matrix& Ct = *p_Ct;

  index_t n_At_cols = At.n_cols();
  index_t n_B_cols = B.n_cols();

  for(int i = 0; i < n_At_cols; i++) {
    Vector A_i;
    At.MakeColumnVector(i, &A_i);
  
    for(int k = 0; k < n_B_cols; k++) {
      Vector B_k;
      B.MakeColumnVector(k, &B_k);
      
      Ct.set(k, i, LogSumMapExpVectors(A_i, B_k));
    }
  }
}



void LogMatrixMultiplyATransOverwrite(const Matrix &At,
				      const Vector &x,
				      Vector* p_y) {
  Vector& y = *p_y;

  index_t n_At_cols = At.n_cols();

  for(int i = 0; i < n_At_cols; i++) {
    Vector A_i;
    At.MakeColumnVector(i, &A_i);

    y[i] = LogSumMapExpVectors(A_i, x);
  }
}


void LogMatrixMultiplyOverwrite(const Vector &x,
				const Matrix &B,
				Vector* p_y) {
  Vector& y = *p_y;
  
  index_t n_B_cols = B.n_cols();
  
  for(int k = 0; k < n_B_cols; k++) {
    Vector B_k;
    B.MakeColumnVector(k, &B_k);
    
    y[k] = LogSumMapExpVectors(x, B_k);
  }
}


#endif /* UTILS_H */
