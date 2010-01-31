#include "fl_data_io.h"

/* Compute the set of data rows that belong to this processor*/
bool 
par_split_data(int *startrow, int *my_numrows, int total_numrows, int pe, 
	       int numPE)
{
  int extrarows = total_numrows%numPE;

  if (extrarows > pe)
  {
    *my_numrows = (int)(total_numrows/numPE)+1;
    *startrow = pe*(*my_numrows) + 1;
  }
  else
  {
    *my_numrows = (int)(total_numrows/numPE);
    *startrow = pe*(*my_numrows) + extrarows + 1;
  }

  return true;
}

/* Write the Matrix into a binary file. Matrix is saved row major*/
bool 
write_matrix2bin(const char* fname, Matrix &mat)
{
  FILE *fp;
  int rows, cols;

  if (!(fp = fopen(fname, "w")))
    return false;
  
  // We are flipping the definition of rows and columns here to store the 
  // data matrix row major
  rows = mat.n_cols();
  cols = mat.n_rows();
  fwrite(&rows, sizeof(int), 1, fp);
  fwrite(&cols, sizeof(int), 1, fp);

  for (int i=0; i<rows; i++) {
    double* data_vec = mat.GetColumnPtr(i);
    fwrite(data_vec, sizeof(double), cols, fp);
  }

  fclose(fp);

  return true;
}

/* This function, unlike the write function takes a file pointer and requires 
the number of columns to be passed in since it is assumed that the calling 
method would have opened this file to read the dimensions of the data any way
This method reads row by row and saves each row into the column of the matrix 
passed in. So the Matrix passed in should of size cols x rows, i.e. transpose
*/

bool 
read_subset_bin2matrix(FILE* fp, Matrix &data, int start_row, int num_rows, 
		       int num_cols)
{

  fseek(fp, 
	(2*sizeof(int))+((start_row - 1)*num_cols*sizeof(double)), 
	SEEK_SET);
 
 for (int i=0; i < num_rows; i++) {
    double* data_vec = data.GetColumnPtr(i);
    fread(data_vec, sizeof(double), num_cols, fp);
  }
    
  fclose(fp);

  return true;
} 

