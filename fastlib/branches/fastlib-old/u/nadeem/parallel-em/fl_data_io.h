#ifndef FL_DATA_IO_H
#define FL_DATA_IO_H

#include "fastlib/fastlib.h"
#include <stdio.h>

bool par_split_data(int *startrow, int *my_numrows, int total_numrows, int pe, 
		    int numPE);
bool write_matrix2bin(const char* fname, Matrix &mat);
bool read_subset_bin2matrix(FILE* fp, Matrix &data, int start_row, 
			    int num_rows, int num_cols);
#endif
