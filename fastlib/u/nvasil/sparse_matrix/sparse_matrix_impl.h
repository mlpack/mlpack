/*
 * =====================================================================================
 * 
 *       Filename:  sparse_matrix_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  12/02/2007 10:18:02 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 *
 */

SparseMatrix::SparseMatrix() {

}

void SparseMatrix::Init(index_t num_of_rows, 
		                    index_t num_of_columns,
		                    index_t nnz_per_row) {
  num_of_rows_=num_of_rows;
	num_of_columns_=num_of_columns;
	if (num_of_rows_ < num_of_columns_) {
	  FATAL("Num of rows  %i is less than the number or columns %i",
				   num_of_rows_, num_of_columns_);
	}
	map_ = new Epetra_Map(num_of_rows_, 0, comm_);
	matrix_ = new Epetra_CrsMatrix((Epetra_DataAccess)0, *map_, nnz_per_row);
}

void SparseMatrix::Init(index_t num_of_rows,
	                      index_t num_of_columns,
												index_t *nnz_per_row) {
  num_of_rows_=num_of_rows;
	num_of_columns_=num_of_columns;
	if (num_of_rows_ < num_of_columns_) {
	  FATAL("Num of rows  %i is less than the number or columns %i",
				   num_of_rows_, num_of_columns_);
	}
	map_ = new Epetra_Map(num_of_rows_, 0, comm_);
	matrix_ = new Epetra_CrsMatrix((Epetra_DataAccess)0 , *map_, nnz_per_row);
}


void SparseMatrix::Init(const std::vector<index_t> &rows,
	                      const std::vector<index_t> &columns,	
		                    const Vector &values, 
			                  index_t nnz_per_row, 
			                  index_t dimension) {
  if (nnz_per_row > 0 && dimension > 0) {
	  Init(dimension, dimension, nnz_per_row);
  } else {
		num_of_rows_ = 0;
		num_of_columns_=0;
		std::map<index_t, index_t> frequencies;
		for(index_t i=0; i<(index_t)rows.size(); i++) {
			frequencies[rows[i]]++;
			frequencies[columns[i]]++;
		  if (rows[i]>num_of_rows_) {
			  num_of_rows_ = rows[i];
			}
			if (columns[i]>num_of_columns_) {
			  num_of_columns_ = columns[i];
			}
		}
    if ((index_t)frequencies.size()-1!=num_of_rows_) {
		  NONFATAL("Some of the rows are zeros only!");
		}
		index_t *nnz= new index_t[dimension_];
		for(index_t i=0; i<num_of_rows_; i++) {
		  nnz[i]=frequencies[i];
		}
	  Init(num_of_rows_, num_of_columns_, nnz);	
		delete []nnz;
		Load(rows, columns, values);
 	}
}

void SparseMatrix::Init(const std::vector<index_t> &rows,
                        const std::vector<index_t> &columns,	
                        const std::vector<double>  &values, 
                        index_t nnz_per_row, 
                        index_t dimension) {
	Vector temp;
	temp.Alias((double *)&values[0], values.size());
	Init(rows, columns, temp, nnz_per_row, dimension);

}
void SparseMatrix::StartLoadingRows() {
   my_global_elements_ = map_->MyGlobalElements();
}

void SparseMatrix::LoadRow(index_t row, 
		                       std::vector<index_t> &columns, 
		                       Vector &values) {
	DEBUG_ASSERT(values.length() == (index_t)columns.size());
  matrix_->InsertGlobalValues(my_global_elements_[row], 
		                          values.length(), 
															values.ptr(), 
															&columns[0]);

}

void SparseMatrix::LoadRow(index_t row, 
		                       index_t *columns,
		                       Vector &values) {
  matrix_->InsertGlobalValues(my_global_elements_[row], 
		                          values.length(), 
															values.ptr(), 
															&columns[0]);

}

void SparseMatrix::LoadRow(index_t row, 
		                       std::vector<index_t> &columns,
		                       std::vector<double>  &values) {
  matrix_->InsertGlobalValues(my_global_elements_[row], 
		                          values.size(), 
															&values[0], 
															&columns[0]);

}

void SparseMatrix::LoadRow(index_t row,
	                         index_t num,	
		                       index_t *columns,
		                       double  *values) {
	matrix_->InsertGlobalValues(my_global_elements_[row], 
		                          num, 
															values, 
															columns);

}

void SparseMatrix::EndLoading() {
  matrix_->FillComplete();
}

void SparseMatrix::Load(const std::vector<index_t> &rows, 
		                    const std::vector<index_t> &columns, 
												const Vector &values) {
  my_global_elements_ = map_->MyGlobalElements();
	index_t i=0;
	index_t cur_row = rows[i];
  index_t prev_row= rows[i];
	std::vector<index_t> indices;
	std::vector<double> row_values;	
	while (true) {
	  while (rows[cur_row]==rows[prev_row]) {
		  indices.push_back(columns[i]);
			row_values.push_back(values[i]);
			i++;
		  prev_row=i-1;
		  cur_row=i;	
    }
    matrix_->InsertGlobalValues(my_global_elements_[cur_row], 
				                        values.length(), 
																&row_values[0], 
																&indices[0]);
		if (i==dimension_) {
		  break;
		}
	}
}

double SparseMatrix::get(index_t r, index_t c) {
  DEBUG_BOUNDS(r, num_of_rows_);
	DEBUG_BOUNDS(c, num_of_columns_);
	index_t global_row = my_global_elements_[r];
	index_t num_of_entries;
	double *values;
	index_t *indices;
  matrix_->ExtractGlobalRowView(global_row, num_of_entries, values, indices);
	index_t *pos = std::find(indices, indices+num_of_entries, c);
	if (pos==indices+num_of_entries) {
	  return 0;
	}
  return values[*pos];
}

void SparseMatrix::set(index_t r, index_t c, double v) {
  DEBUG_BOUNDS(r, num_of_rows_);
	DEBUG_BOUNDS(c, num_of_columns_);
	if (get(r,c)!=0) {
	  matrix_->InsertGlobalValues(my_global_elements_[r], 1, &v, &c);
	} else {
    matrix_->ReplaceGlobalValues(my_global_elements_[r], 1, &v, &c); 
	}
}
void SparseMatrix::MakeSymmetric() {
	index_t num_of_entries;
	double  *values;
	index_t *indices;
  for(index_t i=0; i<dimension_; i++) {
    index_t global_row = my_global_elements_[i];
    matrix_->ExtractGlobalRowView(global_row, num_of_entries, values, indices);
    for(index_t j=0; j<num_of_entries; j++) {
		  if (unlikely(get(i, indices[j])!=values[j])) {
			  set(i, indices[j], values[j]);
			}
		}
  } 
}

