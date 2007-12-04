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
	matrix_ = new Epetra_CrsMatrix(Copy , map_, nnz_per_row);
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
	matrix_ = new Epetra_CrsMatrix(Copy , map_, nnz_per_row);
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
		map<index_t, index_t> frequencies;
		for(index_t i=0; i<rows.length(); i++) {
			frequencies[rows[i]]++;
			frequencies[columns[i]]++;
		  if (rows[i]>num_of_rows_) {
			  num_of_rows_ = rows[i];
			}
			if (columns[i]>num_of_columns_) {
			  num_of_columns_ = columns[i];
			}
		}
    if (frequency.size()-1!=num_of_rows_) {
		  NONFATAL("Some of the rows are zeros only!");
		}
		index_t *nnz= new index_t[dimension_];
		for(index_t i=0; i<num_of_rows__; i++) {
		  nnz[i]=frequency[i];
		}
	  Init(num_of_rows_, num_of_columns_, nnz);	
		delete []nnz;
		Load(rows, columns, values);
 	}
}

void SparseMatrix::StartLoadingRows() {
   MyGlobalElements_ = matrix_->MyGlobalElements();
}

void SparseMatrix::LoadRow(index_t row, 
		                       std::vector<index_t> &columns, 
		                       Vector &values) {
  matrix_->InsertGlobalValues(MyGlobalElements_[row], 
		                          values.size(), 
															&row_values, 
															&columns);

}

void SparseMatrix::EndLoading() {
  matrix_->FillComplete();
	matrix_->MakeDataContiguous();
}

void SparseMatrix::Load(Vector &rows, 
		                    std::vector<index_t> &columns, 
												Vector &values) {
  MyGlobalElements_ = matrix_->MyGlobalElements();
	index_t i=0;
	index_t cur_row = rows[i];
  index_t prev_row= rows[i];
	vector<index_t> indices;
  vector<index_t> row_values;	
	while (true) {
	  while (rows[curr_row]==rows[prev_row]) {
		  indices.push_back(colums[i]);
			row_values.push_back(values[i]);
			i++;
		  prev_row=i-1;
		  curr_row=i;	
    }
    matrix_->InsertGlobalValues(MyGlobalElements_[curr_row], 
				                        values.size(), 
																&row_values, 
																&indices);
		if (i==dimension_) {
		  break;
		}
	}
}

double SparseMatrix::get(ndex_t r, index_t c) {
  DEBUG_BOUNDS(r, num_of_rows_);
	DEBUG_BOUNDS(c, num_of_columns_);
	index_t global_row = MyGlobalElements_[r];
	index_t num_of_entries;
	double *values;
	index_t *indices;
  matrix_->ExtractctGlobalRowView(global_row, num_of_entires, values, indices);
	index_t *pos = std::find(indices, indices+num_of_entries, c);
	if (*pos==indices+num_of_entries) {
	  return 0;
	}
  return values[*pos];
}

void SparseMatrix::set(index_t r, index_t c, double v) {
  DEBUG_BOUNDS(r, num_of_rows_);
	DEBUG_BOUNDS(c, num_of_columns_);
	if (get(r,c)!=0) {
	  matrix_->InsertGlobalValues(MyGlobalElements_[r], 1, &v, &c);
	} else {
    matrix_->ReplaceGlobalValues(MyGlobalElements_[r], 1, &v, &c); 
	}
}
void SparseMatrix::MakeSymmetric() {
	index_t num_of_entries;
	double  *values;
	index_t *indices;
  for(index_t i=0; i<dimension_; i++) {
    index_t global_row = MyGlobalElements_[i];
    matrix_->ExtractctGlobalRowView(global_row, num_of_entries, values, indices);
    for(index_t j=0; j<num_of_entries; j++) {
		  if (unlikely(get(i, indices[j])!=values[j])) {
			  set(i, indices[j], values[j]);
			}
		}
  } 
}

