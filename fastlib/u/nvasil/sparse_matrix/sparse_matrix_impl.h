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

void SparseMatrix::Init(index_t dimension, index_t nnz_per_row) {
  dimension_ = dimension;
	map_ = new Epetra_Map(dimension_, 0, comm_);
	matrix_ = new Epetra_CrsMatrix(Copy , map_, nnz_per_row);
}

void SparseMatrix::Init(index_t dimension, index_t *nnz_per_row) {
  dimension_ = dimension;
	map_ = new Epetra_Map(dimension_, 0, comm_);
	matrix_ = new Epetra_CrsMatrix(Copy , map_, nnz_per_row);
}


void SparseMatrix::Init(const std::vector<index_t> &rows,
	                      const std::vector<index_t> &columns,	
		                    const Vector &values, 
			                  index_t nnz_per_row, 
			                  index_t dimension) {
  if (nnz_per_row > 0 && dimension > 0) {
	  Init(dimension, nnz_per_row);
  } else {
		dimension_ = 0;
		map<index_t, index_t> frequencies;
		for(index_t i=0; i<rows.length(); i++) {
			frequencies[rows[i]]++;
			frequencies[columns[i]]++;
		  if (rows[i]>dimension_) {
			  dimension_ = rows[i];
			}
		}
    dimension_=frequencies.size();
    index_t *nnz= new index_t[dimension_];
		for(index_t i=0; i<dimension_; i++) {
		  nnz[i]=frequency[i];
		}
	  Init(dimension, nnz);	
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
	
