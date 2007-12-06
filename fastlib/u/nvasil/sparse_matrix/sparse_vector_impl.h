/*
 * =====================================================================================
 * 
 *       Filename:  sparse_vector_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  12/03/2007 07:12:26 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

inline SparseVector::SparseVector(std::vector<index_t> &indices, 
		                              Vector &values,
																	index_t dimension) {
	Init(indices, values, dimension);
}

inline SparseVector::SparseVector(std::map<index_t, double> &data, 
		                              index_t dimension) {
  Init(data, dimension);
}

inline SparseVector::SparseVector(index_t estimated_non_zero_elements, 
		                              index_t dimension) {
  Init(estimated_non_zero_elements, dimension);
}

inline SparseVector::SparseVector(Epetra_CrsMatrix *one_dim_matrix, 
		                              index_t dimension) {
  Init(one_dim_matrix, dimension);
}

inline SparseVector::SparseVector(const SparseVector &other) {
  Copy(other);
}
inline void SparseVector::Init() {
  map_ =  new Epetra_Map(dimension_, 0, comm_);
	own_ =  true;
}
	
inline void SparseVector::Init(std::vector<index_t> &indices, 
		                    Vector &values, 
												index_t  dimension) {
	Init();
	if (unlikely((index_t)indices.size()!=values.length())) {
	  FATAL("Indices vector has %i elements while Values vectors has %i\n", 
				  (index_t)indices.size(), values.length());
	}
	std::vector<index_t>::iterator it = std::max_element(indices.begin(), indices.end()); 
	dimension_ = *it+1;
	if (dimension>0) {
	  if (dimension > dimension_) {
		  dimension_= dimension;
		} else {
		  FATAL("Error, the requested dimension was %i, while the maximum "
					  "index in the data is %i", dimension, dimension_);
		}
	}
	vector_ = new Epetra_CrsMatrix((Epetra_DataAccess)0, *map_, (index_t)indices.size());
  my_global_elements_ = map_->MyGlobalElements();
  vector_->InsertGlobalValues(*my_global_elements_, 
		                          values.length(), 
															values.ptr(), 
															&indices[0]);
	start_ = 0;
	end_   = dimension_-1;
}

void SparseVector::Init(std::vector<index_t> &indices, std::vector<double> &values, 
			      index_t dimension) {
	Init();
	if (unlikely(indices.size()!=values.size())) {
	  FATAL("Indices vector has %i elements while Values vectors has %i\n", 
				  (index_t)indices.size(), values.size());
	}
	std::vector<index_t>::iterator it = std::max_element(indices.begin(), indices.end()); 
	dimension_ = *it+1;
	if (dimension>0) {
	  if (dimension > dimension_) {
		  dimension_= dimension;
		} else {
		  FATAL("Error, the requested dimension was %i, while the maximum "
					  "index in the data is %i", dimension, dimension_);
		}
	}
	vector_ = new Epetra_CrsMatrix((Epetra_DataAccess)0, *map_, indices.size());
  my_global_elements_ = map_->MyGlobalElements();
  vector_->InsertGlobalValues(*my_global_elements_, 
		                          values.size(), 
															&values[0], 
															&indices[0]);
	start_ = 0;
	end_   = dimension_-1;
}
	
void SparseVector::Init(index_t *indices, double *values, index_t len, index_t dimension) {
  Init();
	vector_ = new Epetra_CrsMatrix((Epetra_DataAccess)0, *map_, len);
  my_global_elements_ = map_->MyGlobalElements();
  vector_->InsertGlobalValues(*my_global_elements_, 
		                          len, 
															values, 
															indices);
	start_ = 0;
	end_   = dimension_-1;

}


inline void SparseVector::Init(std::map<index_t, double> &data, index_t dimension) {
  Init();
  std::map<index_t, double>::iterator it=max_element(data.begin(), data.end());
	dimension_ = it->first+1;
	if (dimension>0) {
	  if (dimension > dimension_) {
		  dimension_= dimension;
		} else {
		  FATAL("Error, the requested dimension was %i, while the maximum "
					  "index in the data is %i", dimension, dimension_);
		}
	}
	vector_ = new Epetra_CrsMatrix((Epetra_DataAccess)0, *map_, data.size());
	index_t *indices = new index_t[data.size()];
	double  *values  = new double[data.size()];
	index_t i = 0;
	for(it=data.begin(); it!=data.end(); it++) {
	  indices[i] = it->first;
		values[i]  = it->second;
		i++;
	}
	my_global_elements_ = map_->MyGlobalElements();
  vector_->InsertGlobalValues(*my_global_elements_, 
		                          data.size(), 
															values, 
															indices);
  delete []indices;
	delete []values;
  start_ = 0;
	end_   = dimension_-1;

}

inline void SparseVector::Init(index_t estimated_non_zero_elements, index_t dimension) {
  Init(); 
  vector_ = new Epetra_CrsMatrix((Epetra_DataAccess)0, *map_, estimated_non_zero_elements);
	dimension_ = dimension;
	start_ = 0;
	end_   = dimension_-1;
}

inline void SparseVector::Init(Epetra_CrsMatrix *one_dim_matrix, index_t dimension) {
  Init();
	vector_ = one_dim_matrix;
	*map_  = one_dim_matrix->RowMap();
	dimension_ = dimension;
	start_ = 0;
	end_   = dimension_-1;
}

inline void SparseVector::Copy(const SparseVector &other) {
  Init();
	vector_ =  new Epetra_CrsMatrix(*other.vector_);
  dimension_ = other.dimension_;
	start_=other.start_;
	end_=other.end_;
}
inline void SparseVector::Destruct() {
  if (own_ == true) {
	  delete vector_;
	}
	delete map_;
}

inline void SparseVector::MakeSubvector(index_t start_index, index_t len, SparseVector* dest) {
  DEBUG_BOUNDS(start_ + start_index, end_+1);
	DEBUG_BOUNDS(start_ + start_index+len-1, end_+1);
	dest->Init(this->vector_, dimension_);
  dest->set_start(start_index + start_);
  dest->set_end(start_ + start_index + len-1);	
}
  
inline double SparseVector::get(index_t i) {
  DEBUG_BOUNDS(start_+i, end_+1);
	index_t pos = start_+i;
	double  *values;
	index_t *indices;
	index_t num_of_elements;
	vector_->ExtractGlobalRowView(*my_global_elements_, num_of_elements, values, indices);
  index_t *p = std::find(indices, indices+num_of_elements, pos);
	if (p!=indices+num_of_elements) {
	  return values[p-indices];
	} else {
	  return 0;
	}
}

inline void SparseVector::set(index_t i, double  value) {
  DEBUG_BOUNDS(start_+i, end_+1);
	index_t pos = start_+i;
	if (get(i)!=0) {
	  vector_->ReplaceGlobalValues(*my_global_elements_, 1, &value, &pos);
	} else {
	  vector_->InsertGlobalValues(*my_global_elements_, 1, &value, &pos);
	}
}

inline void SparseVector::Lock() {
  if (unlikely(vector_==NULL)) {
	  FATAL("Attempted to Lock but you haven't initialized the Epetra_CrsMatrix pointer\n");
	}
	vector_->FillComplete();
}

inline void SparseVector::set_start(index_t ind) {
  start_ = ind;
}

inline void SparseVector::set_end(index_t ind) {
  end_ = ind;
}

