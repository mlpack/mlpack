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
  map_    = NULL;
	issymmetric_=false;
}

SparseMatrix::SparseMatrix(const index_t num_of_rows,
                           const index_t num_of_columns,
  												 const index_t nnz_per_row) {
  Init(num_of_rows, num_of_columns, nnz_per_row);
}
void SparseMatrix::Init(const index_t num_of_rows, 
		                    const index_t num_of_columns,
		                    const index_t nnz_per_row) {
  issymmetric_ = false;
	if likely(num_of_rows < num_of_columns) {
	  FATAL("Num of rows %i should be greater than the num of columns %i\n",
				  num_of_rows, num_of_columns);
	}
  num_of_rows_=num_of_rows;
	num_of_columns_=num_of_columns;
	dimension_ = num_of_rows_;
	if (num_of_rows_ < num_of_columns_) {
	  FATAL("Num of rows  %i is less than the number or columns %i",
				   num_of_rows_, num_of_columns_);
	}
	map_ = new Epetra_Map(num_of_rows_, 0, comm_);
	matrix_ = Teuchos::rcp(
			         new Epetra_CrsMatrix((Epetra_DataAccess)0, *map_, nnz_per_row));
	StartLoadingRows();
}

void SparseMatrix::Init(index_t num_of_rows,
	                      index_t num_of_columns,
												index_t *nnz_per_row) {
  issymmetric_ = false;
	num_of_rows_=num_of_rows;
	num_of_columns_=num_of_columns;
	dimension_ = num_of_rows;
	if (num_of_rows_ < num_of_columns_) {
	  FATAL("Num of rows  %i is less than the number or columns %i",
				   num_of_rows_, num_of_columns_);
	}
	map_ = new Epetra_Map(num_of_rows_, 0, comm_);
	matrix_ = Teuchos::rcp(
               new Epetra_CrsMatrix((Epetra_DataAccess)0 , *map_, nnz_per_row));
	StartLoadingRows();
}


void SparseMatrix::Init(const std::vector<index_t> &rows,
	                      const std::vector<index_t> &columns,	
		                    const Vector &values, 
			                  index_t nnz_per_row, 
			                  index_t dimension) {
  issymmetric_ = false;
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
		num_of_columns_++;
		num_of_rows_++;
    if ((index_t)frequencies.size()!=num_of_rows_) {
		  NONFATAL("Some of the rows are zeros only!");
		}
		index_t *nnz= new index_t[dimension_];
		for(index_t i=0; i<num_of_rows_; i++) {
		  nnz[i]=frequencies[i];
		}
	  Init(num_of_rows_, num_of_columns_, nnz);	
		//delete []nnz;
 	}
	Load(rows, columns, values);
}

void SparseMatrix::Init(const std::vector<index_t> &rows,
                        const std::vector<index_t> &columns,	
                        const std::vector<double>  &values, 
                        index_t nnz_per_row, 
                        index_t dimension) {
	issymmetric_ = false;
	Vector temp;
	temp.Alias((double *)&values[0], values.size());
	Init(rows, columns, temp, nnz_per_row, dimension);

}

void SparseMatrix::Init(std::string filename) {
  issymmetric_ = false;
	FILE *fp = fopen(filename.c_str(), "r");
	if (fp == NULL) {
	  FATAL("Cannot open %s, error: %s", 
				  filename.c_str(), 
					strerror(errno));
	}
	std::vector<index_t> rows;
	std::vector<index_t> cols;
	std::vector<double>  vals;
	while (!feof(fp)) {
		index_t r, c;
		double v;
		fscanf(fp,"%i %i %lg\n", &r, &c, &v);
		rows.push_back(r);
		cols.push_back(c);
		vals.push_back(v);
	}
	fclose(fp);	
	/*for(index_t i=0; i< (index_t)rows.size(); i++) {
	  printf("%i %i %lg\n", rows[i], cols[i], vals[i]);
	}*/
	Init(rows, cols, vals, -1, -1);
}

void SparseMatrix::Destruct() {
  if (map_!=NULL) {
	  delete map_;
	}
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
	DEBUG_ASSERT(rows.size() ==columns.size());
	DEBUG_ASSERT((index_t)columns.size() == values.length());
  my_global_elements_ = map_->MyGlobalElements();
	index_t i=0;
	index_t cur_row = rows[i];
  index_t prev_row= rows[i];
	std::vector<index_t> indices;
	std::vector<double> row_values;	
	while (true) {
		indices.clear();
		row_values.clear();
	  while (likely((rows[cur_row]==rows[prev_row]) && 
					        (i < (index_t)rows.size()))) {
		  indices.push_back(columns[i]);
			row_values.push_back(values[i]);
			i++;
		  prev_row=i-1;
		  cur_row=i;	
    }
    matrix_->InsertGlobalValues(my_global_elements_[rows[prev_row]], 
				                        row_values.size(), 
																&row_values[0], 
																&indices[0]);
		prev_row=cur_row;
    if (i >= (index_t)rows.size()) {
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
  return values[(ptrdiff_t)(pos-indices)];
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
 issymmetric_ = true;	
}

void SparseMatrix::Eig(index_t num_of_eigvalues,
	                     std::string eigtype,	
		                   Matrix *eigvectors, 
											 std::vector<double> *real_eigvalues,
											 std::vector<double> *imag_eigvalues) {
  index_t block_size = 4;
	Teuchos::RCP<Epetra_MultiVector> ivec =  Teuchos::rcp(new Epetra_MultiVector(*map_, 
			                                                      block_size,
																										        num_of_eigvalues));
  // Fill it with random numbers
  ivec->Random();

  // Setup the eigenproblem, with the matrix A and the initial vectors ivec
	Teuchos::RCP<Anasazi::BasicEigenproblem<double,MV,OP>  >problem = 
      Teuchos::rcp(new Anasazi::BasicEigenproblem<double,MV,OP>(matrix_, ivec));

  // The 2-D laplacian is symmetric. Specify this in the eigenproblem.
  if (issymmetric_ == true) {
	  problem->setHermitian(true);
	}

  // Specify the desired number of eigenvalues
  problem->setNEV(num_of_eigvalues);

  // Signal that we are done setting up the eigenvalue problem
  bool ierr = problem->setProblem();

  // Check the return from setProblem(). If this is true, there was an
  // error. This probably means we did not specify enough information for
  // the eigenproblem.
  if unlikely(ierr == true) {
	  FATAL("Trilinos solver error, you probably didn't specify enough information"
				  "for the eigenvalue problem\n");
	}

  // Specify the verbosity level. Options include:
  // Anasazi::Errors 
  //   This option is always set
  // Anasazi::Warnings 
  //   Warnings (less severe than errors)
  // Anasazi::IterationDetails 
  //   Details at each iteration, such as the current eigenvalues
  // Anasazi::OrthoDetails 
  //   Details about orthogonality
  // Anasazi::TimingDetails
  //   A summary of the timing info for the solve() routine
  // Anasazi::FinalSummary 
  //   A final summary 
  // Anasazi::Debug 
  //   Debugging information
  int verbosity = Anasazi::Warnings     + 
		              Anasazi::Errors       + 
									Anasazi::FinalSummary + 
									Anasazi::TimingDetails;

  // Choose which eigenvalues to compute
  // Choices are:
  // LM - target the largest magnitude  [default]
  // SM - target the smallest magnitude 
  // LR - target the largest real 
  // SR - target the smallest real 
  // LI - target the largest imaginary
  // SI - target the smallest imaginary

  // Create the parameter list for the eigensolver
  Teuchos::ParameterList my_pl;
  my_pl.set( "Verbosity", verbosity);
  my_pl.set( "Which", eigtype);
  my_pl.set( "Block Size", block_size);
  my_pl.set( "Num Blocks", 20);
  my_pl.set( "Maximum Restarts", 100);
  my_pl.set( "Convergence Tolerance", 1.0e-8);

  // Create the Block Krylov Schur solver
  // This takes as inputs the eigenvalue problem and the solver parameters
  Anasazi::BlockKrylovSchurSolMgr<double,MV,OP> 
    my_block_krylov_schur(problem, my_pl);

  // Solve the eigenvalue problem, and save the return code
  Anasazi::ReturnType solver_return = my_block_krylov_schur.solve();

  // Check return code of the solver: Unconverged, Failed, or OK
  switch (solver_return) {
    // UNCONVERGED
    case Anasazi::Unconverged: 
      NONFATAL("Anasazi::BlockKrylovSchur::solve() did not converge!\n");  
			return ;    
    // CONVERGED
    case Anasazi::Converged:
      NONFATAL("Anasazi::BlockKrylovSchur::solve() converged!\n");
  }
  // Get eigensolution struct
  Anasazi::Eigensolution<double, Epetra_MultiVector> sol = problem->getSolution();
  // Get the number of eigenpairs returned
  int num_of_eigvals_returned = sol.numVecs;
	NONFATAL("The solver returned less eigenvalues (%i) "
			     "than requested (%i)\n", num_of_eigvalues,
		       num_of_eigvals_returned);

  // Get eigenvectors
	eigvectors->Init(dimension_, num_of_eigvals_returned);
	Teuchos::RCP<Epetra_MultiVector> evecs = sol.Evecs;
	evecs->ExtractCopy(eigvectors->GetColumnPtr(0), dimension_);

  // Get eigenvalues
  std::vector<Anasazi::Value<double> > evals = sol.Evals;
  real_eigvalues->resize(num_of_eigvals_returned);
	for(index_t i=0; i<num_of_eigvals_returned; i++) {
	  real_eigvalues->assign(i, evals[i].realpart);
	  if (issymmetric_ == false) {
      imag_eigvalues->resize(num_of_eigvals_returned);
		  imag_eigvalues->assign(i, evals[i].imagpart);
	  }
	}

  // Test residuals
  // Generate a (numev x numev) dense matrix for the eigenvalues...
  // This matrix is automatically initialized to zero
  Teuchos::SerialDenseMatrix<int, double> d(num_of_eigvalues, 
			                                      num_of_eigvalues);

  // Add the eigenvalues on the diagonals (only the real part since problem is Hermitian)
  for (int i=0; i<num_of_eigvalues; i++) {
    d(i,i) = evals[i].realpart;
  }

  // Generate a multivector for the product of the matrix and the eigenvectors
  Epetra_MultiVector res(*map_, num_of_eigvalues); 

  // R = A*evecs
  matrix_->Apply( *evecs, res);

  // R -= evecs*D 
  //    = A*evecs - evecs*D
  MVT::MvTimesMatAddMv( -1.0, *evecs, d, 1.0, res);

  // Compute the 2-norm of each vector in the MultiVector
  // and store them to a std::vector<double>
  std::vector<double> norm_res(num_of_eigvalues);
  MVT::MvNorm(res, &norm_res);
}

