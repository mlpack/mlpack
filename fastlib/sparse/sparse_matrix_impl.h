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

SparseMatrix::SparseMatrix(std::string textfile) {
	  Init(textfile);
}

SparseMatrix::~SparseMatrix()  {
	  Destruct();
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

void SparseMatrix::Init(const Epetra_CrsMatrix &other) {
  issymmetric_ = false;
	matrix_ = Teuchos::rcp(new Epetra_CrsMatrix(other));
	map_ = new Epetra_Map(matrix_->RowMap());
	dimension_ = other.NumGlobalRows();
	num_of_rows_=dimension_;
	num_of_columns_=dimension_;
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
		if (num_of_rows_ < num_of_columns_) {
		  FATAL("At this point we only support rows (%i) >= columns (%i)\n", 
					  num_of_rows_, num_of_columns_);
		}
		dimension_ = num_of_rows_;
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

void SparseMatrix::Copy(const SparseMatrix &other) {
  num_of_rows_ = other.num_of_rows_;
	num_of_columns_ = other.num_of_columns_;
	dimension_   = other.dimension_;
	map_ = new Epetra_Map(num_of_rows_, 0, comm_);
	issymmetric_ = other.issymmetric_;
	if (other.matrix_->Filled()==false) {
	  matrix_ = Teuchos::rcp(new Epetra_CrsMatrix(Epetra_DataAccess(0), *map_, 10));
		this->StartLoadingRows();
		for(index_t r=0; r<num_of_rows_; r++){
		 index_t global_row = other.my_global_elements_[r];
	   index_t num_of_entries;
	   double *values;
	   index_t *indices;
     other.matrix_->ExtractGlobalRowView(global_row, num_of_entries, values, indices);
		 this->LoadRow(r, num_of_entries, indices, values);
		}
  } else {
	  matrix_ = Teuchos::rcp(new Epetra_CrsMatrix(*(other.matrix_.get())));
	}
}

void SparseMatrix::Destruct() {
  if (map_!=NULL) {
	  delete map_;
		map_=NULL;
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
	matrix_->OptimizeStorage();
}

void SparseMatrix::MakeSymmetric() {
	index_t num_of_entries;
	double  *values;
	index_t *indices;
	my_global_elements_ = map_->MyGlobalElements();
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

void SparseMatrix::SetDiagonal(const Vector &vector) { 
  if (unlikely(vector.length()!=dimension_)) {
	  FATAL("Vector should have the same dimension with the matrix!\n");
	}
	for(index_t i=0; i<dimension_; i++) {
	  set(i,i, vector[i]);
	}
} 

void SparseMatrix::SetDiagonal(const double scalar) { 
	for(index_t i=0; i<dimension_; i++) {
	  set(i,i, scalar);
	}
} 

double SparseMatrix::get(index_t r, index_t c) const {
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


void SparseMatrix::Negate(){
    double *values;
	  index_t *indices;
	  index_t num_of_entries;
    my_global_elements_ = map_->MyGlobalElements();
	  for(index_t i=0; i<num_of_rows_; i++) {
	    index_t global_row = my_global_elements_[i];
      matrix_->ExtractGlobalRowView(global_row, num_of_entries, values, indices);
      for(index_t j=0; j<num_of_entries; j++) {
	      values[j]=-values[j];
	    }
    }
  }

template<typename FUNC>
void SparseMatrix::ApplyFunction(FUNC &function){
    double *values;
	  index_t *indices;
	  index_t num_of_entries;
    my_global_elements_ = map_->MyGlobalElements();
	  for(index_t i=0; i<num_of_rows_; i++) {
	    index_t global_row = my_global_elements_[i];
      matrix_->ExtractGlobalRowView(global_row, num_of_entries, values, indices);
      for(index_t j=0; j<num_of_entries; j++) {
	      values[j]=function(values[j]);
	    }
    }
  }

void SparseMatrix::Eig(index_t num_of_eigvalues,
	                     std::string eigtype,	
		                   Matrix *eigvectors, 
											 std::vector<double> *real_eigvalues,
											 std::vector<double> *imag_eigvalues) {
  if (unlikely(!matrix_->Filled())) {
	  FATAL("You have to call EndLoading before running eigenvalues otherwise "
				  "it will fail\n");
	}
  
	index_t block_size=10;
  retry:	
	Teuchos::RCP<Epetra_MultiVector> ivec =  Teuchos::rcp(new 
																						Epetra_MultiVector(*map_, 
			                                                         block_size));
  // Fill it with random numbers
  ivec->Random();
  // Setup the eigenproblem, with the matrix A and the initial vectors ivec
	Teuchos::RCP<Anasazi::BasicEigenproblem<double,MV,OP>  >problem = 
      Teuchos::rcp(new Anasazi::BasicEigenproblem<double,MV,OP>(matrix_, ivec));

  // The 2-D laplacian is symmetric. Specify this in the eigenproblem.
  if (issymmetric_ == true) {
	  problem->setHermitian(true);
	} else {
	  problem->setHermitian(false);
	}

  // Specify the desired number of eigenvalues
  problem->setNEV(num_of_eigvalues);

  // Signal that we are done setting up the eigenvalue problem
  bool ierr = problem->setProblem();

  // Check the return from setProblem(). If this is true, there was an
  // error. This probably means we did not specify enough information for
  // the eigenproblem.
  if unlikely(ierr == false) {
	  FATAL("Trilinos solver error, you probably didn't specify enough information "
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
								//	Anasazi::IterationDetails +
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
  double tolerance=1.0e-7;
  
	Teuchos::ParameterList my_pl;
  my_pl.set( "Verbosity", verbosity);
  my_pl.set( "Which", eigtype);
  my_pl.set( "Block Size", block_size);
  my_pl.set( "Num Blocks", 20);
  my_pl.set( "Maximum Restarts", 2);
  my_pl.set( "Convergence Tolerance", tolerance);

  // Create the Block Krylov Schur solver
  // This takes as inputs the eigenvalue problem and the solver parameters
	Teuchos::RCP<Anasazi::BlockKrylovSchurSolMgr<double,MV,OP> >  my_block_krylov_schur;
	my_block_krylov_schur=Teuchos::rcp(new 
			Anasazi::BlockKrylovSchurSolMgr<double,MV,OP> (problem, my_pl));

  // Solve the eigenvalue problem, and save the return code
  Anasazi::ReturnType solver_return = my_block_krylov_schur->solve();

  // Check return code of the solver: Unconverged, Failed, or OK
  switch (solver_return) {
    // UNCONVERGED
    case Anasazi::Unconverged:
			block_size*=2;
		  if (block_size<64) {
				tolerance=tolerance*10;
			  NONFATAL("Didn't converge, increasing block_size to %i\n and retrying ", 
					    	 (int)block_size);
				goto retry;
			}	
      FATAL("Anasazi::BlockKrylovSchur::solve() did not converge!\n");  
			return ;    
    // CONVERGED
    case Anasazi::Converged:
      NONFATAL("Anasazi::BlockKrylovSchur::solve() converged!\n");
  }
  // Get eigensolution struct
  Anasazi::Eigensolution<double, Epetra_MultiVector> sol = problem->getSolution();
  // Get the number of eigenpairs returned
  int num_of_eigvals_returned = sol.numVecs;
	if (num_of_eigvals_returned < num_of_eigvalues) {
	  NONFATAL("The solver returned less eigenvalues (%i) "
		  	     "than requested (%i)\n", num_of_eigvalues,
		         num_of_eigvals_returned);
	}

  // Get eigenvectors
	eigvectors->Init(dimension_, num_of_eigvals_returned);
	Teuchos::RCP<Epetra_MultiVector> evecs = sol.Evecs;
	evecs->ExtractCopy(eigvectors->GetColumnPtr(0), dimension_);

  // Get eigenvalues
  std::vector<Anasazi::Value<double> > evals = sol.Evals;
  real_eigvalues->resize(0);
	for(index_t i=0; i<num_of_eigvals_returned; i++) {
	  real_eigvalues->push_back(evals[i].realpart);
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

void SparseMatrix::LinSolve(Vector &b, // must be initialized (space allocated)
			                      Vector *x, // must be initialized (space allocated)
								            double tolerance,
								            index_t iterations) {

	if (matrix_->Filled()==false && matrix_->StorageOptimized()==false){
	  FATAL("You should call EndLoading() first\n");
	}
	Epetra_Vector tempb(View, *map_, b.ptr());
  Epetra_Vector tempx(View, *map_, x->ptr());
  // create linear problem
 	Epetra_LinearProblem problem(matrix_.get(), &tempx, &tempb);
 	// create the AztecOO instance
	AztecOO solver(problem);
	solver.SetAztecOption( AZ_precond, AZ_Jacobi);
	solver.Iterate(iterations, tolerance);
	NONFATAL("Solver performed %i iterations, true residual %lg",
			     solver.NumIters(), solver.TrueResidual());
}


void SparseMatrix::IncompleteCholesky(index_t level_fill,
		                                  double drop_tol,
																			SparseMatrix *u, 
																			Vector       *d,
																			double *condest) {
  Ifpack_CrsIct *ict=NULL;
  ict = new Ifpack_CrsIct(*matrix_, drop_tol, level_fill);
  // Init values from A
  ict->InitValues(*matrix_);
  // compute the factors
  if (unlikely(ict->Factor()<0)) {
		NONFATAL("Cholesky factorization failed!\n");
	};
  // and now estimate the condition number
  ict->Condest(false, *condest);
	u->Init(ict->U());
	ict->D().ExtractCopy(d->ptr());
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

inline void Sparsem::Add(const SparseMatrix &a, 
			                   const SparseMatrix &b, 
												 SparseMatrix *result) {
  DEBUG_ASSERT(a.num_of_rows_==b.num_of_rows_);
	DEBUG_ASSERT(a.num_of_columns_==b.num_of_columns_);
  DEBUG_ASSERT(a.num_of_rows_==result->num_of_rows_);
	DEBUG_ASSERT(a.num_of_columns_==result->num_of_columns_);
	result->StartLoadingRows();
  for(index_t r=0; r<a.num_of_rows_; r++) {
	  index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
		a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r], num1, values1, indices1);
  	b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r], num2, values2, indices2);
		std::vector<double>  values3;
		std::vector<index_t> indices3;
		index_t i=0;
		index_t j=0;
		while (likely(i<num1 && j<num2)) {
		  while (indices1[i] < indices2[j]) {
			  values3.push_back(values1[i]);
				indices3.push_back(indices1[i]);
			  i++;	
        if unlikely((i>=num1)) {
				  break;
				}
		  }
			if ( likely(i<num1) && indices1[i] == indices2[j]) {
			  values3.push_back(values1[i] + values2[j]);
			  indices3.push_back(indices1[i]);
			} else {
			  values3.push_back(values2[j]);
			  indices3.push_back(indices2[j]);	
			}
			j++;
		}
		if (i<num1) {
		  values3.insert(values3.end(), values1+i, values1+num1);
		  indices3.insert(indices3.end(), indices1+i, indices1+num1);
		}
		if (j<num2) {
		  values3.insert(values3.end(), values2+j, values2+num2);
		  indices3.insert(indices3.end(), indices2+j, indices2+num2);
		}
    result->LoadRow(r, indices3, values3);
  }
}

inline void Sparsem::Subtract(const SparseMatrix &a,
                              const SparseMatrix &b,
										          SparseMatrix *result) {
  DEBUG_ASSERT(a.num_of_rows_==b.num_of_rows_);
	DEBUG_ASSERT(a.num_of_columns_==b.num_of_columns_);
	DEBUG_ASSERT(a.num_of_rows_==result->num_of_rows_);
	DEBUG_ASSERT(a.num_of_columns_==result->num_of_columns_);
	// If you try assigning the results to an already initialized matrix
	// you might get unexpected results. The following assertions
	// prevent you partially from that 
	DEBUG_ASSERT(&a != result); 
	DEBUG_ASSERT(&b != result);
	result->StartLoadingRows();
  for(index_t r=0; r<a.num_of_rows_; r++) {
	  index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
		a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r], num1, values1, indices1);
  	b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r], num2, values2, indices2);
		std::vector<double>  values3;
		std::vector<index_t> indices3;
		index_t i=0;
		index_t j=0;
		while (likely(i<num1 && j<num2)) {
		  while (indices1[i] < indices2[j]) {
			  values3.push_back(values1[i]);
				indices3.push_back(indices1[i]);
			  i++;	
        if unlikely((i>=num1)) {
				  break;
				}
		  }
			if (likely(i<num1) && indices1[i] == indices2[j]) {
			  double diff=values1[i] - values2[j];
				if (diff!=0) {
			    values3.push_back(diff);
				  indices3.push_back(indices1[i]);
			  }
		  } else {
			  values3.push_back(-values2[j]);
			  indices3.push_back(indices2[j]);	
			}
			j++;
		}
		if (i<num1) {
		  values3.insert(values3.end(), values1+i, values1+num1);
			indices3.insert(indices3.end(), indices1+i, indices1+num1);
		}
		if (j<num2) {
		  for(index_t k=j; k<num2; k++) {
		    values3.push_back(-values2[k]);
			  indices3.push_back(indices2[k]);
			}
	  }
    result->LoadRow(r, indices3, values3);
	}
}

inline void Sparsem::Multiply(const SparseMatrix &a,
                              const SparseMatrix &b,
										          SparseMatrix *result) {
  DEBUG_ASSERT(a.num_of_columns_ == b.num_of_rows_);
  DEBUG_ASSERT(a.num_of_rows_ == result->num_of_rows_);
  DEBUG_ASSERT(b.num_of_columns_ == result->num_of_columns_);
  // If you try assigning the results to an already initialized matrix
	// you might get unexpected results. The following assertions
	// prevent you partially from that 
  DEBUG_ASSERT(&a != result); 
	DEBUG_ASSERT(&b != result);

	if (b.issymmetric_ == true) {
    for(index_t r1=0; r1<a.num_of_rows_; r1++) {
			std::vector<index_t> indices3;
			std::vector<double>  values3;
			index_t num1;
			double  *values1;
			index_t *indices1;
      a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r1],
					                            num1, values1, indices1);
	    for(index_t r2=0; r2<b.num_of_rows_; r2++) {
		    index_t num2;
	      double  *values2;
				index_t *indices2;
	     	b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r2], 
						                            num2, values2, indices2);
	      index_t i=0;
	      index_t j=0;
	      double dot_product=0;
	      while (likely(i<num1 && j<num2)) {
		      while (indices1[i] < indices2[j]) {
		        i++;	
            if unlikely((i>=num1)) {
			        break;
			      }
	        }
		      if (likely(i<num1) && indices1[i] == indices2[j]) {
		        dot_product += values1[i] * values2[j];
		      } 
		      j++;
	      }
				if (dot_product!=0) {
				  indices3.push_back(r2);
					values3.push_back(dot_product);
				}
			}
			result->LoadRow(r1, indices3, values3);
		  indices3.clear();
			values3.clear();
	  }
  } else {
    for(index_t r1=0; r1<a.num_of_rows_; r1++) {
      index_t num1;
			double  *values1;
			index_t *indices1;
      a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r1],
					                            num1, values1, indices1);
			double dot_product=0;
			std::vector<index_t> indices3;
			std::vector<double>  values3;
			for(index_t r2=0; r2<b.num_of_columns_; r2++) {
				for(index_t k=0; k< num1; k++) {
			    dot_product += values1[k]*b.get(indices1[k], r2);
				}
				if (dot_product!=0){
				  indices3.push_back(r2);
					values3.push_back(dot_product);
				}
				dot_product=0;
			}
			if (!indices3.empty()) {
			  result->LoadRow(r1, indices3, values3);
			}
      indices3.clear();
			values3.clear();
		}
	}
}


inline void Sparsem::MultiplyT(SparseMatrix &a,
  											       SparseMatrix *result) {
	bool flag=a.issymmetric_;
	a.issymmetric_=true;
	Multiply(a, a, result);
	a.issymmetric_=flag;
	result->issymmetric_=true;
}

inline void Sparsem::Multiply(const SparseMatrix &mat,
                              const Vector &vec,
 										          Vector *result,
														  bool transpose_flag) {
  Epetra_Vector temp_in(View, *(mat.map_), (double *)vec.ptr());
	Epetra_Vector temp_out(View, *(mat.map_), (double *)result->ptr());
	mat.matrix_->Multiply(transpose_flag, temp_in, temp_out);
}

inline void Sparsem::Multiply(const SparseMatrix &mat,
                              const double scalar,
 										          SparseMatrix *result) {
  result->Copy(mat);
	result->Scale(scalar);
}

inline void Sparsem::DotMultiply(const SparseMatrix &a,
                                 const SparseMatrix &b,
                                 SparseMatrix *result) {
  DEBUG_ASSERT(a.num_of_columns_ == b.num_of_rows_);
	DEBUG_ASSERT(a.num_of_rows_ == result->num_of_rows_);
	DEBUG_ASSERT(b.num_of_columns_ == result->num_of_columns_);
	// If you try assigning the results to an already initialized matrix
	// you might get unexpected results. The following assertions
	// prevent you partially from that 
  DEBUG_ASSERT(&a != result); 
	DEBUG_ASSERT(&b != result);
	for(index_t r=0; r<a.num_of_rows_; r++) {
	  std::vector<index_t> indices3;
		std::vector<double>	 values3;
		indices3.clear();
		values3.clear();
		index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
	  a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r], num1, values1, indices1);
    b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r], num2, values2, indices2);
	  index_t i=0;
	  index_t j=0;
		while (likely(i<num1 && j<num2)) {
		  while (indices1[i] < indices2[j]) {
		    i++;	
        if unlikely((i>=num1)) {
  		    break;
			  }
	    }
		  if ( likely(i<num1) && indices1[i] == indices2[j]) {
	      values3.push_back(values1[i] * values2[j]);
				indices3.push_back(indices1[i]);
			} 
			j++;
	  }
		result->LoadRow(r, indices3, values3);
  }
}


