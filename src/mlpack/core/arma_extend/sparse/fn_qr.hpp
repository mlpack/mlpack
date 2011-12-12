//! QR decomposition for sparse matrices
template<typename T1>
inline
bool
sp_qr
  (
         Mat<typename T1::elem_type>&   Q,
         SpMat<typename T1::elem_type>&   R,
   T1& X
  )
  {
  arma_extra_debug_sigprint();
  
  //arma_debug_check( (&Q == &R), "qr(): Q and R are the same object");

  //cs_dln *cs_dl_qr (const cs_dl *A, const cs_dls *S) ;

  cs_di A;
  A.nzmax = X.n_nonzero;
  A.m = X.n_rows;
  A.n = X.n_cols;
  A.p = (int*) &X.col_ptrs[0];
  A.i = (int*) &X.row_indices[0];
  A.x = &X.values[0];
  A.nz = -1; // Indicate to cxsparse that this is a CSC matrix

  cs_dis *S;
  S = cs_sqr(3, &A, 1);

  cs_din *res;
  res = cs_di_qr(&A, S);
  cs_sfree(S);


  cs_print (res->L, 1);
  std::cout << std::endl;
  cs_print (res->U, 1);
  std::cout << std::endl;

  for(uword i = 0; i < X.n_cols; ++i)
    std::cout << res->B[i] << '\t';
  std::cout << std::endl;

  // Copy values from res->L to Q
  assert(res->L->nz == -1);
  size_t nzmax = res->L->nzmax;

  SpMat<typename T1::elem_type> V;

  V.set_size (res->L->m, res->L->n);
  V.values.resize (nzmax);
  V.row_indices.resize (nzmax);

  memmove (&V.values[0], res->L->x, sizeof(typename T1::elem_type) * nzmax);
  memmove (&V.row_indices[0], res->L->i, sizeof(int) * nzmax);
  memmove (&V.col_ptrs[0], res->L->p, sizeof(int) * res->L->n + 1);

  access::rw (V.n_nonzero) = nzmax;
  access::rw (V.n_elem) = nzmax;

  Q.eye(V.n_rows, V.n_cols);
  Mat<typename T1::elem_type> M, vvH;
  M.eye(V.n_rows, V.n_cols);
  Col<typename T1::elem_type> v;
  for(uword i = 0; i < X.n_cols; ++i)
    {
    //v = V.col(i) / res->B[i];
    vvH = v * trans(v);
    Q -= vvH;
    M *= vvH;
    }
  Q += M;

  // Copy values from res->U to R
  assert(res->U->nz == -1);
  nzmax = res->U->nzmax;

  R.set_size (res->U->m, res->U->n);
  R.values.resize (nzmax);
  R.row_indices.resize (nzmax);

  memmove (&R.values[0], res->U->x, sizeof(typename T1::elem_type) * nzmax);
  memmove (&R.row_indices[0], res->U->i, sizeof(int) * nzmax);
  memmove (&R.col_ptrs[0], res->U->p, sizeof(int) * res->U->n + 1);

  access::rw (R.n_nonzero) = nzmax;
  access::rw (R.n_elem) = nzmax;

  return true;
  }
