

template<typename eT>
class SpSubview : public Base<eT, SpSubview<eT> >
  {
  public: const SpMat<eT>& m;

  public:

  typedef eT elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

#if (ARMA_VERSION_MAJOR) >= 3
  static const bool is_row = false;
  static const bool is_col = false;
#endif

  const uword aux_row1;
  const uword aux_col1;
  const uword n_rows;
  const uword n_cols;
  const uword n_elem;
  const uword n_nonzero;

  protected:

  arma_inline SpSubview(const SpMat<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols);
  arma_inline SpSubview(      SpMat<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols);

  public:

  inline ~SpSubview();

  inline void operator+= (const eT val);
  inline void operator-= (const eT val);
  inline void operator*= (const eT val);
  inline void operator/= (const eT val);

  template<typename T1> inline void operator=  (const Base<eT,T1>& x);
  template<typename T1> inline void operator+= (const Base<eT,T1>& x);
  template<typename T1> inline void operator-= (const Base<eT,T1>& x);
  template<typename T1> inline void operator%= (const Base<eT,T1>& x);
  template<typename T1> inline void operator/= (const Base<eT,T1>& x);

  /* not doing anything special
  inline void operator=  (const SpSubview& x);
  inline void operator+= (const SpSubview& x);
  inline void operator-= (const SpSubview& x);
  inline void operator%= (const SpSubview& x);
  inline void operator/= (const SpSubview& x);
  */

  /*
  inline static void extract(SpMat<eT>& out, const SpSubview& in);

  inline static void  plus_inplace(Mat<eT>& out, const subview& in);
  inline static void minus_inplace(Mat<eT>& out, const subview& in);
  inline static void schur_inplace(Mat<eT>& out, const subview& in);
  inline static void   div_inplace(Mat<eT>& out, const subview& in);
  */

  inline void fill(const eT val);
  inline void zeros();
  inline void ones();
  inline void eye();

  inline eT& operator[](const uword i);
  inline eT  operator[](const uword i) const;

  inline eT& operator()(const uword i);
  inline eT  operator()(const uword i) const;

  inline eT& operator()(const uword in_row, const uword in_col);
  inline eT  operator()(const uword in_row, const uword in_col) const;

  inline eT&         at(const uword in_row, const uword in_col);
  inline eT          at(const uword in_row, const uword in_col) const;

  inline bool check_overlap(const SpSubview& x) const;

  inline bool is_vec() const;

/* not yet
  inline       SpSubview_row<eT> row(const uword row_num);
  inline const SpSubview_row<eT> row(const uword row_num) const;

  inline            SpSubview_row<eT> operator()(const uword row_num, const span& col_span);
  inline      const SpSubview_row<eT> operator()(const uword row_num, const span& col_span) const;

  inline       SpSubview_col<eT> col(const uword col_num);
  inline const SpSubview_col<eT> col(const uword col_num) const;

  inline            SpSubview_col<eT> operator()(const span& row_span, const uword col_num);
  inline      const SpSubview_col<eT> operator()(const span& row_span, const uword col_num) const;

  inline            Col<eT>  unsafe_col(const uword col_num);
  inline      const Col<eT>  unsafe_col(const uword col_num) const;

  inline       SpSubview<eT> rows(const uword in_row1, const uword in_row2);
  inline const SpSubview<eT> rows(const uword in_row1, const uword in_row2) const;

  inline       SpSubview<eT> cols(const uword in_col1, const uword in_col2);
  inline const SpSubview<eT> cols(const uword in_col1, const uword in_col2) const;

  inline       SpSubview<eT> submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2);
  inline const SpSubview<eT> submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const;

  inline            SpSubview<eT> submat    (const span& row_span, const span& col_span);
  inline      const SpSubview<eT> submat    (const span& row_span, const span& col_span) const;

  inline            SpSubview<eT> operator()(const span& row_span, const span& col_span);
  inline      const SpSubview<eT> operator()(const span& row_span, const span& col_span) const;

  inline       diagview<eT> diag(const s32 in_id = 0);
  inline const diagview<eT> diag(const s32 in_id = 0) const;
*/

  inline void swap_rows(const uword in_row1, const uword in_row2);
  inline void swap_cols(const uword in_col1, const uword in_col2);

  private:
  friend class SpMat<eT>;
  SpSubview();

  };

/*
template<typename eT>
class SpSubview_col : public SpSubview<eT>
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  inline void operator= (const SpSubview<eT>& x);
  inline void operator= (const SpSubview_col& x);

  template<typename T1>
  inline void operator= (const Base<eT,T1>& x);

  inline       SpSubview_col<eT> rows(const uword in_row1, const uword in_row2);
  inline const SpSubview_col<eT> rows(const uword in_row1, const uword in_row2) const;

  inline       SpSubview_col<eT> subvec(const uword in_row1, const uword in_row2);
  inline const SpSubview_col<eT> subvec(const uword in_row1, const uword in_row2) const;


  protected:

  inline SpSubview_col(const Mat<eT>& in_m, const uword in_col);
  inline SpSubview_col(      Mat<eT>& in_m, const uword in_col);

  inline SpSubview_col(const Mat<eT>& in_m, const uword in_col, const uword in_row1, const uword in_n_rows);
  inline SpSubview_col(      Mat<eT>& in_m, const uword in_col, const uword in_row1, const uword in_n_rows);


  private:

  friend class Mat<eT>;
  friend class Col<eT>;
  friend class SpSubview<eT>;

  SpSubview_col();
  };

template<typename eT>
class SpSubview_row : public SpSubview<eT>
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  inline void operator= (const SpSubview<eT>& x);
  inline void operator= (const SpSubview_row& x);

  template<typename T1>
  inline void operator= (const Base<eT,T1>& x);

  inline       SpSubview_row<eT> cols(const uword in_col1, const uword in_col2);
  inline const SpSubview_row<eT> cols(const uword in_col1, const uword in_col2) const;

  inline       SpSubview_row<eT> subvec(const uword in_col1, const uword in_col2);
  inline const SpSubview_row<eT> subvec(const uword in_col1, const uword in_col2) const;


  protected:

  inline SpSubview_row(const Mat<eT>& in_m, const uword in_row);
  inline SpSubview_row(      Mat<eT>& in_m, const uword in_row);

  inline SpSubview_row(const Mat<eT>& in_m, const uword in_row, const uword in_col1, const uword in_n_cols);
  inline SpSubview_row(      Mat<eT>& in_m, const uword in_row, const uword in_col1, const uword in_n_cols);


  private:

  friend class Mat<eT>;
  friend class Row<eT>;
  friend class SpSubview<eT>;

  SpSubview_row();
  };
*/
