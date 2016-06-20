#if (ARMA_VERSION_MAJOR < 7 && ARMA_VERSION_MINOR < 200)
template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
op_min::min_with_index(const Proxy<T1>& P, uword& index_of_min_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_elem = P.get_n_elem();
  
  if(n_elem == 0)
    {
    arma_debug_check(true, "min(): object has no elements");
    
    return Datum<eT>::nan;
    }
  
  eT    best_val   = priv::most_pos<eT>();
  uword best_index = 0;
  
  if(Proxy<T1>::use_at == false)
    {
    typedef typename Proxy<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    for(uword i=0; i<n_elem; ++i)
      {
      const eT tmp = A[i];
      
      if(tmp < best_val)  { best_val = tmp;  best_index = i; }
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      for(uword i=0; i < n_cols; ++i)
        {
        const eT tmp = P.at(0,i);
        
        if(tmp < best_val)  { best_val = tmp;  best_index = i; }
        }
      }
    else
    if(n_cols == 1)
      {
      for(uword i=0; i < n_rows; ++i)
        {
        const eT tmp = P.at(i,0);
        
        if(tmp < best_val)  { best_val = tmp;  best_index = i; }
        }
      }
    else
      {
      uword count = 0;
      
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        const eT tmp = P.at(row,col);
        
        if(tmp < best_val)  { best_val = tmp;  best_index = count; }
        
        ++count;
        }
      }
    }
  
  index_of_min_val = best_index;
  
  return best_val;
  }

template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
op_min::min_with_index(const ProxyCube<T1>& P, uword& index_of_min_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_elem = P.get_n_elem();
  
  if(n_elem == 0)
    {
    arma_debug_check(true, "min(): object has no elements");
    
    return Datum<eT>::nan;
    }
  
  eT    best_val   = priv::most_pos<eT>();
  uword best_index = 0;
  
  if(ProxyCube<T1>::use_at == false)
    {
    typedef typename ProxyCube<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    for(uword i=0; i < n_elem; ++i)
      {
      const eT tmp = A[i];
      
      if(tmp < best_val)  { best_val = tmp;  best_index = i; }
      }
    }
  else
    {
    const uword n_rows   = P.get_n_rows();
    const uword n_cols   = P.get_n_cols();
    const uword n_slices = P.get_n_slices();
    
    uword count = 0;
    
    for(uword slice=0; slice < n_slices; ++slice)
    for(uword   col=0;   col < n_cols;   ++col  )
    for(uword   row=0;   row < n_rows;   ++row  )
      {
      const eT tmp = P.at(row,col,slice);
      
      if(tmp < best_val)  { best_val = tmp;  best_index = count; }
      
      ++count;
      }
    }
  
  index_of_min_val = best_index;
  
  return best_val;
  }

  template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
op_max::max_with_index(const Proxy<T1>& P, uword& index_of_max_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_elem = P.get_n_elem();
  
  if(n_elem == 0)
    {
    arma_debug_check(true, "max(): object has no elements");
    
    return Datum<eT>::nan;
    }
  
  eT    best_val   = priv::most_neg<eT>();
  uword best_index = 0;
  
  if(Proxy<T1>::use_at == false)
    {
    typedef typename Proxy<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    for(uword i=0; i<n_elem; ++i)
      {
      const eT tmp = A[i];
      
      if(tmp > best_val)  { best_val = tmp;  best_index = i; }
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      for(uword i=0; i < n_cols; ++i)
        {
        const eT tmp = P.at(0,i);
        
        if(tmp > best_val)  { best_val = tmp;  best_index = i; }
        }
      }
    else
    if(n_cols == 1)
      {
      for(uword i=0; i < n_rows; ++i)
        {
        const eT tmp = P.at(i,0);
        
        if(tmp > best_val)  { best_val = tmp;  best_index = i; }
        }
      }
    else
      {
      uword count = 0;
      
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        const eT tmp = P.at(row,col);
        
        if(tmp > best_val)  { best_val = tmp;  best_index = count; }
        
        ++count;
        }
      }
    }
  
  index_of_max_val = best_index;
  
  return best_val;
  }

template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
op_max::max_with_index(const ProxyCube<T1>& P, uword& index_of_max_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_elem = P.get_n_elem();
  
  if(n_elem == 0)
    {
    arma_debug_check(true, "max(): object has no elements");
    
    return Datum<eT>::nan;
    }
  
  eT    best_val   = priv::most_neg<eT>();
  uword best_index = 0;
  
  if(ProxyCube<T1>::use_at == false)
    {
    typedef typename ProxyCube<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    for(uword i=0; i < n_elem; ++i)
      {
      const eT tmp = A[i];
      
      if(tmp > best_val)  { best_val = tmp;  best_index = i; }
      }
    }
  else
    {
    const uword n_rows   = P.get_n_rows();
    const uword n_cols   = P.get_n_cols();
    const uword n_slices = P.get_n_slices();
    
    uword count = 0;
    
    for(uword slice=0; slice < n_slices; ++slice)
    for(uword   col=0;   col < n_cols;   ++col  )
    for(uword   row=0;   row < n_rows;   ++row  )
      {
      const eT tmp = P.at(row,col,slice);
      
      if(tmp > best_val)  { best_val = tmp;  best_index = count; }
      
      ++count;
      }
    }
  
  index_of_max_val = best_index;
  
  return best_val;
  }

template<typename elem_type, typename derived>
inline
arma_warn_unused
uword
Base<elem_type,derived>::index_min() const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  uword index = 0;
  
  if(P.get_n_elem() == 0)
    {
    arma_debug_check(true, "index_min(): object has no elements");
    }
  else
    {
    op_min::min_with_index(P, index);
    }
  
  return index;
  }


template<typename elem_type, typename derived>
inline
arma_warn_unused
uword
Base<elem_type,derived>::index_max() const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  uword index = 0;
  
  if(P.get_n_elem() == 0)
    {
    arma_debug_check(true, "index_max(): object has no elements");
    }
  else
    {
    op_max::max_with_index(P, index);
    }
  
  return index;
  }
#endif