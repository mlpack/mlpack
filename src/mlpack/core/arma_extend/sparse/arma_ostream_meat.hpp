// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// Copyright (C) 2008-2011 Conrad Sanderson
// 
// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)


//! \addtogroup arma_ostream
//! @{

template<typename eT>
inline
std::streamsize
arma_ostream_new::modify_stream(std::ostream& o, typename SpMat<eT>::const_iterator begin, typename SpMat<eT>::const_iterator end) {
  o.unsetf(ios::showbase);
  o.unsetf(ios::uppercase);
  o.unsetf(ios::showpos);

  o.fill(' ');

  std::streamsize cell_width;

  bool use_layout_B  = false;
  bool use_layout_C  = false;

  for(typename SpMat<eT>::const_iterator it = begin; it != end; ++it)
    {
    const eT val = *it;

    if(
      val >= eT(+100) ||
      ( (is_signed<eT>::value == true) && (val <= eT(-100)) ) ||
      ( (is_non_integral<eT>::value == true) && (val > eT(0)) && (val <= eT(+1e-4)) ) ||
      ( (is_non_integral<eT>::value == true) && (is_signed<eT>::value == true) && (val < eT(0)) && (val >= eT(-1e-4)) )
      )
      {
      use_layout_C = true;
      break;
      }

    if(
      (val >= eT(+10)) || ( (is_signed<eT>::value == true) && (val <= eT(-10)) )
      )
      {
      use_layout_B = true;
      }
    }

  if(use_layout_C == true)
    {
    o.setf(ios::scientific);
    o.setf(ios::right);
    o.unsetf(ios::fixed);
    o.precision(4);
    cell_width = 13;
    }
  else
  if(use_layout_B == true)
    {
    o.unsetf(ios::scientific);
    o.setf(ios::right);
    o.setf(ios::fixed);
    o.precision(4);
    cell_width = 10;
    }
  else
    {
    o.unsetf(ios::scientific);
    o.setf(ios::right);
    o.setf(ios::fixed);
    o.precision(4);
    cell_width = 9;
    }

  return cell_width;
}

template<typename eT> 
inline 
void 
arma_ostream_new::print(std::ostream& o, const SpMat<eT>& m, const bool modify)
  {
  arma_extra_debug_sigprint();

  const arma_ostream_state stream_state(o);

  // TODO: Figure this out:
  //const std::streamsize cell_width = o.width();
  const std::streamsize cell_width = modify ? modify_stream<eT>(o, m.begin(), m.end()) : o.width();

  const uword m_n_rows = m.n_rows;
  const uword m_n_cols = m.n_cols;

  typename SpMat<eT>::const_iterator begin = m.begin();
  typename SpMat<eT>::const_iterator end = m.end();

  if(m.is_empty() == false)
    {   
    if(m_n_cols > 0)
      {   
      if(cell_width > 0)
        {
	// An efficient row_iterator would make this simpler and faster
        for(uword row=0; row < m_n_rows; ++row)
          {   
          for(uword col=0; col < m_n_cols; ++col)
            {   
            // the cell width appears to be reset after each element is printed,
            // hence we need to restore it
            o.width(cell_width);
	    eT val = 0;
	    for(typename SpMat<eT>::const_iterator it = begin; it != end; ++it)
	      {
		if(it != end && it.row == row && it.col == col)
		{
		  val = *it;
		  break;
		}
	      }
	    arma_ostream::print_elem(o,eT(val));
            }   
    
          o << '\n';
          }   
        }   
      else
        {   
	// An efficient row_iterator would make this simpler and faster
        for(uword row=0; row < m_n_rows; ++row)
          {   
          for(uword col=0; col < m_n_cols; ++col)
            {   
	    eT val = 0;
	    for(typename SpMat<eT>::const_iterator it = begin; it != end; ++it)
	      {
		if(it != end && it.row == row && it.col == col)
		{
		  val = *it;
		  break;
		}
	      }
	    arma_ostream::print_elem(o,eT(val));
            o << ' ';
            }   
    
          o << '\n';
          }   
        }   
      }   
    }   
  else
    {   
    o << "[matrix size: " << m_n_rows << 'x' << m_n_cols << "]\n";
    }   
  
  o.flush();
  stream_state.restore(o);

  }

template<typename eT> 
inline 
void 
arma_ostream_new::print_trans(std::ostream& o, const SpMat<eT>& m, const bool modify)
  {
  arma_extra_debug_sigprint();

  const arma_ostream_state stream_state(o);

  // TODO: Figure this out:
  //const std::streamsize cell_width = o.width();
  const std::streamsize cell_width = modify ? modify_stream<eT>(o, m.begin(), m.end()) : o.width();

  // trans(mat(m,n)) = mat(n,m) => we swap cols and rows
  const uword m_n_rows = m.n_cols;
  const uword m_n_cols = m.n_rows;

  typename SpMat<eT>::const_iterator it = m.begin();
  typename SpMat<eT>::const_iterator end = m.end();

  if(m.is_empty() == false)
    {   
    if(m_n_cols > 0)
      {   
      if(cell_width > 0)
        {
	for(uword row=0; row < m_n_rows; ++row)
          {   
	  for(uword col=0; col < m_n_cols; ++col)
            {   
            o.width(cell_width);
	    if(it != end && it.row == col && it.col == row )
	      {
	      arma_ostream::print_elem(o,eT(*it));
	      ++it;
	      }
	    else
	      {
	      arma_ostream::print_elem(o,eT(0));
	      }
            }   
    
          o << '\n';
          }   
        }   
      else
        {   
	for(uword row=0; row < m_n_rows; ++row)
          {   
	  for(uword col=0; col < m_n_cols; ++col)
            {   
	    eT val = 0;
	    if(it != end && it.row == col && it.col == row )
	      {
	      arma_ostream::print_elem(o,eT(*it));
	      ++it;
	      }
	    else
	      {
	      arma_ostream::print_elem(o,eT(0));
	      }
            o << ' ';
            }   
    
          o << '\n';
          }   
        }   
      }   
    }   
  else
    {   
    o << "[matrix size: " << m_n_rows << 'x' << m_n_cols << "]\n";
    }   
  
  o.flush();
  stream_state.restore(o);

  }


//! @}

