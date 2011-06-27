/***
 * @file arma_ostream_prefixed_meat.hpp
 * @author Ryan Curtin
 *
 * Slight change of arma_ostream_meat.hpp so that the arma output functions can
 * work with IO seamlessly.  We do not need to reimplement arma_ostream_state
 * (we can reuse it).  We implement one class, which uses PrefixedOutStream
 * where the Armadillo base library uses std::ostream.  It does not need to
 * implement every function from arma_ostream.
 */
#include <fastlib/fx/io.h>

template<typename eT>
inline
void
arma_ostream_prefixed::print_elem_zero(mlpack::io::PrefixedOutStream& o)
  {
  const std::streamsize orig_precision = o.destination.precision();

  o.destination.precision(0);

  o << eT(0);

  o.destination.precision(orig_precision);
  }

template<typename eT>
arma_inline
void
arma_ostream_prefixed::print_elem(mlpack::io::PrefixedOutStream& o, const eT& x)
  {
  if(x != eT(0))
    {
    o << x;
    }
  else
    {
    arma_ostream_prefixed::print_elem_zero<eT>(o);
    }
  }

template<typename T>
inline
void
arma_ostream_prefixed::print_elem(mlpack::io::PrefixedOutStream& o,
                                  const std::complex<T>& x)
  {
  if( (x.real() != T(0)) || (x.imag() != T(0)) )
    {
    std::ostringstream ss;
    ss.flags(o.destination.flags());
    ss.precision(o.destination.precision());

    ss << '(' << x.real() << ',' << x.imag() << ')';
    o << ss.str();
    }
  else
    {
    o << "(0,0)";
    }
  }

template<typename eT>
inline
void
arma_ostream_prefixed::print(mlpack::io::PrefixedOutStream& o, const Mat<eT>& m,
                             const bool modify)
  {

  arma_extra_debug_sigprint();

  const arma_ostream_state stream_state(o.destination);

  const std::streamsize cell_width = modify ?
      arma_ostream::modify_stream(o.destination, m.memptr(), m.n_elem) :
      o.destination.width() - 8 /* -8 for [?????] prefix */;

  const u32 m_n_rows = m.n_rows;
  const u32 m_n_cols = m.n_cols;

  if(m.is_empty() == false)
    {
    if(m_n_cols > 0)
      {
      if(cell_width > 0)
        {
        for(u32 row=0; row < m_n_rows; ++row)
          {
          for(u32 col=0; col < m_n_cols; ++col)
            {
            // the cell width appears to be reset after each element is printed,
            // hence we need to restore it
            o.destination.width(cell_width);
            arma_ostream_prefixed::print_elem(o, m.at(row,col));
            }

          o << '\n';
          }
        }
      else
        {
        for(u32 row=0; row < m_n_rows; ++row)
          {
          for(u32 col=0; col < m_n_cols - 1; ++col)
            {
            arma_ostream_prefixed::print_elem(o, m.at(row, col));
            o << ' ';
            }

          arma_ostream_prefixed::print_elem(o, m.at(row, m_n_cols - 1));
          o << '\n';
          }
        }
      }
    }
  else
    {
    o << "[matrix size: " << m_n_rows << 'x' << m_n_cols << "]\n";
    }

  o.destination.flush();
  stream_state.restore(o.destination);
  }

template<typename eT>
inline
void 
arma_ostream_prefixed::print(mlpack::io::PrefixedOutStream& o,
                             const Cube<eT>& x, const bool modify)
  {
  arma_extra_debug_sigprint();

  const arma_ostream_state stream_state(o.destination);

  const std::streamsize cell_width = modify ?
      arma_ostream::modify_stream(o.destination, x.memptr(), x.n_elem) :
      o.destination.width() - 8 /* -8 for [?????] prefix */;

  if(x.is_empty() == false)
    {
    for(u32 slice=0; slice < x.n_slices; ++slice)
      {
      o << "[cube slice " << slice << ']' << '\n';
      o.destination.width(cell_width);
      arma_ostream_prefixed::print(o, x.slice(slice), false);
      o << '\n';
      }
    }
  else
    {
    o << "[cube size: " << x.n_rows << 'x' << x.n_cols << 'x' << x.n_slices
        << "]\n";
    }

  stream_state.restore(o.destination);
  }

template<typename oT>
inline
void 
arma_ostream_prefixed::print(mlpack::io::PrefixedOutStream& o,
                             const field<oT>& x)
  {
  arma_extra_debug_sigprint();

  const arma_ostream_state stream_state(o.destination);

  /* -8 for [?????] prefix */
  const std::streamsize cell_width = o.destination.width() - 8;

  const u32 x_n_rows = x.n_rows;
  const u32 x_n_cols = x.n_cols;

  if(x.is_empty() == false)
    {
    for(u32 col = 0; col < x_n_cols; ++col)
      {
      o << "[field column " << col << "]\n";

      for(u32 row = 0; row < x_n_rows; ++row)
        {
        o.destination.width(cell_width);
        o << x.at(row,col) << '\n';
        }

      o << '\n';
      }
    }
  else
    {
    o << "[field size: " << x_n_rows << 'x' << x_n_cols << "]\n";
    }

  o.destination.flush();
  stream_state.restore(o.destination);
  }

template<typename oT>
inline
void
arma_ostream_prefixed::print(mlpack::io::PrefixedOutStream& o,
                             const subview_field<oT>& x)
  {
  arma_extra_debug_sigprint();

  const arma_ostream_state stream_state(o.destination);

  /* -8 for [?????] prefix */
  const std::streamsize cell_width = o.destination.width() - 8;

  const u32 x_n_rows = x.n_rows;
  const u32 x_n_cols = x.n_cols;

  for(u32 col = 0; col < x_n_cols; ++col)
    {
    o << "[field column " << col << "]\n";
    for(u32 row = 0; row < x_n_rows; ++row)
      {
      o.destination.width(cell_width);
      o << x.at(row, col) << '\n';
      }

    o << '\n';
    }

  o.destination.flush();
  stream_state.restore(o.destination);
  }
