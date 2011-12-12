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

class arma_ostream_new
  {
  public:

  template<typename eT> inline static std::streamsize modify_stream(std::ostream& o, typename SpMat<eT>::const_iterator begin, typename SpMat<eT>::const_iterator end);
  template<typename eT> inline static void print(std::ostream& o, const SpMat<eT>& m, const bool modify);
  template<typename eT> inline static void print_trans(std::ostream& o, const SpMat<eT>& m, const bool modify);
  };

//! @}

