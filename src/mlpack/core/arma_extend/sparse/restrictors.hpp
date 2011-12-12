// Copyright (C) 2011 Ryan Curtin <ryan@igglybob.com>
//
//// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

//! \addtogroup restrictors
//! @{

template<typename T> struct arma_cxsparse_only { };

template<> struct arma_cxsparse_only<double> { typedef double result; };


template<typename T> struct arma_SpMat_SpCol_SpRow_only { };

template<typename eT> struct arma_SpMat_SpCol_SpRow_only< SpMat<eT> > { typedef SpMat<eT> result; };
template<typename eT> struct arma_SpMat_SpCol_SpRow_only< SpCol<eT> > { typedef SpCol<eT> result; };
template<typename eT> struct arma_SpMat_SpCol_SpRow_only< SpRow<eT> > { typedef SpRow<eT> result; };


//! @}
