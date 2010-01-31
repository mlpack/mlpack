/*@HEADER
// ***********************************************************************
//
//       Ifpack: Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2002) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//@HEADER
*/

#ifndef _ifp_parameters_h_
#define _ifp_parameters_h_

#include <Ifpack_ConfigDefs.h>

#include <Teuchos_map.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Epetra_CombineMode.h>

namespace Ifpack {

//define enum values to which parameter names will be mapped.
enum parameter {
  //parameters of type double
  absolute_threshold,
  relative_threshold,
  drop_tolerance,
  fill_tolerance,
  relax_value,

  //parameters of type int
  //(if you add or remove int parameters, be sure to
  //update FIRST_INT_PARAM and LAST_INT_PARAM macros below, as
  //they are used below and in ifp_parameters.cpp)
  level_fill,
  level_overlap,
  num_steps,

  //mixed type parameters
  use_reciprocal,
  overlap_mode
};

#define FIRST_INT_PARAM Ifpack::level_fill
#define LAST_INT_PARAM Ifpack::num_steps

//define struct with union of all Ifpack parameters
struct param_struct {
  int int_params[LAST_INT_PARAM-FIRST_INT_PARAM+1];
  double double_params[FIRST_INT_PARAM];
  bool use_reciprocal;
  Epetra_CombineMode overlap_mode;
};

Teuchos::map<string,parameter>& key_map();

void initialize_string_map();

string upper_case(const string& s);

void set_parameters(const Teuchos::ParameterList& parameterlist,
                    param_struct& params,
                    bool cerr_warning_if_unused=false);

}//namespace Ifpack

#endif //_ifp_parameters_h_

