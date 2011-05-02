/** @file tools.h
 *
 *  This file is a collection of tools useful for operating on matrices
 *
 *  @author Nishant Mehta (niche)
 *  @bug No known bugs
 */

#ifndef TOOLS_H
#define TOOLS_H

#define INSIDE_TOOLS_H

using namespace arma;
using namespace std;

void RemoveRows(const mat& X, uvec rows_to_remove, mat& X_mod);


#include "tools_impl.h"
#undef INSIDE_TOOLS_H

#endif
