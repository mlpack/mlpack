/**************************************************************************
Contains modul PBEADEF.H of ZIB optimizer MCF

AUTHOR: Andreas Loebel

This software was developed at ZIB Berlin. Maintenance and revisions 
on responsibility of Dr. Andreas Loebel

Dr. Andreas Loebel
Ortlerweg 29b, 12207 Berlin

Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)
Scientific Computing - Optimization
Takustr. 7, 14195 Berlin-Dahlem

Copyright (c) 1997-2000 ZIB.            All rights reserved.
Copyright (c) 2000-2003 ZIB & Loebel.   All rights reserved.
**************************************************************************/
/*  LAST EDIT: Tue Jun  3 10:52:46 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: pbeadef.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _PBEADEF_H
#define _PBEADEF_H


#define MCF_PRIMAL_BEA_PARAMETERS \
    long , \
    MCF_arc_p, \
    MCF_arc_p, \
    MCF_cost_p

#define MCF_DECLARE_PRIMAL_BEA_PARAMETERS \
    long m, \
    MCF_arc_p arcs, \
    MCF_arc_p stop_arcs, \
    MCF_cost_p red_cost_of_bea



#define MCF_COMPUTE_RED_COST( y ) \
      y = arc->cost - arc->tail->potential + arc->head->potential



/*
#define MCF_DUAL_INFEAS( c ) \
    ( c < -MCF_ZERO_EPS && \
      (arc->ident == MCF_AT_LOWER || arc->ident == MCF_AT_ZERO) ) \
 || ( c > MCF_ZERO_EPS && \
    (arc->ident == MCF_AT_UPPER || arc->ident == MCF_AT_ZERO) )
*/
#define MCF_DUAL_INFEAS( c ) \
    ( c < -MCF_ZERO_EPS && arc->ident == MCF_AT_LOWER ) \
 || ( c > MCF_ZERO_EPS && arc->ident == MCF_AT_UPPER )
        


#define MCF_CMP_ASSIGN_BEST( c, best, abs_best, arc_index ) \
{ \
    if( c + abs_best < -MCF_ZERO_EPS && \
            (arc->ident == MCF_AT_LOWER || arc->ident == MCF_AT_ZERO) ) \
    { \
        best = c; \
        abs_best = MCF_ABS(c); \
        arc_index = arc; \
    } \
    else if( c - abs_best > MCF_ZERO_EPS && \
            (arc->ident == MCF_AT_UPPER || arc->ident == MCF_AT_ZERO) ) \
    { \
        best = c; \
        abs_best = MCF_ABS(c); \
        arc_index = arc; \
    } \
}



#define MCF_IMPROVE( vertex, first_list, next_list ) \
{ \
      for( arc = j->vertex->first_list; arc; arc = arc->next_list ) \
      if( arc->ident > MCF_BASIC ) \
      { \
           MCF_COMPUTE_RED_COST( red_cost ); \
           MCF_CMP_ASSIGN_BEST( red_cost, most_neg, abs_most_neg, bea ); \
      } \
}                                                         

#include "mcfdefs.h"

#endif /* _PBEADEF_H */
