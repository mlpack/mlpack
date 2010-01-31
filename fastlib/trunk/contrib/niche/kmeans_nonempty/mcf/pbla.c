/**************************************************************************
Contains modul PBLA.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:55:17 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: pbla.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "pbla.h"


#define MCF_TEST_MIN( nod, ex, comp ) \
{ \
      if( *delta comp temp ) \
      { \
            iminus = nod; \
            *delta = temp; \
            *xchange = ex; \
      } \
}




MCF_node_p MCF_primal_iminus( 
                      MCF_flow_p delta,
                      long *xchange,
                      MCF_node_p iplus, 
                      MCF_node_p jplus,
                      MCF_node_p *w
                      )
{
    MCF_node_p iminus = NULL;
    MCF_flow_t temp;




    /* There are two paths from iplus and jplus to the root.
       The first common node of these two paths and an arc on the
       cycle [iplus,root]+[jplus,root]-([iplus,root]*[jplus,root]
       will be determined, "*" means the average of the two
       paths.
        */
    while( iplus != jplus )
    {
        if( iplus->subtreesize < jplus->subtreesize )
        {
            /* Proceed in path from iplus to root. */
            if( iplus->orientation )
            {
#ifdef MCF_LOWER_BOUNDS
                temp = iplus->basic_arc->lower;
                if( temp > -MCF_UNBOUNDED )
                {
                    temp = iplus->flow - temp;
                    MCF_TEST_MIN( iplus, 0, > );
                }
#else
                temp = iplus->flow;
                MCF_TEST_MIN( iplus, 0, > );
#endif              
            }
            else
            {
                temp = iplus->basic_arc->upper;
                if( temp < MCF_UNBOUNDED )
                {
                    temp -= iplus->flow;
                    MCF_TEST_MIN( iplus, 0, > );
                }
            }
            iplus = iplus->pred;
        }
        else
        {
            /* Proceed in path from jplus to root. */
            if( jplus->orientation )
            {
                temp = jplus->basic_arc->upper;
                if( temp < MCF_UNBOUNDED )
                {
                    temp -= jplus->flow;
                    MCF_TEST_MIN( jplus, 1, >= );
                }
            }
            else
            {
#ifdef MCF_LOWER_BOUNDS
                temp = jplus->basic_arc->lower;
                if( temp > -MCF_UNBOUNDED )
                {
                    temp = jplus->flow - temp;
                    MCF_TEST_MIN( jplus, 1, >= );
                }
#else
                temp = jplus->flow;
                MCF_TEST_MIN( jplus, 1, >= );
#endif
            }
            jplus = jplus->pred;
        }
    } 

    *w = iplus;

    return iminus;
}
