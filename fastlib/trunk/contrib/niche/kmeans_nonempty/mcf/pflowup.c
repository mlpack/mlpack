/**************************************************************************
Contains modul PFLOWUP.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:55:51 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: pflowup.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "pflowup.h"


void MCF_primal_update_flow( 
                 MCF_node_p iplus,
                 MCF_node_p jplus,
                 MCF_node_p w,
                 MCF_flow_t delta
                 )
{
    /* Update arc flows from iplus to w along the basis 
       path [iplus, root].
    */
    for( ; iplus != w; iplus = iplus->pred )
    {
        if( iplus->orientation )
            iplus->flow -= delta;
        else
            iplus->flow += delta;
    }

    /* Update arc flows from jplus to w along the basis 
       path [iplus, root]. 
    */
    for( ; jplus != w; jplus = jplus->pred )
    {
        if( jplus->orientation )
            jplus->flow += delta;
        else
            jplus->flow -= delta;
    }
}
