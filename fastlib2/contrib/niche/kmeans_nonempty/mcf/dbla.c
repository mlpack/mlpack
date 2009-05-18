/**************************************************************************
Contains modul DBLA.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 23:39:52 2003 by Andreas Loebel (boss.local.de)  */
#ident "$Id: dbla.c,v 1.2 2003/06/03 22:18:39 bzfloebe Exp $"


#include "dbla.h"


MCF_node_p MCF_dual_iminus_cycle( MCF_DECLARE_DUAL_BLA_PARAMETERS )
{
    MCF_node_p node, stop;        
    MCF_node_p iminus = NULL;
    long subtreesize, best_subtreesize = 0;
    MCF_arc_p arc;
    MCF_flow_t flow, infeas, best_infeas = 0;
    
    // must be floating point because of possible arithmetic overflow
    double best = 0.0, tmp;   
    
    n = 0; /* to prevent compiler warnings */

    stop = stop_nodes;
    node = nodes;
    for( node++; node < stop; node++ )
    {
        arc = node->basic_arc;
        flow = node->flow;
#ifdef MCF_LOWER_BOUNDS
        infeas = arc->lower - flow;
#else
        infeas = -flow;
#endif      
        if( infeas > (MCF_flow_t)MCF_ZERO_EPS )
        {
            subtreesize = node->subtreesize;
            tmp = ( MCF_ABS(infeas * infeas) );
            if( tmp * subtreesize > best * best_subtreesize )
            {
                iminus = node;
                best = tmp;
                best_subtreesize = subtreesize;
                best_infeas = infeas;
            }
        }
        else if( (infeas = (arc->upper - flow)) < (MCF_flow_t)(-MCF_ZERO_EPS) )
        {
            subtreesize = node->subtreesize;
            tmp = ( MCF_ABS(infeas * infeas) );
            if( tmp * subtreesize > best * best_subtreesize )
            {
                iminus = node;
                best = tmp;
                best_subtreesize = subtreesize;
                best_infeas = infeas;
            }
        }
    }

    if( !iminus )
        return NULL; 

    *delta = best_infeas;
    return iminus;
}
