/**************************************************************************
Contains modul DSIMPLEX.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 17:19:30 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: dsimplex.c,v 1.2 2003/06/03 15:19:43 bzfloebe Exp $"


#undef MCF_DEBUG_NO


#include "dsimplex.h"


long MCF_dual_net_simplex(  MCF_network_p net )
{
    MCF_flow_t    delta;
    MCF_flow_t    new_flow = MCF_ZERO;
    long          opt = 0;
    long          xchange;
    long          cycle_ori;
    MCF_node_p    nodes      = net->nodes;
    MCF_node_p    stop_nodes = net->stop_nodes;
    MCF_node_p    iplus;
    MCF_node_p    jplus;
    MCF_node_p    iminus;
    MCF_node_p    jminus;
    MCF_node_p    w; 
    MCF_arc_p     bea;
    MCF_arc_p     bla;
    MCF_arc_p     arcs          = net->arcs;
    MCF_arc_p     dummy_arcs    = net->dummy_arcs;
    MCF_arc_p     stop_dummy    = net->stop_dummy;
    long          n = net->n;
    long          new_set;
    MCF_cost_t    red_cost_of_bea;
    long          *iterations = &(net->iterations);
    MCF_node_p    (*dual_iminus)( MCF_DECLARE_DUAL_BLA_PARAMETERS )
                = (MCF_node_p(*)( MCF_DUAL_BLA_PARAMETERS ))(net->find_iminus);



#if defined MCF_DEBUG || 0
    printf( "%8ld: %10.6f\n", *iterations, MCF_dual_obj(net) );
#endif


    while( !opt )
    {       
        /* Find basis leaving arc if not primal feasible */
        if( (iminus = dual_iminus( n, nodes, stop_nodes, &delta )) != NULL )
        {
            (*iterations)++;

            /* Find basis entering arc bea */ 
            bea = MCF_dual_bea( net, iminus, &xchange, &cycle_ori, 
                               &red_cost_of_bea, delta );

            /* If the dual problem is unbounded, the primal
               problem is infeasible
            */
            if( red_cost_of_bea >= LONG_MAX /* MCF_UNBOUNDED-1*/ )
            {
                net->feasible = 0;
                return 1;
            }

            /* Initialize j-, i+ and i-:
               j+ should be the predecessor of i+ and
               j- should be the predecessor of i-
               */
            jminus = iminus->pred;
            if( xchange )
            {
                iplus = bea->head;
                jplus = bea->tail;
            }
            else
            {
                iplus = bea->tail;
                jplus = bea->head;
            }

            /* Assign basis leaving arc */
            bla = iminus->basic_arc;

            /* Find the first common node w in the paths 
               from j+ and j- to the root
               */
            w = MCF_dual_w( jplus, jminus );
            
            /* New ident of bla */
            if( delta > (MCF_flow_t)MCF_ZERO )
                new_set = MCF_AT_LOWER;
            else
                new_set = MCF_AT_UPPER;
                        
            /* The new flow value of the bea */
            switch( bea->ident )
            {
            case MCF_AT_UPPER:
                new_flow = bea->upper - MCF_ABS(delta);
                break;
            case MCF_AT_LOWER:
#ifdef MCF_LOWER_BOUNDS
                new_flow = bea->lower + MCF_ABS(delta);
#else
                new_flow = MCF_ABS(delta);
#endif
                break;
            case MCF_AT_ZERO:
                if( cycle_ori == xchange )
                    new_flow = -MCF_ABS(delta);
                else
                    new_flow = MCF_ABS(delta);
                break;
            default:
                break;
            }
            
            /* Update basis structure:
               treeup was written for the primal network simplex;
               it is assumed that the duals are not feasible,
               therefore the negative duals have to be deliverd.
               */
            MCF_update_tree( cycle_ori, !xchange, 
                        MCF_ABS(delta), new_flow, iplus, jplus, iminus, 
                        jminus, w, bea, red_cost_of_bea );

            /* Update ident of bea and bla */
            bea->ident = MCF_BASIC;
            bla->ident = new_set;
        }
        else
        {
            /* problem is primal feasible too */
            opt = 1;

            /* If there exists an artificial basic arc with positive flow, the
               network has no feasible solution or the algorithm can't delete
               all artificial basic arcs (perhaps the assumed cost of the
               artificial basic arcs are not big enough).  
            */
            w = net->nodes;
            for( w++; w != net->stop_nodes; w++ )
            {
                arcs = w->basic_arc;
                if( arcs >= dummy_arcs && arcs < stop_dummy )
                {
                    if( w->flow )
                    {
                        fprintf( stderr,"No feasible flow found\n");
                        net->feasible = 0;
                        return 1;
                    }
                }
            }
        }

#if defined MCF_DEBUG || 0
        if( !(*iterations % 1000) )
            printf( "%8ld: %10.6f\n", *iterations, MCF_dual_obj(net) );
#endif
    }

    MCF_primal_feasible( net );
#ifdef MCF_DEBUG
    MCF_dual_feasible(net);
#endif

    return 0;
}
