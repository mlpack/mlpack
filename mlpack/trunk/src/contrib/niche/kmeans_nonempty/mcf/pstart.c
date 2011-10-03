/**************************************************************************
Contains modul PSTART.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:57:04 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: pstart.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "pstart.h"


long MCF_primal_start_artificial( MCF_network_p net )
{      
    MCF_node_p node, root;
    MCF_arc_p arc;
    void *stop;
    MCF_flow_t lower, upper;
    double tmp;
    MCF_cost_t potential_from_root;





    /* Initialize arc_cost and potential_from_root. 
       If the costs of the original arcs are too large the
       simplex algorithm possibly can't find a feasible flow without
       artificial arcs (that means all artificial arcs with flow zero).
       Than the MCF_MAX_ART_COST in defines.h has to be increased!
    */
    potential_from_root = (MCF_cost_t) (-2)*MCF_MAX_ART_COST;


    /* Initialize artificial node (called root) */
    root = node = net->nodes; node++;
    root->basic_arc = NULL;
    root->pred = NULL;
    root->child = node;
    root->right_sibling = NULL;
    root->left_sibling = NULL;
    root->subtreesize = (net->n) + 1;
    root->orientation = 0;
    root->potential = (MCF_cost_t) -MCF_MAX_ART_COST;
    root->flow = MCF_ZERO;
    root->firstout = NULL;
    root->firstin = NULL;

    for( arc = net->arcs, stop = net->stop_arcs; arc != stop; arc++ )
    {
#ifdef MCF_LOWER_BOUNDS
        lower = arc->lower;
#else
        lower = (MCF_flow_t)0;
#endif
        upper = arc->upper;
#ifdef MCF_FLOAT
        tmp = (double)upper - (double)lower;
        if( tmp < (double)-MCF_ZERO_EPS )
#else
        tmp = upper - lower;
        if( tmp < MCF_ZERO_EPS )
#endif        
        {
            fprintf( stderr,"Problem infeasible" );
            return 1;
        }
        if( tmp < MCF_ZERO_EPS )
        {
            arc->ident = MCF_FIXED;
            arc->tail->flow -= lower;
            arc->head->flow += lower;
        }
        else if( upper < 1000000 && arc->cost < 0 )
        {
            arc->ident = MCF_AT_UPPER;
            arc->tail->flow -= upper;
            arc->head->flow += upper;
        }
        else if( lower > -MCF_UNBOUNDED && -lower <= upper )
        {
            arc->ident = MCF_AT_LOWER;
            arc->tail->flow -= lower;
            arc->head->flow += lower;
        }
        else if( upper < MCF_UNBOUNDED )
        {
            arc->ident = MCF_AT_UPPER;
            arc->tail->flow -= upper;
            arc->head->flow += upper;
        }
        else
            arc->ident = MCF_AT_ZERO;
    }
        
    arc = net->dummy_arcs;
    for( stop = net->stop_nodes; node != stop; arc++, node++ )
    {
        node->basic_arc = arc;
        node->pred = root;
        node->child = NULL;
        node->right_sibling = node + 1; 
        node->left_sibling = node - 1;
        node->subtreesize = 1;

        arc->cost = MCF_MAX_ART_COST;
#ifdef MCF_LOWER_BOUNDS
        arc->lower = (MCF_flow_t)MCF_ZERO;
#endif
        arc->upper = 2*MCF_UNBOUNDED + 1; /* >max delta for a variable */

        node->flow += node->balance;
        
        if( node->flow >= 0 ) 
        /* Make the artificial arc (i,0) active. */
        {
            node->orientation = MCF_UP; 
            node->potential = MCF_ZERO;
            arc->tail = node;
            arc->head = root;                
            arc->ident = MCF_BASIC;
        }
        else 
        /* Make the artificial arc (0,i) active. */
        {
            node->orientation = MCF_DOWN;
            node->flow = -node->flow;
            node->potential = potential_from_root;
            arc->tail = root;
            arc->head = node;
            arc->ident = MCF_BASIC;
        }
    }

    node--; root++;
    node->right_sibling = NULL;
    root->left_sibling = NULL;

    return 0;
}
