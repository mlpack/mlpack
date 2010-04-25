/**************************************************************************
Contains modul DSTART.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:46:13 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: dstart.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "dstart.h"


#define ERROR( str, returncode ) \
{ \
    fprintf( stderr, "DUAL NETWORK SIMPLEX: " ); \
    fprintf( stderr, str ); \
    MCF_FREE( old_lower ); \
    MCF_FREE( old_upper ); \
    return returncode; \
}




long MCF_dual_start_artificial( MCF_network_p net )
{      
    MCF_node_p tmp, node, root;
    MCF_arc_p arc;
    void *stop;
    MCF_flow_t l, u;
    MCF_flow_p old_lower = NULL, old_upper = NULL;
    MCF_flow_p lower, upper;
    MCF_cost_t red_cost;
    MCF_cost_t potential_from_root;
    long no_free_arcs;
    
    int latitude_warning = 0;
    





    potential_from_root = (-2)*MCF_MAX_ART_COST;

    /* Initialize artificial node (called root) */
    root = node = net->nodes; node++;
    root->basic_arc = NULL;
    root->pred = NULL;
    root->child = node;
    root->right_sibling = NULL;
    root->left_sibling = NULL;
    root->subtreesize = (net->n) + 1;
    root->orientation = 0;
    root->potential = -MCF_MAX_ART_COST;
    root->flow = MCF_ZERO;
    root->firstout = NULL;
    root->firstin = NULL;

    /* search for free or one-bounded variables */
    arc = net->arcs;
    stop = net->stop_arcs;
#ifdef MCF_LOWER_BOUNDS
    for( no_free_arcs = 1; arc < (MCF_arc_p)stop && no_free_arcs; arc++ )
        if( arc->lower <= MCF_ZERO_EPS - MCF_UNBOUNDED 
           || arc->upper >= MCF_UNBOUNDED - MCF_ZERO_EPS )
            no_free_arcs = 0;
#else
    no_free_arcs = 1;
#endif
        

    if( !no_free_arcs )
        /* call a phase I for the dual network simplex */
    {
        lower = old_lower = (MCF_flow_p)calloc( net->m, sizeof(MCF_flow_t) );
        upper = old_upper = (MCF_flow_p)calloc( net->m, sizeof(MCF_flow_t) );
        
        if( !(old_lower && old_upper) )
            ERROR( "not enough memory\n", 1 );
        
        /* store old bounds */
        stop = net->stop_arcs;
        arc = net->arcs;
        for( ; arc < (MCF_arc_p)stop; arc++, lower++, upper++ )
        {
#ifdef MCF_LOWER_BOUNDS
            l = *lower = arc->lower;
#else
            l = *lower = (MCF_flow_t)0;
#endif
            u = *upper = arc->upper;
            if( u - l < -MCF_ZERO_EPS )
            {
                while( arc >= net->arcs )
                {
#ifdef MCF_LOWER_BOUNDS
                    arc->lower = *lower;
#endif
                    arc->upper = *upper;
                    arc--;
                    lower--;
                    upper--;
                }
                net->feasible = 0;
                ERROR( "problem infeasible\n", 0 );
            }
            else if( l > -MCF_UNBOUNDED && u < MCF_UNBOUNDED )
            {
#ifdef MCF_LOWER_BOUNDS
                arc->lower = 0;
#endif
                arc->upper = 0;
                arc->ident = MCF_FIXED;
            }
            else if( l <= MCF_ZERO_EPS - MCF_UNBOUNDED && u < MCF_UNBOUNDED )
            {
#ifdef MCF_LOWER_BOUNDS
                arc->lower = -1;
#endif
                arc->upper = 0;
                if( arc->cost < 0 )
                    arc->ident = MCF_AT_UPPER;
                else
                {
                    arc->ident = MCF_AT_LOWER;
                    arc->tail->flow += 1;
                    arc->head->flow -= 1;
                }
            }
            else if( l > -MCF_UNBOUNDED )
            {
#ifdef MCF_LOWER_BOUNDS
                arc->lower = 0;
#endif
                arc->upper = 1;
                if( arc->cost < 0 )
                {
                    arc->ident = MCF_AT_UPPER;
                    arc->tail->flow -= 1;
                    arc->head->flow += 1;
                }
                else
                    arc->ident = MCF_AT_LOWER;              
            }
            else /* free variable */
            {
#ifdef MCF_LOWER_BOUNDS
                arc->lower = -1;
#endif
                arc->upper = 1;
                if( arc->cost < 0 )
                {
                    arc->ident = MCF_AT_UPPER;
                    arc->tail->flow -= 1;
                    arc->head->flow += 1;
                }
                else
                {
                    arc->ident = MCF_AT_LOWER;  
                    arc->tail->flow += 1;
                    arc->head->flow -= 1;
                }
            }
        }

        /* prepare dummy arcs and node potentials */
        arc = net->dummy_arcs;
        for( stop = net->stop_nodes; node < (MCF_node_p)stop; arc++, node++ )
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
            
            /* node->flow += node->balance; balance = 0! */
            
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
        root--;
        

        /* call primal routine */
        MCF_primal_net_simplex( net );
        if( MCF_ABS(MCF_primal_obj( net )) > (double)MCF_ZERO_EPS )
            net->feasible = 0;
        else
            net->feasible = 1;
        
        /* prepare node->flows */
        node = root;
        stop = net->stop_nodes;
        for( node++; node < (MCF_node_p)stop; node++ )
            node->flow = node->balance;
                    
        /* restore old bounds and update the right basis solution */
        stop = net->stop_arcs;
        lower = old_lower;
        upper = old_upper;
        for( arc = net->arcs; arc < (MCF_arc_p)stop; arc++, lower++, upper++ )
        {
#ifdef MCF_LOWER_BOUNDS
            l = arc->lower = *lower;
#else
            l = *lower;
#endif
            u = arc->upper = *upper;
            if( arc->ident == MCF_FIXED )
            {
                if( u - l > MCF_ZERO_EPS )
                {
                    red_cost = arc->cost - arc->tail->potential 
                        + arc->head->potential;
                    if( red_cost < (MCF_cost_t)MCF_ZERO )
                    {
                        arc->ident = MCF_AT_UPPER;
                        arc->tail->flow -= u;
                        arc->head->flow += u;
                    }
                    else
                    {
                        arc->ident = MCF_AT_LOWER;
                        arc->tail->flow -= l;
                        arc->head->flow += l;
                    }
                }
                else
                {
                    arc->tail->flow -= l;
                    arc->head->flow += l;
                }
            }
        }

        if( !net->feasible )
        {
            fprintf( stderr, "DUAL NETWORK SIMPLEX: " );
            fprintf( stderr, "dual problem infeasible\n" );
            return 1;
        }
        
        
        /* compute new basis solution */
        tmp = node = root; 
GO_DOWN: 
        tmp = node->child; 
        if( tmp ) 
        { 
ITERATE:   
            node = tmp; 
            goto GO_DOWN; 
        } 
RIGHT: 
        if( node == root ) 
            goto CONTINUE; 
        if( node->orientation == MCF_UP )
            node->pred->flow += node->flow;
        else 
            node->pred->flow -= node->flow;          
        tmp = node->right_sibling; 
        if( tmp ) 
            goto ITERATE; 
        node = node->pred; 
        goto RIGHT; 
CONTINUE: 
        
        MCF_FREE( old_lower );
        MCF_FREE( old_upper );
    }

    else /* no_free_arc */

    {
        /* prepare dummy arcs and node potentials */
        stop = net->stop_nodes;
        for( arc = net->dummy_arcs; node < (MCF_node_p)stop; arc++, node++ )
        {
            node->basic_arc = arc;
            node->pred = root;  
            node->child = NULL;   
            node->right_sibling = node + 1; 
            node->left_sibling = node - 1;
            node->subtreesize = 1;
            node->flow = node->balance;
            node->orientation = MCF_UP; 
            node->potential = MCF_ZERO;
            
            arc->cost = MCF_MAX_ART_COST;
#ifdef MCF_LOWER_BOUNDS
            arc->lower = (MCF_flow_t)MCF_ZERO;
#endif
            arc->upper = (MCF_flow_t)MCF_ZERO;
            arc->tail = node;
            arc->head = root;                
            arc->ident = MCF_BASIC;
        }
                    
        node--; root++;
        node->right_sibling = NULL;
        root->left_sibling = NULL;
        root--;
        
        stop = net->stop_arcs;
        for( arc = net->arcs; arc < (MCF_arc_p)stop; arc++ )
        {
#ifdef MCF_LOWER_BOUNDS
            if( arc->upper - arc->lower <= (MCF_flow_t)MCF_ZERO_EPS )
#else
            if( arc->upper <= (MCF_flow_t)MCF_ZERO_EPS )
#endif
            {
                arc->ident = MCF_FIXED;
#ifdef MCF_LOWER_BOUNDS
                arc->tail->flow -= arc->lower;
                arc->head->flow += arc->lower;
#endif
            }
            else if( arc->cost < 0 )
            {
                arc->ident = MCF_AT_UPPER;
                arc->tail->flow -= arc->upper;
                arc->head->flow += arc->upper;
            }
            else
            {
                arc->ident = MCF_AT_LOWER;
#ifdef MCF_LOWER_BOUNDS
                arc->tail->flow -= arc->lower;
                arc->head->flow += arc->lower;
#endif
            }
            
            if( !latitude_warning 
                && 
                (arc->tail->flow > MCF_UNBOUNDED 
                 || arc->tail->flow < -MCF_UNBOUNDED 
                 ||
                 arc->head->flow > MCF_UNBOUNDED 
                 || arc->head->flow < -MCF_UNBOUNDED )
                )
            {
                latitude_warning = 1;
                printf( "\nwarning(%s:%d): ", __FILE__, __LINE__ );
                printf( "latitude of your bounds may be too large!\n" );
            }
        }
    }

    return 0;
}
