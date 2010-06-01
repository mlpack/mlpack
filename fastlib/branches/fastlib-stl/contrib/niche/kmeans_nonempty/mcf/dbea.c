/**************************************************************************
Contains modul DBEA.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:40:37 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: dbea.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "dbea.h"


#ifdef MCF_FLOAT
#define MCF_SCAN_ARC_LIST( first, next, ex, bound_ident ) \
{ \
    for( arc = first; arc; arc = arc->next ) \
    { \
        ident = arc->ident; \
        if( arc->tail->mark != arc->head->mark ) \
        { \
            if( ident == bound_ident ) \
            { \
                dual = arc->cost - arc->tail->potential \
                    + arc->head->potential; \
                if( MCF_ABS(dual) < abs_smallest_dual ) \
                { \
                    bea = arc; \
                    *xchange = ex; \
                    smallest_dual = dual; \
                    abs_smallest_dual = MCF_ABS(dual); \
                } \
if( dual >= LONG_MAX ) \
    printf( "(%s:%d): red_cost %f > UNBOUNDED\n", __FILE__, __LINE__, dual ); \
            } \
            else if( ident == MCF_AT_ZERO ) \
            { \
                bea = arc; \
                *xchange = ex; \
                smallest_dual = (MCF_cost_t)MCF_ZERO; \
                abs_smallest_dual = (MCF_cost_t)MCF_ZERO; \
                goto FOUND_BEA; \
            } \
        } \
    } \
}
#else
#define MCF_SCAN_ARC_LIST( first, next, ex, bound_ident ) \
{ \
    for( arc = first; arc; arc = arc->next ) \
    { \
        ident = arc->ident; \
        if( arc->tail->mark != arc->head->mark ) \
        { \
            if( ident == bound_ident ) \
            { \
                dual = arc->cost - arc->tail->potential \
                    + arc->head->potential; \
                if( MCF_ABS(dual) < abs_smallest_dual ) \
                { \
                    bea = arc; \
                    *xchange = ex; \
                    smallest_dual = dual; \
                    abs_smallest_dual = MCF_ABS(dual); \
                } \
if( dual >= LONG_MAX ) \
    printf( "(%s:%d): red_cost %ld > UNBOUNDED\n", __FILE__, __LINE__, dual ); \
            } \
            else if( ident == MCF_AT_ZERO ) \
            { \
                bea = arc; \
                *xchange = ex; \
                smallest_dual = (MCF_cost_t)MCF_ZERO; \
                abs_smallest_dual = (MCF_cost_t)MCF_ZERO; \
                goto FOUND_BEA; \
            } \
        } \
    } \
}
#endif
    


MCF_arc_p MCF_dual_bea( 
                MCF_network_p net,
                MCF_node_p iminus,
                long *xchange,
                long *cycle_ori,
                MCF_cost_p red_cost_of_bea,
                MCF_flow_t delta
                )
{
    MCF_node_p node, tmp;
    MCF_node_p root = net->nodes;
    MCF_arc_p arc;
    MCF_arc_p bea = NULL;
    long ident;
    MCF_cost_t dual;
    MCF_cost_t smallest_dual = (MCF_cost_t)LONG_MAX; /*UNBOUNDED*/
    MCF_cost_t abs_smallest_dual = (MCF_cost_t)LONG_MAX; /*UNBOUNDED*/
            




    /* search on the smaller subtree of the subtrees T1 and T2:
       subtree T1 contains the root!
       */
    if( iminus->subtreesize * 2 < net->n )
    /* search on subtree under iminus (T2) */
    {

        /* unmark subtree T1 */
        tmp = node = root;
        tmp->mark = 0;   
DOWN_1: 
        tmp = node->child;
        if( tmp )
        {  
MARK_1: 
            if( tmp != iminus )
                tmp->mark = 0;   
            else 
            { 
                tmp = tmp->right_sibling;
                if( tmp ) 
                    goto MARK_1;
                else
                {
                    if( iminus == node->right_sibling )
                        node = node->pred;
                    goto RIGHT_1;
                }
            } 
            node = tmp; 
            goto DOWN_1;
        } 
RIGHT_1: 
        if( node == root )
            goto CONTINUE_1;
        tmp = node->right_sibling;
        if( tmp ) 
            goto MARK_1; 
        node = node->pred; 
        goto RIGHT_1;
CONTINUE_1:
        
        /* mark subtree T2 */
        tmp = node = iminus; 
        tmp->mark = 1;   
DOWN_2:
        tmp = node->child; 
        if( tmp )
        { 
MARK_2:
            tmp->mark = 1;  
            node = tmp; 
            goto DOWN_2;
        }  
RIGHT_2: 
        if( node == iminus)
            goto CONTINUE_2; 
        tmp = node->right_sibling; 
        if( tmp ) 
            goto MARK_2;
        node = node->pred;
        goto RIGHT_2; 
CONTINUE_2: 
        
        /* scan all arcs of the set delta(T2) */
        if( (iminus->orientation == MCF_DOWN && delta < (MCF_flow_t)(-MCF_ZERO))
           || (iminus->orientation == MCF_UP && delta > (MCF_flow_t)MCF_ZERO) )
        {
            *cycle_ori = 0;
            tmp = node = iminus; 
            MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 1, MCF_AT_LOWER );
            MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 0, MCF_AT_UPPER );
DOWN_3:
            tmp = node->child; 
            if( tmp ) 
            {  
MARK_3:                 
                MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 1, MCF_AT_LOWER );
                MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 0, MCF_AT_UPPER );
                node = tmp;
                goto DOWN_3;
            }  
RIGHT_3: 
            if( node == iminus) 
                goto CONTINUE_3; 
            tmp = node->right_sibling; 
            if( tmp ) 
                goto MARK_3; 
            node = node->pred; 
            goto RIGHT_3;  
CONTINUE_3:
            ;
        }
        else
        {
            *cycle_ori = 1;
            tmp = node = iminus; 
            MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 1, MCF_AT_UPPER );
            MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 0, MCF_AT_LOWER); 
DOWN_4: 
            tmp = node->child; 
            if( tmp ) 
            {  
MARK_4:
                MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 1, MCF_AT_UPPER );
                MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 0, MCF_AT_LOWER); 
                node = tmp; 
                goto DOWN_4; 
            }  
RIGHT_4: 
            if( node == iminus) 
                goto CONTINUE_4; 
            tmp = node->right_sibling; 
            if( tmp ) 
                goto MARK_4; 
            node = node->pred; 
            goto RIGHT_4;  
CONTINUE_4:
            ;
        }
    }
    else
    /* serach on subtree under the root */
    {
        /* mark subtree T1 */
        tmp = node = root;  
        tmp->mark = 1;   
DOWN_5: 
        tmp = node->child; 
        if( tmp ) 
        {  
MARK_5: 
            if( tmp != iminus ) 
                tmp->mark = 1;
            else 
            { 
                tmp = tmp->right_sibling; 
                if( tmp ) 
                    goto MARK_5; 
                else 
                {
                    if( iminus == node->right_sibling )
                        node = node->pred;
                    goto RIGHT_5;
                }
            }
            node = tmp; 
            goto  DOWN_5; 
        }  
RIGHT_5: 
        if( node == root ) 
            goto CONTINUE_5; 
        tmp = node->right_sibling; 
        if( tmp ) 
            goto MARK_5; 
        node = node->pred; 
        goto RIGHT_5;  
CONTINUE_5: 


        /* unmark subtree T2 */
        tmp = node = iminus;
        tmp->mark = 0;     
DOWN_6: 
        tmp = node->child; 
        if( tmp ) 
        {  
MARK_6:                 
            tmp->mark = 0;    
            node = tmp; 
            goto DOWN_6; 
        }  
RIGHT_6: 
        if( node == iminus) 
            goto CONTINUE_6; 
        tmp = node->right_sibling; 
        if( tmp ) 
            goto MARK_6; 
        node = node->pred; 
        goto RIGHT_6;  
CONTINUE_6: 

        /* scan all arcs of the set delta(T1) */
        if( (iminus->orientation == MCF_DOWN && delta > (MCF_flow_t)MCF_ZERO)
         || (iminus->orientation == MCF_UP && delta < (MCF_flow_t)(-MCF_ZERO)) )
        {
            *cycle_ori = 1;
            tmp = node = root;  
            MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 0, MCF_AT_LOWER );
            MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 1, MCF_AT_UPPER );
DOWN_7: 
            tmp = node->child; 
            if( tmp ) 
            {  
MARK_7: 
                if( tmp != iminus ) 
                { 
                    MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 0, MCF_AT_LOWER );
                    MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 1, MCF_AT_UPPER );
                } 
                else 
                { 
                    tmp = tmp->right_sibling; 
                    if( tmp ) 
                        goto MARK_7; 
                    else 
                    {
                        if( iminus == node->right_sibling )
                            node = node->pred;
                        goto RIGHT_7;
                    }   
                } 
                node = tmp; 
                goto DOWN_7; 
            }  
RIGHT_7: 
            if( node == root ) 
                goto CONTINUE_7; 
            tmp = node->right_sibling; 
            if( tmp ) 
                goto MARK_7; 
            node = node->pred; 
            goto RIGHT_7;  
CONTINUE_7: 
            ;
        }
        else
        {
            *cycle_ori = 0;
            tmp = node = root;
            MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 0, MCF_AT_UPPER );
            MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 1, MCF_AT_LOWER );
DOWN_8: 
            tmp = node->child; 
            if( tmp ) 
            {  
MARK_8: 
                if( tmp != iminus ) 
                {                   
                    MCF_SCAN_ARC_LIST( tmp->firstin, nextin, 0, MCF_AT_UPPER );
                    MCF_SCAN_ARC_LIST( tmp->firstout, nextout, 1,MCF_AT_LOWER );
                } 
                else 
                { 
                    tmp = tmp->right_sibling; 
                    if( tmp ) 
                        goto MARK_8; 
                    else 
                    {
                        if( iminus == node->right_sibling )
                            node = node->pred;
                        goto RIGHT_8;
                    }   
                } 
                node = tmp; 
                goto DOWN_8; 
            }  
RIGHT_8: 
            if( node == root ) 
                goto CONTINUE_8; 
            tmp = node->right_sibling; 
            if( tmp ) 
                goto MARK_8; 
            node = node->pred; 
            goto RIGHT_8;  
CONTINUE_8: 
            ;
        }   
    }

FOUND_BEA:

    *red_cost_of_bea = smallest_dual;
    return bea;
}
