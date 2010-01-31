/**************************************************************************
Contains modul TREEUP.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:58:02 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: treeup.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "treeup.h"


void MCF_update_tree( 
                 long cycle_ori,
                 long new_orientation,
                 MCF_flow_t delta,
                 MCF_flow_t new_flow,
                 MCF_node_p iplus,
                 MCF_node_p jplus,
                 MCF_node_p iminus,
                 MCF_node_p jminus,
                 MCF_node_p w,
                 MCF_arc_p bea,
                 MCF_cost_t sigma
                 )
{

    MCF_arc_p basic_arc_temp; 
    MCF_arc_p new_basic_arc;  
    MCF_node_p father;         
    MCF_node_p temp;           
    MCF_node_p new_pred;       
    long orientation_temp;
    long subtreesize_temp;      
    long subtreesize_iminus;    
    long new_subtreesize;       
    MCF_flow_t flow_temp;       






    /* Initalize sigma */
    if( (bea->tail == jplus && sigma < 0) ||
       (bea->tail == iplus && sigma > 0) )
        sigma = MCF_ABS(sigma);
    else
        sigma = -(MCF_ABS(sigma));
    
    /* Update of the node potentials */
    father = iminus;
    father->potential += sigma;
RECURSION:
    temp = father->child;
    if( temp )
    {
ITERATION:
        temp->potential += sigma;
        father = temp;
        goto RECURSION;
    }
TEST:
    if( father == iminus )
        goto CONTINUE;
    temp = father->right_sibling;
    if( temp )
        goto ITERATION;
    father = father->pred;
    goto TEST;

CONTINUE:

    /* Traverse subtree unter iminus in reversed order. */
    temp = iplus;
    father = temp->pred;
    new_subtreesize = subtreesize_iminus = iminus->subtreesize;
    new_pred = jplus;
    new_basic_arc = bea;
    while( temp != jminus )
    {
        /* Cut subtree under temp from subtree under father */
        if( temp->right_sibling )
            temp->right_sibling->left_sibling = temp->left_sibling;
        if( temp->left_sibling )
            temp->left_sibling->right_sibling = temp->right_sibling;
        else father->child = temp->right_sibling;

        /* temp becomes first child of new_pred */
        temp->pred = new_pred;
        temp->right_sibling = new_pred->child;
        if( temp->right_sibling )
            temp->right_sibling->left_sibling = temp;
        new_pred->child = temp;
        temp->left_sibling = 0;

        /* Save orientation, flow, subtreesize und basic_arc. */
        orientation_temp = !(temp->orientation); 
        if( orientation_temp == cycle_ori )
            flow_temp = temp->flow + delta;
        else
            flow_temp = temp->flow - delta;
        basic_arc_temp = temp->basic_arc;
        subtreesize_temp = temp->subtreesize;

        /* Update orientation, flow, subtreesize und basic_arc of temp. */
        temp->orientation = new_orientation;
        temp->flow = new_flow;
        temp->basic_arc = new_basic_arc;
        temp->subtreesize = new_subtreesize;

        /* Prepare for next iteration in while-loop. */
        new_pred = temp;
        new_orientation = orientation_temp;
        new_flow = flow_temp;
        new_basic_arc = basic_arc_temp;
        new_subtreesize = subtreesize_iminus - subtreesize_temp;        
        temp = father;
        father = temp->pred;
    } 

    /* Update flow and subtreesize on path from jminus and jplus to w. */
    if( delta > MCF_ZERO_EPS )
    {
        for( temp = jminus; temp != w; temp = temp->pred )
        {
            temp->subtreesize -= subtreesize_iminus;
            if( temp->orientation != cycle_ori )
                temp->flow += delta;
            else
                temp->flow -= delta;
        }
        for( temp = jplus; temp != w; temp = temp->pred )
        {
            temp->subtreesize += subtreesize_iminus;
            if( temp->orientation == cycle_ori )
                temp->flow += delta;
            else
                temp->flow -= delta;
        }
    }
    else
    {
        for( temp = jminus; temp != w; temp = temp->pred )
            temp->subtreesize -= subtreesize_iminus;
        for( temp = jplus; temp != w; temp = temp->pred )
            temp->subtreesize += subtreesize_iminus;
    }

}
