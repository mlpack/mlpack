/**************************************************************************
Contains modul TREEUP.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:58:20 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: treeup.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _TREEUP_H
#define _TREEUP_H


#include "mcfdefs.h"


extern void MCF_update_tree(
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
                            );


#endif
