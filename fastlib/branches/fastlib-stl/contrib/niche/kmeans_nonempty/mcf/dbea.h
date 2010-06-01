/**************************************************************************
Contains modul DBEA.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:40:41 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: dbea.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _DBEA_H
#define _DBEA_H


#include "mcfdefs.h"


extern MCF_arc_p MCF_dual_bea( MCF_network_p net, MCF_node_p iminus, 
                               long *xchange, long *cycle_ori, 
                               MCF_cost_p red_cost_of_bea, MCF_flow_t delta );


#endif
