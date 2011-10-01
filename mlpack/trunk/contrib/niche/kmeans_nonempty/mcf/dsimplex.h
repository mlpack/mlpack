/**************************************************************************
Contains modul DSIMPLEX.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:45:56 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: dsimplex.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _DSIMPLEX_H
#define _DSIMPLEX_H

#include "mcfdefs.h"
#include "mcfutil.h"
#include "dbladef.h"
#include "dbea.h"
#include "dw.h"
#include "treeup.h"


extern long MCF_dual_net_simplex( MCF_network_p net );


#endif
