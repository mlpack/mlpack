/**************************************************************************
Contains modul DBLA.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:40:50 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: dbla.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _DBLA_H
#define _DBLA_H


#include "mcfdefs.h"
#include "dbladef.h"


extern MCF_node_p MCF_dual_iminus_cycle( MCF_DECLARE_DUAL_BLA_PARAMETERS );
extern MCF_node_p MCF_dual_iminus_mpp_30_5( MCF_DECLARE_DUAL_BLA_PARAMETERS );
extern MCF_node_p MCF_dual_iminus_mpp_50_10( MCF_DECLARE_DUAL_BLA_PARAMETERS );


#endif
