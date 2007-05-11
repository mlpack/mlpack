// Copyright (c) 1994 Darren Erik Vengroff
//
// File: ami_scan_mac.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/24/94
//
// $Id: ami_scan_mac.h,v 1.10 2003/04/25 00:06:56 tavi Exp $
//
#ifndef _AMI_SCAN_MAC_H
#define _AMI_SCAN_MAC_H

// Macros for defining parameters to AMI_scan()
#define __SPARM_BASE(T,io,n) AMI_STREAM< T ## n > *io ## n
#define __SPARM_1(T,io) __SPARM_BASE(T,io,1)  
#define __SPARM_2(T,io) __SPARM_1(T,io), __SPARM_BASE(T,io,2)
#define __SPARM_3(T,io) __SPARM_2(T,io), __SPARM_BASE(T,io,3)
#define __SPARM_4(T,io) __SPARM_3(T,io), __SPARM_BASE(T,io,4)

// Macros for defining types in a template for AMI_scan()
#define __STEMP_BASE(T,n) class T ## n
#define __STEMP_1(T) __STEMP_BASE(T,1)  
#define __STEMP_2(T) __STEMP_1(T), __STEMP_BASE(T,2)
#define __STEMP_3(T) __STEMP_2(T), __STEMP_BASE(T,3)
#define __STEMP_4(T) __STEMP_3(T), __STEMP_BASE(T,4)

// Temporary space used within AMI_scan
#define __STS_BASE(T,t,n) T ## n t ## n 
#define __STSPACE_1(T,t) __STS_BASE(T,t,1)
#define __STSPACE_2(T,t) __STSPACE_1(T,t) ; __STS_BASE(T,t,2) 
#define __STSPACE_3(T,t) __STSPACE_2(T,t) ; __STS_BASE(T,t,3) 
#define __STSPACE_4(T,t) __STSPACE_3(T,t) ; __STS_BASE(T,t,4) 

// An array of flags.
#define __FSPACE(f,n) AMI_SCAN_FLAG f[n]


// Check stream validity.
#define __CHK_BASE(T,n) {                                               \
    if (T ## n == NULL || T ## n -> status() != AMI_STREAM_STATUS_VALID) {\
        return AMI_ERROR_GENERIC_ERROR;                                 \
    }                                                                   \
}

#define __CHKSTR_1(T) __CHK_BASE(T,1)
#define __CHKSTR_2(T) __CHKSTR_1(T) __CHK_BASE(T,2)
#define __CHKSTR_3(T) __CHKSTR_2(T) __CHK_BASE(T,3)
#define __CHKSTR_4(T) __CHKSTR_3(T) __CHK_BASE(T,4)


// Rewind the input streams prior to performing the scan.
#define __REW_BASE(T,n) {						\
    if ((_ami_err_ = T ## n -> seek(0)) != AMI_ERROR_NO_ERROR) {	\
        return _ami_err_;						\
    }									\
}

#define __REWIND_1(T) __REW_BASE(T,1)
#define __REWIND_2(T) __REWIND_1(T) __REW_BASE(T,2)
#define __REWIND_3(T) __REWIND_2(T) __REW_BASE(T,3)
#define __REWIND_4(T) __REWIND_3(T) __REW_BASE(T,4)


// Set the input flags to true before entering the do loop so that the
// initial values will be read.
#define __SET_IF_BASE(f,n) f[n-1] = 1

#define __SET_IF_1(f) __SET_IF_BASE(f,1)
#define __SET_IF_2(f) __SET_IF_1(f); __SET_IF_BASE(f,2)
#define __SET_IF_3(f) __SET_IF_2(f); __SET_IF_BASE(f,3)
#define __SET_IF_4(f) __SET_IF_3(f); __SET_IF_BASE(f,4)

// If the flag is set, then read inputs into temporary space.  Set the
// flag based on whether the read was succesful or not.  If it was
// unsuccessful for any reason other than EOS, then break out of the
// scan loop.  If the flag is not currently set, then either the scan
// management object did not take the last input or the last time we
// tried to read from this file we failed.  If we read successfully
// last time, then reset the flag.
#define __STSR_BASE(t,ts,f,g,e,n)					    \
if (f[n-1]) {								    \
    if (!(f[n-1] = g[n-1] =						    \
          ((e = ts ## n->read_item(&t ## n)) == AMI_ERROR_NO_ERROR))) {	    \
        if (e != AMI_ERROR_END_OF_STREAM) {				    \
            break;							    \
        }								    \
    }									    \
} else {								    \
    f[n-1] = g[n-1];							    \
}

#define __STS_READ_1(t,ts,f,g,e) __STSR_BASE(t,ts,f,g,e,1) 
#define __STS_READ_2(t,ts,f,g,e) __STS_READ_1(t,ts,f,g,e)		    \
        __STSR_BASE(t,ts,f,g,e,2)
#define __STS_READ_3(t,ts,f,g,e) __STS_READ_2(t,ts,f,g,e)		    \
        __STSR_BASE(t,ts,f,g,e,3)
#define __STS_READ_4(t,ts,f,g,e) __STS_READ_3(t,ts,f,g,e)		    \
        __STSR_BASE(t,ts,f,g,e,4)

// Write outputs.  Only write if the flag is set.  If there is an
// error during the write, then break out of the scan loop.
#define __STSW_BASE(u,us,f,e,n)						    \
if (f[n-1] && (e = us ## n -> write_item(u ## n)) != AMI_ERROR_NO_ERROR) {  \
    break;								    \
}

#define __STS_WRITE_1(u,us,f,e) __STSW_BASE(u,us,f,e,1)
#define __STS_WRITE_2(u,us,f,e) __STS_WRITE_1(u,us,f,e) __STSW_BASE(u,us,f,e,2)
#define __STS_WRITE_3(u,us,f,e) __STS_WRITE_2(u,us,f,e) __STSW_BASE(u,us,f,e,3)
#define __STS_WRITE_4(u,us,f,e) __STS_WRITE_3(u,us,f,e) __STSW_BASE(u,us,f,e,4)


// Arguments to the operate() call
#define __SCA_BASE(t,n) t ## n
#define __SCALL_ARGS_1(t) __SCA_BASE(t,1)
#define __SCALL_ARGS_2(t) __SCALL_ARGS_1(t), __SCA_BASE(t,2) 
#define __SCALL_ARGS_3(t) __SCALL_ARGS_2(t), __SCA_BASE(t,3) 
#define __SCALL_ARGS_4(t) __SCALL_ARGS_3(t), __SCA_BASE(t,4) 

// Operate on the inputs to produce the outputs.
#define __SCALL_BASE(t,nt,if,sop,u,nu,of) \
    sop->operate(__SCALL_ARGS_ ## nt (*t), if, __SCALL_ARGS_ ## nu (&u), of)

#define __SCALL_OP_1_1(t,if,sop,u,of) __SCALL_BASE(t,1,if,sop,u,1,of)
#define __SCALL_OP_1_2(t,if,sop,u,of) __SCALL_BASE(t,1,if,sop,u,2,of)
#define __SCALL_OP_1_3(t,if,sop,u,of) __SCALL_BASE(t,1,if,sop,u,3,of)
#define __SCALL_OP_1_4(t,if,sop,u,of) __SCALL_BASE(t,1,if,sop,u,4,of)

#define __SCALL_OP_2_1(t,if,sop,u,of) __SCALL_BASE(t,2,if,sop,u,1,of)
#define __SCALL_OP_2_2(t,if,sop,u,of) __SCALL_BASE(t,2,if,sop,u,2,of)
#define __SCALL_OP_2_3(t,if,sop,u,of) __SCALL_BASE(t,2,if,sop,u,3,of)
#define __SCALL_OP_2_4(t,if,sop,u,of) __SCALL_BASE(t,2,if,sop,u,4,of)

#define __SCALL_OP_3_1(t,if,sop,u,of) __SCALL_BASE(t,3,if,sop,u,1,of)
#define __SCALL_OP_3_2(t,if,sop,u,of) __SCALL_BASE(t,3,if,sop,u,2,of)
#define __SCALL_OP_3_3(t,if,sop,u,of) __SCALL_BASE(t,3,if,sop,u,3,of)
#define __SCALL_OP_3_4(t,if,sop,u,of) __SCALL_BASE(t,3,if,sop,u,4,of)

#define __SCALL_OP_4_1(t,if,sop,u,of) __SCALL_BASE(t,4,if,sop,u,1,of)
#define __SCALL_OP_4_2(t,if,sop,u,of) __SCALL_BASE(t,4,if,sop,u,2,of)
#define __SCALL_OP_4_3(t,if,sop,u,of) __SCALL_BASE(t,4,if,sop,u,3,of)
#define __SCALL_OP_4_4(t,if,sop,u,of) __SCALL_BASE(t,4,if,sop,u,4,of)

// Handle the no input case.
#define __SCALL_BASE_O(sop,u,nu,of) \
    sop->operate(__SCALL_ARGS_ ## nu (&u), of)

#define __SCALL_OP_O_1(sop,u,of) __SCALL_BASE_O(sop,u,1,of)
#define __SCALL_OP_O_2(sop,u,of) __SCALL_BASE_O(sop,u,2,of)
#define __SCALL_OP_O_3(sop,u,of) __SCALL_BASE_O(sop,u,3,of)
#define __SCALL_OP_O_4(sop,u,of) __SCALL_BASE_O(sop,u,4,of)

// Handle the no output case.
#define __SCALL_BASE_I(t,nt,if,sop) \
    sop->operate(__SCALL_ARGS_ ## nt (*t), if)

#define __SCALL_OP_I_1(t,if,sop) __SCALL_BASE_I(t,1,if,sop)
#define __SCALL_OP_I_2(t,if,sop) __SCALL_BASE_I(t,2,if,sop)
#define __SCALL_OP_I_3(t,if,sop) __SCALL_BASE_I(t,3,if,sop)
#define __SCALL_OP_I_4(t,if,sop) __SCALL_BASE_I(t,4,if,sop)


// The template for the whole AMI_scan(), with inputs and outputs.
#define __STEMPLATE(in_arity, out_arity)				    \
template< __STEMP_ ## in_arity (T), class SC, __STEMP_ ## out_arity (U) >   \
AMI_err AMI_scan( __SPARM_ ## in_arity (T,_ts_),			    \
                  SC *soper, __SPARM_ ## out_arity (U,_us_))		    \
{	    								    \
    __STSPACE_ ## in_arity (T,*_t_);					    \
    __STSPACE_ ## out_arity (U,_u_);					    \
	    								    \
    __FSPACE(_if_,in_arity);						    \
    __FSPACE(_lif_,in_arity);						    \
    __FSPACE(_of_,out_arity);						    \
	    								    \
    AMI_err _op_err_, _ami_err_;					    \
	    								    \
    __CHKSTR_ ## in_arity (_ts_)                                            \
    __CHKSTR_ ## out_arity (_us_)                                           \
    __REWIND_ ## in_arity (_ts_)					    \
    soper->initialize();						    \
                                                                            \
    __SET_IF_ ## in_arity (_if_);					    \
                                                                            \
    do {	    						    	    \
	    								    \
        __STS_READ_ ## in_arity (_t_,_ts_,_if_,_lif_,_ami_err_)		    \
            								    \
        _op_err_ = __SCALL_OP_ ## in_arity ## _ ##			    \
            out_arity(_t_,_if_,soper,_u_,_of_);				    \
	    								    \
        __STS_WRITE_ ## out_arity(_u_,_us_,_of_,_ami_err_)		    \
            								    \
    } while (_op_err_ == AMI_SCAN_CONTINUE);				    \
	    								    \
    if ((_ami_err_ != AMI_ERROR_NO_ERROR) &&				    \
        (_ami_err_ != AMI_ERROR_END_OF_STREAM)) {			    \
        return _ami_err_;						    \
    }	    								    \
    	    								    \
    return AMI_ERROR_NO_ERROR;						    \
}

// The template for the whole AMI_scan(), with no inputs.  This is
// based on __STEMPLATE_() and could be merged into one big macro at
// the expense of having to define multiple versions of __STEMP_N()
// and __SPARM_N() to handle the case N = 0.
#define __STEMPLATE_O(out_arity)					    \
template< class SC, __STEMP_ ## out_arity (U) >				    \
AMI_err AMI_scan( SC *soper, __SPARM_ ## out_arity (U,_us_))		    \
{	    								    \
    __STSPACE_ ## out_arity (U,_u_);					    \
	    								    \
    __FSPACE(_of_,out_arity);						    \
	    								    \
    AMI_err _op_err_, _ami_err_;					    \
                                                                            \
    __CHKSTR_ ## out_arity (_us_)                                           \
    soper->initialize();						    \
	    								    \
    do {	    							    \
	    								    \
        _op_err_ = __SCALL_OP_O_ ## out_arity(soper,_u_,_of_);		    \
	    								    \
        __STS_WRITE_ ## out_arity(_u_,_us_,_of_,_ami_err_)		    \
            								    \
    } while (_op_err_ == AMI_SCAN_CONTINUE);				    \
	    								    \
    if ((_ami_err_ != AMI_ERROR_NO_ERROR) &&				    \
        (_ami_err_ != AMI_ERROR_END_OF_STREAM)) {			    \
        return _ami_err_;						    \
    }	    								    \
    	    								    \
    return AMI_ERROR_NO_ERROR;						    \
}

// The template for the whole AMI_scan(), with no outputs.
#define __STEMPLATE_I(in_arity)						    \
template< __STEMP_ ## in_arity (T), class SC >				    \
AMI_err AMI_scan( __SPARM_ ## in_arity (T,_ts_), SC *soper)		    \
{	    								    \
    __STSPACE_ ## in_arity (T,*_t_);					    \
	    								    \
    __FSPACE(_if_,in_arity);						    \
    __FSPACE(_lif_,in_arity);						    \
	    								    \
    AMI_err _op_err_, _ami_err_;					    \
	    								    \
    __CHKSTR_ ## in_arity (_ts_)                                            \
    __REWIND_ ## in_arity (_ts_);					    \
	    								    \
    soper->initialize();						    \
                                                                            \
    __SET_IF_ ## in_arity (_if_);					    \
                                                                            \
    do {	    							    \
	    								    \
        __STS_READ_ ## in_arity (_t_,_ts_,_if_,_lif_,_ami_err_)		    \
            								    \
        _op_err_ = __SCALL_OP_I_ ## in_arity (_t_,_if_,soper);		    \
	    								    \
    } while (_op_err_ == AMI_SCAN_CONTINUE);				    \
	    								    \
    if ((_ami_err_ != AMI_ERROR_NO_ERROR) &&				    \
        (_ami_err_ != AMI_ERROR_END_OF_STREAM)) {			    \
        return _ami_err_;						    \
    }	    								    \
    	    								    \
    return AMI_ERROR_NO_ERROR;						    \
}


// Finally, the templates themsleves.

__STEMPLATE(1,1); __STEMPLATE(1,2); __STEMPLATE(1,3); __STEMPLATE(1,4);
__STEMPLATE(2,1); __STEMPLATE(2,2); __STEMPLATE(2,3); __STEMPLATE(2,4);
__STEMPLATE(3,1); __STEMPLATE(3,2); __STEMPLATE(3,3); __STEMPLATE(3,4);
__STEMPLATE(4,1); __STEMPLATE(4,2); __STEMPLATE(4,3); __STEMPLATE(4,4);

__STEMPLATE_O(1); __STEMPLATE_O(2); __STEMPLATE_O(3); __STEMPLATE_O(4);

__STEMPLATE_I(1); __STEMPLATE_I(2); __STEMPLATE_I(3); __STEMPLATE_I(4);

#endif // _AMI_SCAN_MAC_H 
