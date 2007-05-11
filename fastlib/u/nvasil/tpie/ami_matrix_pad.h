// Copyright (c) 1994 Darren Vengroff
//
// File: ami_matrix_pad.h
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 12/11/94
//
// $Id: ami_matrix_pad.h,v 1.8 2004/08/12 12:35:30 jan Exp $
//
#ifndef _AMI_MATRIX_PAD_H
#define _AMI_MATRIX_PAD_H

// Get definitions for working with Unix and Windows
#include <portability.h>
// Get definition of AMI_scan_object class.
#include <ami_scan.h>

// This is a scan management object designed to pad a rows by cols
// matrix with zeroes so that is becomes an (i * block_extent) by (j *
// block_extent) matrix where i and j are integers as small as
// possible.

template<class T>
class AMI_matrix_pad : AMI_scan_object {
private:
    TPIE_OS_OFFSET cur_row, cur_col;
    TPIE_OS_OFFSET orig_rows, orig_cols;
    TPIE_OS_OFFSET final_rows, final_cols;
public:
    AMI_matrix_pad(TPIE_OS_OFFSET rows, TPIE_OS_OFFSET cols,
                    TPIE_OS_OFFSET block_extent);
    virtual ~AMI_matrix_pad();
    AMI_err initialize(void);
    AMI_err operate(const T &in, AMI_SCAN_FLAG *sfin,
                    T *out, AMI_SCAN_FLAG *sfout);
};

template<class T>
AMI_matrix_pad<T>::AMI_matrix_pad(TPIE_OS_OFFSET rows, TPIE_OS_OFFSET cols,
                                  TPIE_OS_OFFSET block_extent)

{
    orig_rows = rows;
    orig_cols = cols;
    final_rows = block_extent * (((orig_rows - 1) / block_extent) + 1);
    final_cols = block_extent * (((orig_cols - 1) / block_extent) + 1);
}

template<class T>
AMI_matrix_pad<T>::~AMI_matrix_pad()
{
}

template<class T>
AMI_err AMI_matrix_pad<T>::initialize(void)
{
    cur_col = cur_row = 0;
    return AMI_ERROR_NO_ERROR;
}

template<class T>
AMI_err AMI_matrix_pad<T>::operate(const T &in, AMI_SCAN_FLAG *sfin,
                                   T *out, AMI_SCAN_FLAG *sfout)
{
    AMI_err ae;
    
    // If we are within the bounds of the original matrix, simply copy.
    if ((cur_col < orig_cols) && (cur_row < orig_rows)) {
        *out = in;
        *sfout = true;
        ae = AMI_SCAN_CONTINUE;
    } else {
        // Don't take the input.
        *sfin = false;
        // If we are not completely done then write padding.
        if ((*sfout = (cur_row < final_rows))) {
            *out = (T)0;
            ae = AMI_SCAN_CONTINUE;
        } else {
            tp_assert(cur_row == final_rows, "Too many rows.");
            ae = AMI_SCAN_DONE;
        }
    }

    // Increment the column.
    cur_col = (cur_col + 1) % final_cols;

    // Increment the row if needed.
    if (!cur_col) {
        cur_row++;
    }

    return ae;
}




// This is a scan management object designed to unpad a rows by cols
// matrix that was padded by a an object of type scan_matrix_pad.

template<class T>
class AMI_matrix_unpad : AMI_scan_object {
private:
    TPIE_OS_OFFSET cur_row, cur_col;
    TPIE_OS_OFFSET orig_rows, orig_cols;
    TPIE_OS_OFFSET final_rows, final_cols;
public:
    AMI_matrix_unpad(TPIE_OS_OFFSET rows, TPIE_OS_OFFSET cols,
                      TPIE_OS_OFFSET block_extent);
    virtual ~AMI_matrix_unpad();
    AMI_err initialize(void);
    AMI_err operate(const T &in, AMI_SCAN_FLAG *sfin,
                    T *out, AMI_SCAN_FLAG *sfout);
};

template<class T>
AMI_matrix_unpad<T>::AMI_matrix_unpad(TPIE_OS_OFFSET rows, TPIE_OS_OFFSET cols,
                                      TPIE_OS_OFFSET block_extent)

{
    orig_rows = rows;
    orig_cols = cols;
    final_rows = block_extent * (((orig_rows - 1) / block_extent) + 1);
    final_cols = block_extent * (((orig_cols - 1) / block_extent) + 1);
}

template<class T>
AMI_matrix_unpad<T>::~AMI_matrix_unpad()
{
}

template<class T>
AMI_err AMI_matrix_unpad<T>::initialize(void)
{
    cur_col = cur_row = 0;
    return AMI_ERROR_NO_ERROR;
}

template<class T>
AMI_err AMI_matrix_unpad<T>::operate(const T &in, AMI_SCAN_FLAG *sfin,
                                     T *out, AMI_SCAN_FLAG *sfout)
{
    AMI_err ae;
    
    // If we are within the bounds of the original matrix, simply copy.
    if ((cur_col < orig_cols) && (cur_row < orig_rows)) {
        *out = in;
        *sfout = true;
        ae = AMI_SCAN_CONTINUE;
    } else {
        // Don't write anything.
        *sfout = false;

        // If we are not completely done then skip padding.
        if ((*sfin = (cur_row < final_rows))) {
            ae = AMI_SCAN_CONTINUE;
        } else {
            tp_assert(cur_row == final_rows, "Too many rows.");
            ae = AMI_SCAN_DONE;
        }
    }

    // Increment the column.
    cur_col = (cur_col + 1) % final_cols;

    // Increment the row if needed.
    if (!cur_col) {
        cur_row++;
    }

    return ae;        
}

#endif // _AMI_MATRIX_PAD_H 
