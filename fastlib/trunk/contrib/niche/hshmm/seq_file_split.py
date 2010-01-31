#!/usr/bin/env python

import os
import string

seqs_filename = 'training_seqs.out'
seqs_file = open(seqs_filename)
num_seqs = 200

for label in range(0, 2):
    for seq_num in range(0 + label, num_seqs, 2):
        numstring = string.zfill(seq_num, 3)
        cur_seq_filename = 'data/seq_' + numstring + '.out'
        print(cur_seq_filename)
        cur_seq_file = open(cur_seq_filename, 'w')
        
        for i in range(0, 2):
            line = seqs_file.readline()
            cur_seq_file.write(line)
        
        
        
        cur_seq_file.close()

seqs_file.close()


run_script_filename = 'run_mmf3_baumwelch'
run_script_file = open(run_script_filename, 'w')

run_script_file.write('#!/bin/csh\n\n')

mmf_path = '../../tqlong/mmf/'

for seq_num in range(0, num_seqs):
    numstring = string.zfill(seq_num, 3)
    cur_seq_filename = 'data/seq_' + numstring + '.out'
    mmf_profile_filename='profiles/est_mmf_pro_' + numstring + '.dis'
    bw_profile_filename='profiles/est_bw_pro_' + numstring + '.dis'
    run_script_file.write(mmf_path + './mmf3 --seqfile=' + cur_seq_filename + ' --numstate=3 --numsymbol=2 --profile=' + mmf_profile_filename + ' --tolerance=1e-5\n')
    run_script_file.write(mmf_path + './train --type=discrete --algorithm=baumwelch --seqfile=' + cur_seq_filename + ' --guess=' + mmf_profile_filename + ' --profile=' + bw_profile_filename + ' --maxiter=1000 --tolerance=1e-5\n\n')

run_script_file.close()
