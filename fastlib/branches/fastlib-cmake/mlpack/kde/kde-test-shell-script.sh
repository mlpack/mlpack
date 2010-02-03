#/usr/bin/bash
./kde_bin --data=t1.csv --kde/kernel=gaussian --kde/bandwidth=0.01 --kde/scaling=range --kde/multiplicative_expansion --kde/fast_kde_output=fast_kde_output.txt --kde/relative_error=0.01
./fft_kde_bin --data=t1.csv --kde/bandwidth=0.01 --kde/scaling=range --kde/fft_kde_output=fft_kde_output.txt --kde/num_grid_pts_per_dim=128
./original_ifgt_bin --data=t1.csv --kde/bandwidth=0.01 --kde/scaling=range --kde/ifgt_kde_output=ifgt_kde_output.txt --kde/absolute_error=0.01
./fgt_kde_bin --data=t1.csv --kde/bandwidth=0.01 --kde/scaling=range --kde/fgt_kde_output=fgt_kde_output.txt --kde/absolute_error=0.01
