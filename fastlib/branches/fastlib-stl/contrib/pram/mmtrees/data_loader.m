optd_q = load('optd_q.csv')';
optd_r = load('optd_r.csv')';
optd_uq = load('optd_kdtree_uq_vq3.txt');

optd_max = max(optd_r, [], 2);
optd_min = min(optd_r, [], 2);

optd_ranges = [optd_min optd_max];
optd_y = optd_uq(:,1);
