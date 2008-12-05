function timing = GetNaiveTiming(data_file, method)
[status, result] = system( ['wc -l ' data_file] );
first_blank = find(result == ' ', 1, 'first');
num_lines = str2double( result(1:first_blank - 1) );
data_files = GetDataFiles();
exact_nn_timings = [100 100 100 100 100];
decision_tree_timings = [100 100 100 100 100];
kpca_timings = [100 100 100 100 100];
if strcmp(method, 'KDE') == 1
    timing = 3600 / (50000 * 50000) * (num_lines * num_lines);
end;
if strcmp(method, 'Decision Tree') == 1
    timing = 3000;
end;
if strcmp(method, 'KPCA') == 1
    timing = 1800 / (4656 * 4656 * 4656) * (num_lines * num_lines * num_lines);
end;
if strcmp(method, 'PCA') == 1
    timing = 1800 / (4656 * 4656 * 4656) * (num_lines * num_lines * num_lines);
end;
