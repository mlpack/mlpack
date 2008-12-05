function timing = GetWekaTiming(data_file, method)
data_files = GetDataFiles();
exact_nn_timings = [100 100 100 100 100];
decision_tree_timings = [100 100 100 100 100];
if strcmp(method, 'EXACT') == 1
    timing = 1000;
end;
if strcmp(method, 'Decision Tree') == 1
    timing = 1000;
end;
timing = 1000;