function timing = GetWekaTiming(data_file, method)
data_files = GetDataFiles();
exact_nn_timings = [100 100 100 100 100];
decision_tree_timings = [100 100 100 100 100];
if strcmp(method, 'EXACT') == 1
    for i = 1:length(data_files)
        if strcmp(data_file, data_files(i)) == 1
            timing = exact_nn_timings(i);
            break;
        end;
    end;
end;
if strcmp(method, 'Decision Tree') == 1
    for i = 1:length(data_files)
        if strcmp(data_file, data_files(i)) == 1
            timing = decision_tree_timings(i);
            break;
        end;
    end;
end;
