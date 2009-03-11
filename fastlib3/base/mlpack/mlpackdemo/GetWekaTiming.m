function timing = GetWekaTiming(data_file, method)
data_files = GetDataFiles();
timing = eps;
root_path=GetRootPath();
if strcmp(method, 'EXACT') == 1
    if strcmp(data_file, [root_path 'LayoutHistogram.csv']) == 1
        timing = 99.52;
    end;
    if strcmp(data_file, [root_path 'LayoutHistogram-rnd.csv']) == 1
        timing = 9.9;
    end;
    if strcmp(data_file, [root_path 'mnist10k_test.csv']) == 1
        timing = 3505.32;
    end;
    if strcmp(data_file, [root_path 'mnist60k_train.csv']) == 1
        timing = 108000;
    end;
end;
if strcmp(method, 'Decision Tree') == 1
    timing = 1000;
end;