function timing = GetNaiveTiming(data_file, method)
[status, result] = system( ['wc -l ' data_file] );
root_path = GetRootPath();
fid = fopen(data_file, 'rt');
first_line = fgetl(fid);
fclose(fid);
counter = 1;
[token, rem] = strtok(first_line,',');
while ~isempty(rem)
    [token, rem] = strtok(rem,',');
    counter = counter + 1;
end;

first_blank = find(result == ' ', 1, 'first');
num_lines = str2double( result(1:first_blank - 1) );
data_files = GetDataFiles();
exact_nn_timings = [100 100 100 100 100];
decision_tree_timings = [100 100 100 100 100];
kpca_timings = [100 100 100 100 100];

num_lines = sscanf(result, '%f', 1);

timing = 1;

if strcmp(method, 'KDE') == 1
    timing = 3600 / (50000 * 50000) * (num_lines * num_lines);
end;
if strcmp(method, 'Decision Tree') == 1
    timing = 3000;
end;
if strcmp(method, 'KPCA') == 1
    polycoeff = [0.002 -0.1471 25.6272];
    timing = polyval(polycoeff, 0.5 * (counter + num_lines));   
end;
if strcmp(method, 'PCA') == 1
    polycoeff = [0.002 -0.1471 25.6272];
    timing = polyval(polycoeff, 0.5 * (counter + num_lines));
end;
if strcmp(method, 'EMST') == 1
    timing = (num_lines / 4096.0) * (num_lines / 4096.0) * (num_lines / 4096.0) * 3337;
end;
if strcmp(method, 'EXACT') == 1
    if strcmp(data_file, [root_path 'LayoutHistogram.csv']) == 1
        timing = 1105.65;
    end;
    if strcmp(data_file, [root_path 'LayoutHistogram-rnd.csv']) == 1
        timing = 21.2;
    end;
    if strcmp(data_file, [root_path 'mnist10k_test.csv']) == 1
        timing = 321;
    end;
    if strcmp(data_file, [root_path 'mnist60k_train.csv']) == 1
        timing = 8129;
    end;
end;
if strcmp(method, 'APPROX') == 1
    timing = 1000;
end;
