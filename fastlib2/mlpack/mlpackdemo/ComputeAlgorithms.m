function ComputeAlgorithms(data_file, method, handles)

  % Dimension-reduction algorithms.
  if strcmp(method, 'PCA')==1
    data_file = ['/net/hg200/dongryel/MLPACK_test_datasets/' data_file '.csv'];
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../../contrib/tqlong/quicsvd && ./quicsvd_main --A_in=' data_file];
    system(command);
  end
  
  % Nearest neighbor algorithms.
  if strcmp(method, 'EXACT')==1
    data_file = ['/net/hg200/dongryel/MLPACK_test_datasets/' data_file '.csv'];
    data_file = '/net/hc293/dongryel/Research/fastlib2/mlpack/mlpackdemo/dummy.csv';
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../allknn && ./allknn_exe --reference_file=' data_file ' --knns=' int2str(handles.knn)];
    [status, result] = system(command);
    % Convert the list to the adjacency matrix.
    ConvertKnnResultToAdjacencyMatrix('../allknn/result.txt');
  end
  if strcmp(method, 'APPROX')==1
  end
  
  % More algorithms.
  if strcmp(method, 'EMST')==1
    data_file = ['/net/hg200/dongryel/MLPACK_test_datasets/' data_file '.csv'];
    data_file = '/net/hc293/dongryel/Research/fastlib2/mlpack/mlpackdemo/dummy.csv';
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../emst/ && ./emst_main --data=' data_file ' --output_filename=result.txt'];
    [status, result] = system(command);
    % Get the handle to the GUI plot, and plot the density.
    get(handles.axes1);
    data_matrix = load('dummy.csv');
    % Convert the list to the adjacency matrix.
    ConvertKnnResultToAdjacencyMatrix('../emst/result.txt');
    load 'adjacency_matrix.mat' adjacency_matrix;
    gplot(adjacency_matrix, data_matrix);
  end;
  if strcmp(method, 'KDE')==1    
    data_file = ['/net/hg200/dongryel/MLPACK_test_datasets/' data_file '.csv'];
    data_file = '/net/hc293/dongryel/Research/fastlib2/mlpack/mlpackdemo/dummy.csv';
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../kde/ && ./kde_cv_bin --data=' data_file ' --kde/kernel=gaussian --kde/probability=0.8 --kde/relative_error=0.1'];
    [status, result] = system(command);
    % Parse the optimal bandwidth...
    start_positions = findstr(result, 'achieved at the bandwidth value of ');
    end_positions = findstr(result, '/info/sys/node/name:R ');
    offset_length = length('achieved at the bandwidth value of ');
    optimal_bandwidth = str2double(result(start_positions(1) + offset_length:end_positions(1) - 1));
    kde_command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../kde/ && ./dualtree_kde_bin --data=' data_file ' --kde/kernel=gaussian --kde/probability=0.8 --kde/relative_error=0.1 --kde/bandwidth=' num2str(optimal_bandwidth) ' --kde/fast_kde_output=kde_output.txt --kde/loo'];
    % Run the KDE on the optimal bandwidth.
    [kde_status, kde_result] = system(kde_command)
    % Get the handle to the GUI plot, and plot the density.
    get(handles.axes1);
    data_matrix = load('dummy.csv');
    density_values = load('../kde/kde_output.txt');
    size(data_matrix)
    size(density_values)
    scatter3(data_matrix(:, 1), data_matrix(:, 2), density_values);
  end
  if strcmp(method, 'Range search')==1
      
  end
  if strcmp(method, 'NBC')==1
      
  end
  if strcmp(method, 'Decision Trees')==1
      
  end
