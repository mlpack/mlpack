function ComputeAlgorithms(data_file, method, hObject, handles)

  % Dimension-reduction algorithms.
  if strcmp(method, 'PCA')==1
    output_filename = [data_file(1:length(data_file) - 4) '_pca_output.csv'];
    os_separator_positions = find(output_filename == '/');
    last_position = os_separator_positions(length(os_separator_positions)) + 1;
    output_filename = ['/net/hc293/dongryel/Research/fastlib2/mlpack/mlpackdemo/' output_filename(last_position:length(output_filename))];
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../../contrib/tqlong/quicsvd && ./quicsvd_main --A_in=' data_file ' --SVT_out=' output_filename ' --relErr=0.1'];
    [status, result] = system(command);
    % Add the resulting output file to the list.
    set(handles.data_file, 'String', [ get(handles.data_file, 'String') ; { output_filename }]);
    % Plot the dimension reduced dataset.
    data_matrix = load(output_filename);
    data_matrix = data_matrix(:, 1:2);
    plot(data_matrix(:, 1), data_matrix(:, 2), '.');
    csvwrite(output_filename, data_matrix);
    zoom on;
    % Change the dataset to the resulting output file.
    handles.data_file = output_filename;
    set(handles.current_dataset, 'String', handles.data_file);
    guidata(hObject, handles);
  end;
  if strcmp(method, 'MVU')==1
    % Currently, MVU doesn't do anything but display the precomputed MVU
    % result.
    data_matrix = load('/net/hg200/dongryel/MLPACK_test_datasets/LayoutHistogram-rnd_MVU_5.csv');
    [U, S, V] = svd(data_matrix, 'econ');
    U = U(:, 1:2);
    S = S(1:2, 1:2);
    dimension_reduced = U * S;
    plot(dimension_reduced(:, 1), dimension_reduced(:, 2), '.');
    zoom on;
  end;
  
  % Nearest neighbor algorithms.
  if strcmp(method, 'EXACT')==1
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../allknn && ./allknn_exe --reference_file=' data_file ' --knns=' int2str(handles.knn1)];
    [status, result] = system(command);
    % Convert the list to the adjacency matrix.
    ConvertKnnResultToAdjacencyMatrix('../allknn/result.txt', handles.knn1);
  end
  if strcmp(method, 'APPROX')==1
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../../contrib/pram/approx_nn && ./main_dual --q=' data_file ' --r=' data_file ' --ann/knns=' int2str(handles.knn1) ' --doapprox'];
    [status, result] = system(command);
    ConvertKnnResultToAdjacencyMatrix('../../contrib/pram/approx_nn/result.txt', handles.knn1);
  end
  
  % More algorithms.
  if strcmp(method, 'EMST')==1
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../emst/ && ./emst_main --data=' data_file ' --output_filename=result.txt'];
    %[status, result] = system(command);
    system(command)
    % Get the handle to the GUI plot, and plot the density.
    data_matrix = load(data_file);
    % Convert the list to the adjacency matrix.
    ConvertKnnResultToAdjacencyMatrix('../emst/result.txt', handles.knn1);
    load 'adjacency_matrix.mat' adjacency_matrix;
    gplot(adjacency_matrix, data_matrix);
    zoom on;
  end;
  if strcmp(method, 'KPCA')==1
    % Generate a kernel matrix (a random sample), then run SVD.
    [status, result] = system( ['wc -l ' data_file] );
    first_blank = find(result == ' ', 1, 'first');
    numlines = str2double( result(1:first_blank - 1) );
    rate = numlines / 1000;
    % Currently hard-coded the bandwidth...
    bandwidth = handles.bandwidth1;
    kernel_matrix_file = ['/net/hc293/dongryel/Research/fastlib2/contrib/tqlong/quicsvd/' ...
      data_file(find(data_file == '/', 1, 'last') + 1:length(data_file) - 4) '_kernel_matrix_bandwidth_' num2str(bandwidth) '.txt'];
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../../contrib/tqlong/quicsvd && ./gen_kernel_matrix --data=' data_file ' --rate=' num2str(rate) ' --output=' kernel_matrix_file ' --bandwidth=' num2str(bandwidth)];
    [status, result] = system(command);    
    output_filename = [kernel_matrix_file(1:length(kernel_matrix_file) - 4) '_pca_output.csv'];
    os_separator_positions = find(output_filename == '/');
    last_position = os_separator_positions(length(os_separator_positions)) + 1;
    output_filename = ['/net/hc293/dongryel/Research/fastlib2/mlpack/mlpackdemo/' output_filename(last_position:length(output_filename))];
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../../contrib/tqlong/quicsvd && ./quicsvd_main --A_in=' kernel_matrix_file ' --SVT_out=' output_filename ' --relErr=0.1'];
    [status, result] = system(command);
    % Add the resulting output file to the list.
    set(handles.data_file1, 'String', [ get(handles.data_file1, 'String') ; { output_filename }]);
    % Plot the dimension reduced dataset.
    data_matrix = load(output_filename);
    data_matrix = data_matrix(:, 1:2);
    csvwrite(output_filename, data_matrix);
    if size(data_matrix, 2) >= 2
      plot(data_matrix(:, 1), data_matrix(:, 2), '.');
    else
      plot(data_matrix(:, 1), zeros(size(data_matrix, 1), 1), '.');
    end;
    zoom on;
    % Change the dataset to the resulting output file.
    handles.data_file = output_filename;
    set(handles.current_dataset, 'String', handles.data_file);
    guidata(hObject, handles);
  end;
  if strcmp(method, 'KDE')==1
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
    data_matrix = load(data_file);
    density_values = load('../kde/kde_output.txt');
    [U, S, V] = svd(data_matrix, 'econ');
    U = U(:, 1:2);
    S = S(1:2, 1:2);
    data_matrix = U * S;
    scatter3(data_matrix(:, 1), data_matrix(:, 2), density_values);
  end
  if strcmp(method, 'Range search')==1
    data_file
    command = ['setenv LD_LIBRARY_PATH /usr/lib/gcc/x86_64-redhat-linux5E/4.1.2 && cd ../range_search/ && ./ortho_range_search_bin --data=' data_file];
    num_search_windows = size(handles.lower_ranges, 2);
    handles.lower_ranges = handles.lower_ranges';
    handles.upper_ranges = handles.upper_ranges';
    csvwrite('../range_search/lower.csv', handles.lower_ranges);
    csvwrite('../range_search/upper.csv', handles.upper_ranges);
    [status, result] = system(command);
    % Read in the boolean membership vector.
    result_vectors = load('../range_search/results.txt');
    membership_vectors = CreateMembershipVectors(result_vectors);
    % Get the handle to the GUI plot and plot the range search result.
    data_matrix = load(data_file);
    hold off;
    plot(0);
    plot(data_matrix(:, 1), data_matrix(:, 2), '.');
    hold on;
    color_ordering = 'grcmyk';
    for i = 1:num_search_windows
        % Draw the search window, and plot the points that fall with in it.
        rectangle('Position', [handles.lower_ranges(i, 1), handles.lower_ranges(i, 2), ...
            handles.upper_ranges(i, 1) - handles.lower_ranges(i, 1),...
            handles.upper_ranges(i, 2) - handles.lower_ranges(i, 2)]);
        plot_color = [color_ordering(i) '*'];
        plot(data_matrix(membership_vectors{i}, 1), data_matrix(membership_vectors{i}, 2), plot_color);
    end;
    zoom on;
    hold off;
    % Clear the search windows after a search.
    handles.lower_ranges = [];
    handles.upper_ranges = [];
    guidata(hObject, handles);
  end
  if strcmp(method, 'NBC')==1
      
  end
  if strcmp(method, 'Decision Trees')==1
      
  end
