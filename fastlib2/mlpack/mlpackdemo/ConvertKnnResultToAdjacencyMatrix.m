function ConvertKnnResultToAdjacencyMatrix(filename, knn)

% Open the k-nn result file.
result = load(filename);

number_rows_in_result = size(result, 1);
number_of_points = int32(number_rows_in_result / knn);
self = zeros(number_of_points, 3);
for i = 1:number_rows_in_result
    result(i, 1) = result(i, 1) + 1;
    result(i, 2) = result(i, 2) + 1;
    result(i, 3) = 1;
end;
for i = 1:number_of_points
    self(i, 1) = i;
    self(i, 2) = i;
    self(i, 3) = i;
end;
result = [result; self];
% Write the adjacency matrix.
adjacency_matrix = spconvert(result);
save adjacency_matrix adjacency_matrix;
