function ConvertKnnResultToAdjacencyMatrix(filename)

% Open the k-nn result file.
result = load(filename);

number_rows_in_result = size(result, 1);
self = zeros(number_rows_in_result, 3);
for i = 1:number_rows_in_result
    result(i, 1) = result(i, 1) + 1;
    result(i, 2) = result(i, 2) + 1;
    result(i, 3) = 1;
    self(i, 1) = i;
    self(i, 2) = i;
    self(i, 3) = 1;
end;
result = [result; self];
% Write the adjacency matrix.
adjacency_matrix = spconvert(result);
save adjacency_matrix adjacency_matrix;
