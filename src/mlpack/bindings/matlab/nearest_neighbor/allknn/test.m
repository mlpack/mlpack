dataPoints = [0, 0; 1, 1; 3, 3; 0.5, 0; 1000, 0; 1001, 0];
queryPoints = [2,4; 7, 11];
k=2;

% running the emst computation with mlpack
%[distances neighbors] = allknn(dataPoints,k)

[distances neighbors] = allknn(dataPoints, k, 'queryPoints', queryPoints, 'leafSize', 10)

