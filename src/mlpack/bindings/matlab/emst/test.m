dataPoints = [0, 0; 1, 1; 3, 3; 0.5, 0; 1000, 0; 1001, 0];

% running the emst computation with mlpack
result = emst(dataPoints,'method','boruvka')

% running the emst computation with mlpack
result = emst(dataPoints)

% naive
result = emst(dataPoints,'method','naive')
