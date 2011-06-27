function [acc pY_X] = multiClassRodeoClassifier(train, test, classes)

[nTrain d] = size(train);
numClasses = size(classes, 2);
nTest = size(test, 1);

pY_X = [];
nT = zeros(1, numClasses);

time = 0;

for i=1:numClasses

  class_ind = find(train(:, d) == classes(i));

  classTrain = train(class_ind, 1:d-1);

  pY = length(class_ind) / nTrain;

  tic;
  pX_Y = local_rodeo(classTrain, test(:, 1:d-1));
  time = time + toc;
  
  pY_X(:, i) = pY * pX_Y;
  nT(i) = length(find(test(:, d) == classes(i)));

end

[tmp, yI] = max(pY_X');

c = zeros(1, numClasses);
for i = 1:nTest
  if test(i, d) == classes(yI(i))
    c(yI(i)) = c(yI(i)) + 1;
  end
end

acc = sum(c) / nTest;
display(c);
display(nT);
display(acc);
display(time);
