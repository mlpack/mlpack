function [] = kde_classifier(train, test, classes)
  
  [N d] = size(train); 
  
  % The last column are the classes in both the train and test 
  
  C = cell(classes, 3);
  
  for i = 0:classes -1
    I = [];
    n = 0;
    
    for j = 1:N
      if train(j, d) == i
	n = n+1;
	I(n) = j;
      end
    end
    
    C{i+1, 1} = I; % members of the class
    % display(size(I));
    C{i+1, 2} = n / N; % probability of the class
    display(sprintf('%d, %f', i+1, C{i+1, 2}));
    [f C{i+1, 3}] = kde(train(I, 1:d-1)); % computing the bw
  end
  
  [N1 d1] = size(test);
  
  error = 0;
  class_error = zeros(classes,1);
  
  for j = 1:N1
    correct_class = int16(test(j,d1)) + 1;
    
    class_conditionals = zeros(classes, 1);
    
    for i = 1:classes
      temp_query = [test(j, 1:d1 -1); train(C{i, 1}, 1:d1 - 1)];
      f = kde(temp_query, C{i,3});
      if size(f,1) ~= N+1
	display(size(f));
      end
      class_conditionals(i) = log(C{i,2}) + log(f(1));
    end
    
    [max_log_class_cond, predicted_class] = ...
	max(class_conditionals);
    
    if correct_class ~= predicted_class
      error = error + 1;
      class_error(correct_class) = class_error(correct_class) + 1;
    end
  end
  
  error_p = error * 100 / N1;
  
  display(error_p);
  display(class_error);
  
