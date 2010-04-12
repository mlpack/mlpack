loss = load('results/loss.dat');
nn = load('results/nn.dat');

incorrect_inds = find(loss == 1)';
correct_inds = find(loss == 0)';

for i = incorrect_inds
  test = load(sprintf('results/test_%03d.dat', i));
  training = load(sprintf('results/training_%03d.dat', nn(i,2)));
  
  map = load(sprintf('results/optimal_path_%03d.dat', i));
  
  display_mapping(test, training, map);
  fprintf('Incorrect: %dth test example mapped to %dth training example\n', ...
	  i, nn(i,2));
  pause;
end


for i = correct_inds
  test = load(sprintf('results/test_%03d.dat', i));
  training = load(sprintf('results/training_%03d.dat', nn(i,2)));
  
  map = load(sprintf('results/optimal_path_%03d.dat', i));
  
  display_mapping(test, training, map);
  fprintf('Correct: %dth test example mapped to %dth training example\n', ...
	  i, nn(i,2));
  pause;
end
