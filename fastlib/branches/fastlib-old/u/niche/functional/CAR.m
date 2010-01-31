function [filtered_data] = CAR(data)
%CAR Apply CAR filter to EEG data
%
%            [filtered_data] = CAR(DATA)
%            (time,numChannels,numEpochs)

% Nishant Mehta 2007

disp('applying CAR filter');

num_channels = size(data, 2);

for z = 1:num_channels
  filtered_data(:,z,:) = data(:,z,:) - (sum(data,2) / num_channels);
end
