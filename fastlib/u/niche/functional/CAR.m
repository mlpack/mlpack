function [filtered_data] = CAR(data)
%CAR Apply CAR filter to EEG data
%
%            [filtered_data] = CAR(DATA)
%            (time,numChannels,numEpochs)

% Nishant Mehta 2007

disp('applying CAR filter');

for z = 1:28
  filtered_data(:,z,:) = data(:,z,:) - (sum(data,2) / 28);
end
