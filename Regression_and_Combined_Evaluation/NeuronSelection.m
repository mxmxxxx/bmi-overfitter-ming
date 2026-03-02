clc
clear
close all
load('monkeydata_training.mat');

% Select neurons to analyze
neuron_ids = 1:98;

% Get unique reaching angles
reaching_angles = [30, 70, 110, 150, 190, 230, 310, 350];

% Compute mean firing rate per angle
mean_firing_rates = zeros(length(neuron_ids), length(reaching_angles));
std_firing_rates = zeros(length(neuron_ids), length(reaching_angles));

for j = 1:length(neuron_ids)
    neuron_id = neuron_ids(j);
    for k = 1:length(reaching_angles)
        num_trials = size(trial, 1);
        firing_rates = zeros(num_trials, 1);
        for n = 1:num_trials
            spikes = trial(n, k).spikes(neuron_id, :);
            firing_rates(n) = sum(spikes) / length(spikes) * 1000; % Spikes per second
        end
        mean_firing_rates(j, k) = mean(firing_rates); % mean of firing rate of 1 neuron at 1 angle for 100 trails
        std_firing_rates(j, k) = std(firing_rates);
    end
end

% Plot tuning curves
figure; hold on;
for j = 1:length(neuron_ids)
    errorbar(reaching_angles, mean_firing_rates(j, :), std_firing_rates(j, :), '-o', 'LineWidth', 2);
end
xlabel('Reaching Angle (degrees)');
ylabel('Mean Firing Rate (spikes/sec)');
title('Tuning Curves of Selected Neurons');
legend(arrayfun(@(x) ['Neuron ', num2str(x)], neuron_ids, 'UniformOutput', false));
hold off;

%%
num_neurons = 98;
num_angles = 8;
num_top = 20;  % number of top neurons to select per angle

% Preallocate a matrix to store the best 20 neuron indices for each angle.
% Each column corresponds to one reaching angle.
best20NeuronIndices_all = zeros(num_top, num_angles);

for angle_idx = 1:num_angles
    % Sort neurons by mean firing rate for the current reaching angle in descending order.
    [~, sortIndices] = sort(mean_firing_rates(:, angle_idx), 'descend');
    
    % Save the top 20 indices for this angle into the corresponding column.
    best20NeuronIndices_all(:, angle_idx) = sortIndices(1:num_top);
    
    % Plot the tuning curves (firing rate vs. reaching angle) for these 20 neurons.
    figure;
    hold on;
    for i = 1:num_top
        idx = best20NeuronIndices_all(i, angle_idx);
        errorbar(reaching_angles, mean_firing_rates(idx, :), std_firing_rates(idx, :), '-o', 'LineWidth', 2);
    end
    selected_neurons = neuron_ids(best20NeuronIndices_all(:, angle_idx));
    xlabel('Reaching Angle (degrees)');
    ylabel('Mean Firing Rate (spikes/sec)');
    title(sprintf('Tuning Curves of Top 20 Neurons (Selected for Angle %d°)', reaching_angles(angle_idx)));
    legend(arrayfun(@(x) ['Neuron ', num2str(x)], selected_neurons, 'UniformOutput', false), 'Location', 'Best');
    hold off;
end

% Optionally, display the best20NeuronIndices_all matrix
disp('Best 20 Neuron Indices for Each Angle (columns correspond to angles):');
disp(best20NeuronIndices_all);


%%
num_neurons = 98;
num_angles = 8;
num_top = 20;  % number of top neurons to select per angle

% Preallocate a matrix to store the worst 20 neuron indices for each angle.
% Each column corresponds to one reaching angle.
worst20NeuronIndices_all = zeros(num_top, num_angles);

for angle_idx = 1:num_angles
    % Sort neurons by mean firing rate for the current reaching angle in descending order.
    [~, sortIndices] = sort(mean_firing_rates(:, angle_idx), 'ascend');
    
    % Save the top 20 indices for this angle into the corresponding column.
    worst20NeuronIndices_all(:, angle_idx) = sortIndices(1:num_top);
    
    % Plot the tuning curves (firing rate vs. reaching angle) for these 20 neurons.
    figure;
    hold on;
    for i = 1:num_top
        idx = worst20NeuronIndices_all(i, angle_idx);
        errorbar(reaching_angles, mean_firing_rates(idx, :), std_firing_rates(idx, :), '-o', 'LineWidth', 2);
    end
    selected_neurons = neuron_ids(worst20NeuronIndices_all(:, angle_idx));
    xlabel('Reaching Angle (degrees)');
    ylabel('Mean Firing Rate (spikes/sec)');
    title(sprintf('Tuning Curves of Top 20 Neurons (Selected for Angle %d°)', reaching_angles(angle_idx)));
    legend(arrayfun(@(x) ['Neuron ', num2str(x)], selected_neurons, 'UniformOutput', false), 'Location', 'Best');
    hold off;
end

% Optionally, display the best20NeuronIndices_all matrix
disp('Worst 20 Neuron Indices for Each Angle (columns correspond to angles):');
disp(worst20NeuronIndices_all);