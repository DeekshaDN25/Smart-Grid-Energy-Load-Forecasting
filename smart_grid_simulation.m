%% Smart Grid Energy Load Forecasting - MATLAB
% Step 1: Simulate Data
% Step 2: Visualize
% Step 3: Export CSV for Python

clc; clear; close all;

%% 1. SIMULATE DATA (100 hours)
n = 100;
hours = (1:n)';

% Temperature: simple sine wave + noise
temperature = 25 + 10*sin(2*pi*hours/24) + 2*randn(n,1);

% Energy Load: depends on temperature + time + noise
energy_load = 200 + 5*(temperature - 20) + 30*sin(2*pi*hours/24) + 10*randn(n,1);

%% 2. VISUALIZE
figure;

subplot(2,1,1);
plot(hours, temperature, 'r-o', 'LineWidth', 1.5);
title('Temperature over Time');
xlabel('Hour'); ylabel('Temperature (°C)');
grid on;

subplot(2,1,2);
plot(hours, energy_load, 'b-o', 'LineWidth', 1.5);
title('Energy Load over Time');
xlabel('Hour'); ylabel('Load (MW)');
grid on;

sgtitle('Smart Grid - Simulated Data');

%% 3. EXPORT TO CSV
T = table(hours, temperature, energy_load);
writetable(T, 'smart_grid_data.csv');
disp('Data saved to smart_grid_data.csv');
