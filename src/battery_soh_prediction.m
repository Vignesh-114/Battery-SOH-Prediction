%% Battery State of Health (SOH) Prediction
% Author: Vignesh S
% Description:
% Battery SOH analysis and prediction using MATLAB

clear;
clc;
close all;

%% 1. Load Battery Dataset

dataFile = '../data/battery_data.csv';

if ~isfile(dataFile)
    error('Dataset not found: %s', dataFile);
end

data = readtable(dataFile);

disp('Battery Dataset:');
disp(data);

%% 2. Display Dataset Information

disp('Dataset Variables:');
disp(data.Properties.VariableNames);

disp('Dataset Size:');
disp(size(data));

%% 3. Basic Data Cleaning

data = rmmissing(data);

fprintf('Number of samples after cleaning: %d\n', height(data));

%% 4. Identify Battery Parameters

% Modify these variable names according to your CSV file.

cycle = data.Cycle;
capacity = data.Capacity;

%% 5. Define Rated Battery Capacity

ratedCapacity = max(capacity);

%% 6. Calculate State of Health

SOH = (capacity ./ ratedCapacity) * 100;

%% 7. Display SOH

fprintf('\nBattery SOH Range:\n');
fprintf('Minimum SOH: %.2f %%\n', min(SOH));
fprintf('Maximum SOH: %.2f %%\n', max(SOH));

%% 8. Plot Battery Capacity Degradation

figure;

plot(cycle, capacity, 'LineWidth', 1.5);

xlabel('Cycle Number');
ylabel('Battery Capacity');
title('Battery Capacity Degradation');
grid on;

saveas(gcf, '../results/battery_capacity.png');

%% 9. Plot SOH

figure;

plot(cycle, SOH, 'LineWidth', 1.5);

xlabel('Cycle Number');
ylabel('State of Health (%)');
title('Battery State of Health');
grid on;

saveas(gcf, '../results/soh_prediction.png');

%% 10. Simple SOH Prediction Model

X = cycle;
Y = SOH;

model = fitlm(X, Y);

predictedSOH = predict(model, X);

%% 11. Actual vs Predicted SOH

figure;

plot(cycle, SOH, 'LineWidth', 1.5);
hold on;

plot(cycle, predictedSOH, '--', 'LineWidth', 1.5);

xlabel('Cycle Number');
ylabel('SOH (%)');

title('Actual vs Predicted Battery SOH');

legend('Actual SOH', 'Predicted SOH');

grid on;

saveas(gcf, '../results/actual_vs_predicted.png');

%% 12. Model Performance

error = SOH - predictedSOH;

RMSE = sqrt(mean(error.^2));
MAE = mean(abs(error));

fprintf('\nPrediction Performance:\n');
fprintf('RMSE = %.4f %%\n', RMSE);
fprintf('MAE  = %.4f %%\n', MAE);

%% 13. Completion Message

fprintf('\nBattery SOH prediction completed successfully.\n');
