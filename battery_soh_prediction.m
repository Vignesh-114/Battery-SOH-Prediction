%% ============================================================
% AI-BASED BATTERY STATE-OF-HEALTH (SOH) PREDICTION FOR EV
% Complete MATLAB Project
% Author: Vignesh
%
% Models:
% 1. Linear Regression
% 2. Decision Tree
% 3. SVM Regression
% 4. Ensemble Regression
% 5. LSTM Neural Network
%
% ============================================================

clc;
clear;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' AI-BASED BATTERY SOH PREDICTION FOR ELECTRIC VEHICLES\n');
fprintf('============================================================\n');

%% ============================================================
% STEP 1 - GENERATE BATTERY DATA
% ============================================================

rng(1);

numCycles = 500;

Cycle = (1:numCycles)';

% Rated battery capacity
ratedCapacity = 2.0;       % Ah

% Battery capacity degradation
Capacity = ratedCapacity ...
    - 0.00075 .* Cycle ...
    - 0.00000035 .* Cycle.^2;

% Measurement noise
Capacity = Capacity + ...
    0.008 .* randn(numCycles,1);

% Avoid unrealistic capacity values
Capacity(Capacity <= 0) = 0.1;

% Battery voltage
Voltage = 4.2 ...
    - 0.00035 .* Cycle ...
    + 0.015 .* randn(numCycles,1);

% Battery current
Current = 1.5 ...
    + 0.10 .* randn(numCycles,1);

% Battery temperature
Temperature = 25 ...
    + 0.015 .* Cycle ...
    + 0.5 .* randn(numCycles,1);

%% ============================================================
% STEP 2 - CALCULATE SOH
% ============================================================

SOH = (Capacity ./ ratedCapacity) .* 100;

% Keep SOH between 0 and 100
SOH(SOH > 100) = 100;
SOH(SOH < 0) = 0;

%% ============================================================
% STEP 3 - CREATE BATTERY DATA TABLE
% ============================================================

BatteryData = table( ...
    Cycle, ...
    Voltage, ...
    Current, ...
    Temperature, ...
    Capacity, ...
    SOH);

fprintf('\n');
fprintf('============================================================\n');
fprintf(' FIRST 10 BATTERY DATA\n');
fprintf('============================================================\n');

disp(BatteryData(1:10,:));

%% ============================================================
% STEP 4 - SAVE DATASET
% ============================================================

writetable(BatteryData, 'battery_data.csv');

fprintf('\nBattery dataset saved as battery_data.csv\n');

%% ============================================================
% STEP 5 - CAPACITY DEGRADATION GRAPH
% ============================================================

figure;

plot(Cycle, Capacity, ...
    'LineWidth', 2);

xlabel('Battery Cycle');
ylabel('Capacity (Ah)');

title('EV Battery Capacity Degradation');

grid on;

%% ============================================================
% STEP 6 - SOH DEGRADATION GRAPH
% ============================================================

figure;

plot(Cycle, SOH, ...
    'LineWidth', 2);

xlabel('Battery Cycle');
ylabel('State of Health (%)');

title('EV Battery SOH Degradation');

grid on;

%% ============================================================
% STEP 7 - PREPARE AI INPUTS
% ============================================================

% Features
X = [ ...
    Cycle, ...
    Voltage, ...
    Current, ...
    Temperature, ...
    Capacity];

% Target
Y = SOH;

%% ============================================================
% STEP 8 - TRAINING AND TESTING SPLIT
% ============================================================

cv = cvpartition(size(X,1), ...
    'HoldOut', 0.20);

trainIndex = training(cv);
testIndex = test(cv);

XTrain = X(trainIndex,:);
YTrain = Y(trainIndex);

XTest = X(testIndex,:);
YTest = Y(testIndex);

fprintf('\n');
fprintf('============================================================\n');
fprintf(' DATA SPLIT\n');
fprintf('============================================================\n');

fprintf('Training samples : %d\n', length(YTrain));
fprintf('Testing samples  : %d\n', length(YTest));

%% ============================================================
% STEP 9 - NORMALIZATION
% ============================================================

mu = mean(XTrain);

sigma = std(XTrain);

% Prevent division by zero
sigma(sigma == 0) = 1;

XTrainNorm = ...
    (XTrain - mu) ./ sigma;

XTestNorm = ...
    (XTest - mu) ./ sigma;

%% ============================================================
% STEP 10 - LINEAR REGRESSION
% ============================================================

fprintf('\nTraining Linear Regression...\n');

linearModel = fitrlinear( ...
    XTrainNorm, ...
    YTrain);

YPredLinear = predict( ...
    linearModel, ...
    XTestNorm);

%% ============================================================
% STEP 11 - DECISION TREE
% ============================================================

fprintf('Training Decision Tree...\n');

treeModel = fitrtree( ...
    XTrainNorm, ...
    YTrain);

YPredTree = predict( ...
    treeModel, ...
    XTestNorm);

%% ============================================================
% STEP 12 - SVM REGRESSION
% ============================================================

fprintf('Training SVM...\n');

svmModel = fitrsvm( ...
    XTrainNorm, ...
    YTrain, ...
    'KernelFunction', 'gaussian');

YPredSVM = predict( ...
    svmModel, ...
    XTestNorm);

%% ============================================================
% STEP 13 - ENSEMBLE REGRESSION
% ============================================================

fprintf('Training Ensemble Model...\n');

ensembleModel = fitrensemble( ...
    XTrainNorm, ...
    YTrain, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 100);

YPredEnsemble = predict( ...
    ensembleModel, ...
    XTestNorm);

%% ============================================================
% STEP 14 - MODEL PERFORMANCE
% ============================================================

% -----------------------------
% Linear Regression
% -----------------------------

MAE_Linear = mean( ...
    abs(YTest - YPredLinear));

RMSE_Linear = sqrt(mean( ...
    (YTest - YPredLinear).^2));

R2_Linear = 1 - ...
    sum((YTest - YPredLinear).^2) / ...
    sum((YTest - mean(YTest)).^2);


% -----------------------------
% Decision Tree
% -----------------------------

MAE_Tree = mean( ...
    abs(YTest - YPredTree));

RMSE_Tree = sqrt(mean( ...
    (YTest - YPredTree).^2));

R2_Tree = 1 - ...
    sum((YTest - YPredTree).^2) / ...
    sum((YTest - mean(YTest)).^2);


% -----------------------------
% SVM
% -----------------------------

MAE_SVM = mean( ...
    abs(YTest - YPredSVM));

RMSE_SVM = sqrt(mean( ...
    (YTest - YPredSVM).^2));

R2_SVM = 1 - ...
    sum((YTest - YPredSVM).^2) / ...
    sum((YTest - mean(YTest)).^2);


% -----------------------------
% Ensemble
% -----------------------------

MAE_Ensemble = mean( ...
    abs(YTest - YPredEnsemble));

RMSE_Ensemble = sqrt(mean( ...
    (YTest - YPredEnsemble).^2));

R2_Ensemble = 1 - ...
    sum((YTest - YPredEnsemble).^2) / ...
    sum((YTest - mean(YTest)).^2);

%% ============================================================
% STEP 15 - MODEL COMPARISON TABLE
% ============================================================

Model = {
    'Linear Regression'
    'Decision Tree'
    'SVM'
    'Ensemble'
    };

MAE = [
    MAE_Linear
    MAE_Tree
    MAE_SVM
    MAE_Ensemble
    ];

RMSE = [
    RMSE_Linear
    RMSE_Tree
    RMSE_SVM
    RMSE_Ensemble
    ];

R2 = [
    R2_Linear
    R2_Tree
    R2_SVM
    R2_Ensemble
    ];

Results = table( ...
    Model, ...
    MAE, ...
    RMSE, ...
    R2);

fprintf('\n');
fprintf('============================================================\n');
fprintf(' AI MODEL COMPARISON\n');
fprintf('============================================================\n');

disp(Results);

%% ============================================================
% STEP 16 - FIND BEST CLASSICAL ML MODEL
% ============================================================

[~, bestIndex] = min(RMSE);

bestModelName = Model{bestIndex};

fprintf('\n');
fprintf('============================================================\n');
fprintf(' BEST CLASSICAL ML MODEL\n');
fprintf('============================================================\n');

fprintf('Best Model : %s\n', bestModelName);

fprintf('RMSE       : %.4f\n', RMSE(bestIndex));

fprintf('R2         : %.4f\n', R2(bestIndex));

%% ============================================================
% STEP 17 - ACTUAL VS ALL ML PREDICTIONS
% ============================================================

figure;

plot(YTest, 'o-', ...
    'LineWidth', 1.5);

hold on;

plot(YPredLinear, '*-', ...
    'LineWidth', 1.2);

plot(YPredTree, 'x-', ...
    'LineWidth', 1.2);

plot(YPredSVM, 's-', ...
    'LineWidth', 1.2);

plot(YPredEnsemble, 'd-', ...
    'LineWidth', 1.2);

xlabel('Test Sample');

ylabel('SOH (%)');

title('Actual vs Predicted Battery SOH');

legend( ...
    'Actual SOH', ...
    'Linear Regression', ...
    'Decision Tree', ...
    'SVM', ...
    'Ensemble');

grid on;

%% ============================================================
% STEP 18 - SELECT BEST CLASSICAL ML PREDICTION
% ============================================================

if bestIndex == 1

    bestPrediction = YPredLinear;

elseif bestIndex == 2

    bestPrediction = YPredTree;

elseif bestIndex == 3

    bestPrediction = YPredSVM;

else

    bestPrediction = YPredEnsemble;

end

%% ============================================================
% STEP 19 - BEST MODEL GRAPH
% ============================================================

figure;

plot(YTest, 'o-', ...
    'LineWidth', 2);

hold on;

plot(bestPrediction, '*-', ...
    'LineWidth', 2);

xlabel('Test Sample');

ylabel('SOH (%)');

title(['Best Model: ', bestModelName]);

legend( ...
    'Actual SOH', ...
    'Predicted SOH');

grid on;

%% ============================================================
% STEP 20 - PREDICTION ERROR
% ============================================================

predictionError = ...
    YTest - bestPrediction;

figure;

plot(predictionError, ...
    'LineWidth', 1.5);

xlabel('Test Sample');

ylabel('Prediction Error (%)');

title('SOH Prediction Error');

yline(0, '--');

grid on;

%% ============================================================
% STEP 21 - LSTM DATA PREPARATION
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' LSTM NEURAL NETWORK\n');
fprintf('============================================================\n');

fprintf('Preparing sequential battery data...\n');

% Use chronological split for LSTM
numTrainLSTM = floor(0.80 * numCycles);

X_LSTM_Train = X(1:numTrainLSTM,:);
Y_LSTM_Train = Y(1:numTrainLSTM);

X_LSTM_Test = X(numTrainLSTM+1:end,:);
Y_LSTM_Test = Y(numTrainLSTM+1:end);

% Normalize using training data only
muLSTM = mean(X_LSTM_Train);

sigmaLSTM = std(X_LSTM_Train);

sigmaLSTM(sigmaLSTM == 0) = 1;

X_LSTM_Train = ...
    (X_LSTM_Train - muLSTM) ./ sigmaLSTM;

X_LSTM_Test = ...
    (X_LSTM_Test - muLSTM) ./ sigmaLSTM;

% Convert matrix to sequence format
%
% LSTM input format:
% Features x Time Steps

XTrainSequence = X_LSTM_Train';

YTrainSequence = Y_LSTM_Train';

XTestSequence = X_LSTM_Test';

YTestSequence = Y_LSTM_Test';

%% ============================================================
% STEP 22 - CREATE LSTM NETWORK
% ============================================================

fprintf('Creating LSTM network...\n');

layers = [
    
    sequenceInputLayer(5)
    
    lstmLayer(50, ...
        'OutputMode','sequence')
    
    fullyConnectedLayer(25)
    
    reluLayer
    
    fullyConnectedLayer(1)
    
    regressionLayer
    
    ];

%% ============================================================
% STEP 23 - LSTM TRAINING OPTIONS
% ============================================================

options = trainingOptions( ...
    'adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.005, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% ============================================================
% STEP 24 - TRAIN LSTM
% ============================================================

fprintf('Training LSTM network...\n');

try

    lstmModel = trainNetwork( ...
        XTrainSequence, ...
        YTrainSequence, ...
        layers, ...
        options);

catch ME

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf(' LSTM TRAINING ERROR\n');
    fprintf('============================================================\n');

    fprintf('%s\n', ME.message);

    fprintf('\nLSTM requires MATLAB Deep Learning Toolbox.\n');

    lstmModel = [];

end

%% ============================================================
% STEP 25 - LSTM PREDICTION
% ============================================================

if ~isempty(lstmModel)

    fprintf('Predicting SOH using LSTM...\n');

    YPredLSTM = predict( ...
        lstmModel, ...
        XTestSequence);

    % Convert output to column vector
    YPredLSTM = YPredLSTM(:);

    YTestLSTM = YTestSequence(:);

    %% ========================================================
    % STEP 26 - LSTM PERFORMANCE
    % ========================================================

    MAE_LSTM = mean( ...
        abs(YTestLSTM - YPredLSTM));

    RMSE_LSTM = sqrt(mean( ...
        (YTestLSTM - YPredLSTM).^2));

    R2_LSTM = 1 - ...
        sum((YTestLSTM - YPredLSTM).^2) / ...
        sum((YTestLSTM - mean(YTestLSTM)).^2);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf(' LSTM PERFORMANCE\n');
    fprintf('============================================================\n');

    fprintf('LSTM MAE  : %.4f %%\n', MAE_LSTM);

    fprintf('LSTM RMSE : %.4f %%\n', RMSE_LSTM);

    fprintf('LSTM R2   : %.4f\n', R2_LSTM);

    %% ========================================================
    % STEP 27 - ALL MODEL COMPARISON INCLUDING LSTM
    % ========================================================

    Model_All = {
        'Linear Regression'
        'Decision Tree'
        'SVM'
        'Ensemble'
        'LSTM'
        };

    MAE_All = [
        MAE_Linear
        MAE_Tree
        MAE_SVM
        MAE_Ensemble
        MAE_LSTM
        ];

    RMSE_All = [
        RMSE_Linear
        RMSE_Tree
        RMSE_SVM
        RMSE_Ensemble
        RMSE_LSTM
        ];

    R2_All = [
        R2_Linear
        R2_Tree
        R2_SVM
        R2_Ensemble
        R2_LSTM
        ];

    FinalResults = table( ...
        Model_All, ...
        MAE_All, ...
        RMSE_All, ...
        R2_All);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf(' FINAL AI MODEL COMPARISON\n');
    fprintf('============================================================\n');

    disp(FinalResults);

    %% ========================================================
    % STEP 28 - FIND BEST OVERALL MODEL
    % ========================================================

    [~, bestOverallIndex] = ...
        min(RMSE_All);

    bestOverallModel = ...
        Model_All{bestOverallIndex};

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf(' BEST OVERALL MODEL\n');
    fprintf('============================================================\n');

    fprintf('Best Model : %s\n', ...
        bestOverallModel);

    fprintf('RMSE       : %.4f\n', ...
        RMSE_All(bestOverallIndex));

    fprintf('R2         : %.4f\n', ...
        R2_All(bestOverallIndex));

    %% ========================================================
    % STEP 29 - LSTM ACTUAL VS PREDICTED GRAPH
    % ========================================================

    figure;

    plot(YTestLSTM, 'o-', ...
        'LineWidth', 2);

    hold on;

    plot(YPredLSTM, '*-', ...
        'LineWidth', 2);

    xlabel('Test Sample');

    ylabel('SOH (%)');

    title('LSTM Actual vs Predicted Battery SOH');

    legend( ...
        'Actual SOH', ...
        'LSTM Predicted SOH');

    grid on;

    %% ========================================================
    % STEP 30 - LSTM ERROR
    % ========================================================

    LSTMError = ...
        YTestLSTM - YPredLSTM;

    figure;

    plot(LSTMError, ...
        'LineWidth', 1.5);

    xlabel('Test Sample');

    ylabel('Prediction Error (%)');

    title('LSTM SOH Prediction Error');

    yline(0, '--');

    grid on;

else

    fprintf('\nLSTM was skipped because the required toolbox\n');
    fprintf('or another LSTM requirement was not available.\n');

end

%% ============================================================
% STEP 31 - NEW BATTERY SOH PREDICTION
% ============================================================

newCycle = 300;

newVoltage = 4.05;

newCurrent = 1.50;

newTemperature = 30;

newCapacity = 1.82;

newBattery = [
    newCycle ...
    newVoltage ...
    newCurrent ...
    newTemperature ...
    newCapacity];

% Normalize new battery
newBatteryNorm = ...
    (newBattery - mu) ./ sigma;

%% ============================================================
% STEP 32 - PREDICT USING BEST CLASSICAL MODEL
% ============================================================

if bestIndex == 1

    predictedSOH = predict( ...
        linearModel, ...
        newBatteryNorm);

elseif bestIndex == 2

    predictedSOH = predict( ...
        treeModel, ...
        newBatteryNorm);

elseif bestIndex == 3

    predictedSOH = predict( ...
        svmModel, ...
        newBatteryNorm);

else

    predictedSOH = predict( ...
        ensembleModel, ...
        newBatteryNorm);

end

%% ============================================================
% STEP 33 - BATTERY HEALTH CLASSIFICATION
% ============================================================

if predictedSOH >= 90

    condition = 'Excellent';

elseif predictedSOH >= 80

    condition = 'Good';

elseif predictedSOH >= 70

    condition = 'Moderate';

elseif predictedSOH >= 60

    condition = 'Poor';

else

    condition = 'Critical';

end

%% ============================================================
% STEP 34 - FINAL RESULT
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              EV BATTERY SOH PREDICTION\n');
fprintf('============================================================\n');

fprintf('Battery Cycle       : %d\n', newCycle);

fprintf('Battery Voltage     : %.2f V\n', ...
    newVoltage);

fprintf('Battery Current     : %.2f A\n', ...
    newCurrent);

fprintf('Battery Temperature : %.2f C\n', ...
    newTemperature);

fprintf('Battery Capacity    : %.2f Ah\n', ...
    newCapacity);

fprintf('\n');

fprintf('Best Classical Model: %s\n', ...
    bestModelName);

fprintf('Predicted SOH       : %.2f %%\n', ...
    predictedSOH);

fprintf('Battery Condition   : %s\n', ...
    condition);

fprintf('============================================================\n');

%% ============================================================
% STEP 35 - PROJECT SUMMARY
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                  PROJECT COMPLETED\n');
fprintf('============================================================\n');

