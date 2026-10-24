function [trainedClassifier, validationAccuracy] = trainClassifier(total_acc_x_train,total_acc_y_train,total_acc_z_train,trainActivity)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% Custom implementation of a 1-NN classifier (Euclidean, standardized)
% that mimics the interface of the Classification Learner exported code,
% but WITHOUT using fitcknn / crossval / kfoldPredict / kfoldLoss.
%
% Input:
%   trainingData: a table with predictor columns and a categorical
%   response column named 'activity'.
%
% Output:
%   trainedClassifier: struct with fields:
%       - predictFcn: function handle for predictions on a table
%       - CustomKNN: internal struct with training data + standardization
%       - RequiredVariables, About, HowToPredict
%
%   validationAccuracy: scalar in [0,1] giving 7-fold CV accuracy
%
% Usage:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T);
%   yfit = trainedClassifier.predictFcn(Tnew);


% ----------------------------------------------------
% 5. Manual 7-fold cross-validation (no crossval/kfold)
% ----------------------------------------------------
X=[total_acc_x_train,total_acc_y_train,total_acc_z_train];
L=41;
K = 7;
N = size(X, 1);

% Deterministic fold assignment: 1,2,...,K,1,2,... (no random)
foldIndex = mod((0:N-1)', K) + 1;

validationPredictions = trainActivity;  % preallocate with correct type
response = trainActivity;
for k = 1:K
    testMask  = (foldIndex == k);
    trainMask = ~testMask;

    Xtrain = X(trainMask, :);
    Ytrain = response(trainMask);

    Xtest = X(testMask, :);

    % Train model on K-1 folds
    foldModel = trainCustomKNN1NN(Xtrain, Ytrain,L);

    % Predict on the held-out fold
    Yhat = predictCustomKNN1NN(foldModel, Xtest,L);

    validationPredictions(testMask) = Yhat;
end

% Classification accuracy (same scale as 1 - kfoldLoss(...,'ClassifError'))
validationAccuracy = mean(validationPredictions == response)



end % function trainClassifier

function [X,Y]=getData(Xraw,L,extraM)
% generate mean and sigma ,PCA from raw data
% generate sub space Y for training (new)
total_acc_x_train=Xraw(:,1:L);
total_acc_y_train=Xraw(:,L+1:2*L);
total_acc_z_train=Xraw(:,2*L+1:3*L);
rawSensorDataTrain = table(...
    total_acc_x_train, total_acc_y_train, total_acc_z_train);

T_mean = varfun(@Wmean, rawSensorDataTrain);
T_stdv = varfun(@Wstd,rawSensorDataTrain);
%T_pca  = varfun(@Wpca1,rawSensorDataTrain);
% T_pca {'Wpca1_total_acc_x_train'}    {'Wpca1_total_acc_y_train'}    {'Wpca1_total_acc_z_train'}

if isempty(extraM)
    [Wpca1_total_acc_x_train,mu_x,V_x,Wpca2_total_acc_x_train]=test_PCA(total_acc_x_train',[],[]);
    [Wpca1_total_acc_y_train,mu_y,V_y,Wpca2_total_acc_y_train]=test_PCA(total_acc_y_train',[],[]);
    [Wpca1_total_acc_z_train,mu_z,V_z,Wpca2_total_acc_z_train]=test_PCA(total_acc_z_train',[],[]);
else
    [Wpca1_total_acc_x_train,mu_x,V_x,Wpca2_total_acc_x_train]=test_PCA(total_acc_x_train',extraM.mu_x,extraM.V_x);
    [Wpca1_total_acc_y_train,mu_y,V_y,Wpca2_total_acc_y_train]=test_PCA(total_acc_y_train',extraM.mu_y,extraM.V_y);
    [Wpca1_total_acc_z_train,mu_z,V_z,Wpca2_total_acc_z_train]=test_PCA(total_acc_z_train',extraM.mu_z,extraM.V_z);

end

T_pca=table(Wpca1_total_acc_x_train,Wpca1_total_acc_y_train,Wpca1_total_acc_z_train);

X = [T_mean, T_stdv, T_pca];
Y.Wpca1_total_acc_x_train=Wpca1_total_acc_x_train;
Y.Wpca1_total_acc_y_train=Wpca1_total_acc_y_train;
Y.Wpca1_total_acc_z_train=Wpca1_total_acc_z_train;
Y.mu_x=mu_x;
Y.mu_y=mu_y;
Y.mu_z=mu_z;
Y.V_x=V_x;
Y.V_y=V_y;
Y.V_z=V_z;

end

% =========================================================
% Local function: TRAIN a 1-NN classifier with standardize
% =========================================================
function model = trainCustomKNN1NN(Xraw, Y,L)

[Xt,extra]=getData(Xraw,L,[]);
X= table2array(Xt);
% X: N x D numeric matrix
% Y: N x 1 categorical labels

% Standardization parameters
mu = mean(X, 1);
sigma = std(X, 0, 1);

% Avoid division by zero for constant features
sigma(sigma == 0) = 1;

% Standardize training data
Xstd = (X - mu) ./ sigma;

% Store into model struct
model.XTrain = Xstd;
model.YTrain = Y;
model.Mu     = mu;
model.Sigma  = sigma;

% Optionally store class names
model.ClassNames = categories(Y);
model.extra=extra;
end

% =========================================================
% Local function: PREDICT using 1-NN (Euclidean distance)
% =========================================================
function Yhat = predictCustomKNN1NN(model, XnewRaw,L)

Xnewt=getData(XnewRaw,L,model.extra);
Xnew = table2array(Xnewt);
% Xnew: M x D numeric matrix (unstandardized predictors)
% Yhat: M x 1 categorical predictions

% Standardize using training mean/std
Xstd = (Xnew - model.Mu) ./ model.Sigma;

% Replace any NaNs that might occur (e.g., missing values) with 0
Xstd(~isfinite(Xstd)) = 0;

Xtrain = model.XTrain;
Ytrain = model.YTrain;

M = size(Xstd, 1);
Yhat = repmat(Ytrain(1), M, 1);  % preallocate with categorical type

for i = 1:M
    xi = Xstd(i, :);

    % Squared Euclidean distances to all training points
    diffs = Xtrain - xi;
    d2 = sum(diffs.^2, 2);

    % 1-NN: pick closest
    [~, idxMin] = min(d2);
    Yhat(i) = Ytrain(idxMin);
end

end
