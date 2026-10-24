function [PCA1_signal,mu,V,PCA2_signal]=test_PCA(X,mu,V)
% X: 3xN accelerometer window
% rows: [acc_x; acc_y; acc_z]

if isempty(mu) & isempty(V)
    % Step 1: mean removal
    mu = mean(X, 2);
    Xc = X - mu;

    % Step 2: covariance matrix
    C = cov(Xc.');   % transpose so rows = observations
% 1/N (X-Mean)^ (X-mean)


    % Step 3: eigen-decomposition
    [V, D] = eig(C);

    % Sort eigenvalues descending
    [eigvals, idx] = sort(diag(D), 'descend');
    V = V(:, idx);

    % PCA1 direction
    v1 = V(:,1);     % 3x1 vector

    % Step 4: PCA1 time series
    PCA1_signal = v1.' * Xc;   % 1xN

     % PCA1 direction
    v2 = V(:,2);     % 3x1 vector

    PCA2_signal= v2.' * Xc;

elseif ~isempty(mu) & ~isempty(V)

    % Step 1: mean removal
  
    Xc = X - mu;

    % Step 2: covariance matrix
    C = cov(Xc.');   % transpose so rows = observations

   
    % PCA1 direction
    v1 = V(:,1);     % 3x1 vector

    % Step 4: PCA1 time series
    PCA1_signal = v1.' * Xc;   % 1xN

   % PCA1 direction
    v2 = V(:,2);     % 3x1 vector

    PCA2_signal= v2.' * Xc;


else
    error
end

PCA1_signal=PCA1_signal';
PCA2_signal=PCA2_signal';

end
