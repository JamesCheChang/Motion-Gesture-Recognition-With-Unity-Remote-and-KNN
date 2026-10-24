% 1. Sample Data (replace with your actual data)
% j_m
% 
% j_m = 
% 
%   struct with fields:
% 
%     x: [42×41 double]
%     y: [42×41 double]
%     z: [42×41 double]
function check_cov_matrix
load cor_data.mat
 co_j_m_x=cov1(j_m.x);

co_j_m_y=cov1(j_m.y);
co_j_m_z=cov1(j_m.z);
co_r_m_x=cov1(r_m.x);
co_r_m_y=cov1(r_m.y);
co_r_m_z=cov1(r_m.z);
co_w_m_x=cov1(w_m.x);
co_w_m_y=cov1(w_m.y);
co_w_m_z=cov1(w_m.z);
close all

global_min = min(min([co_j_m_x co_j_m_y co_j_m_z co_r_m_x co_r_m_y co_r_m_z co_w_m_x co_w_m_y co_w_m_z]));
global_max = max(max([co_j_m_x co_j_m_y co_j_m_z co_r_m_x co_r_m_y co_r_m_z co_w_m_x co_w_m_y co_w_m_z]));
common_clim = [global_min, global_max];

% 3. Visualize the matrices
figure;
imagesc(co_j_m_x); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_j_m_x');

figure;
imagesc(co_j_m_y); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_j_m_y');

figure;
imagesc(co_j_m_z); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_j_m_z');

%%
% 3. Visualize the matrices
figure;
imagesc(co_r_m_x); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_r_m_x');

figure;
imagesc(co_r_m_y); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_r_m_y');

figure;
imagesc(co_r_m_z); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_r_m_z');

%%
% 3. Visualize the matrices
figure;
imagesc(co_w_m_x); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_w_m_x');

figure;
imagesc(co_w_m_y); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit
title('co_w_m_y');

figure;
imagesc(co_w_m_z); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
%caxis(common_clim); % Set the common color axis limit

title('co_w_m_z');


figure;
subplot(3,1,1),imagesc(co_j_m_z); title('Jump','FontSize',18);% Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
subplot(3,1,2),imagesc(co_r_m_z); title('Rotate','FontSize',18);% Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
subplot(3,1,3),imagesc(co_w_m_z);title('Wheel','FontSize',18); % Visualizes matrix data as an image and scales colors
axis square tight; % Adjust axis for better visual
colorbar; % Add a color bar
end

function C=cov1(X)
X=X';
  mu = mean(X, 2);
    Xc = X - mu;

    % Step 2: covariance matrix
    C = cov(Xc.');   % transpose so rows = observations
% 1/N (X-Mean)^ (X-mean)
end
