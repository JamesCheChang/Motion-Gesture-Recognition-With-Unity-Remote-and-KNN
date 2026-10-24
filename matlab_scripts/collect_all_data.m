function [total_acc_x_train,total_acc_y_train,total_acc_z_train,trainActivity]=collect_all_data
% merge all measurment 103x41 and 23 x 41 : 103+23 are number of total
% samples. 41 is the length of waveform for each sample in X/Y/Z
%load /Users/yuchang/Documents/private2/test_data/rawPhoneSensorData_test.mat
load rawPhoneSensorData_test.mat


total_acc_x_trainB=total_acc_x_train;
total_acc_y_trainB=total_acc_y_train;
total_acc_z_trainB=total_acc_z_train;
trainActivityB=trainActivity;

clear total_acc_x_train total_acc_y_train total_acc_z_train trainActivity
%load /Users/yuchang/Documents/private2/action_raw/rawPhoneSensorData_train.mat
load rawPhoneSensorData_train.mat

trainActivity=[trainActivity trainActivityB];
total_acc_x_train=[total_acc_x_train ;total_acc_x_trainB];
total_acc_y_train=[total_acc_y_train;total_acc_y_trainB];
total_acc_z_train=[total_acc_z_train;total_acc_z_trainB];


if 1
    close all

    j_m.x=[];
    j_m.y=[];
    j_m.z=[];

    r_m.x=[];
    r_m.y=[];
    r_m.z=[];

    w_m.x=[];
    w_m.y=[];
    w_m.z=[];

    k=1;
    figure;
    subplot(3,1,1),plot(total_acc_x_train(k,:));hold on;ylabel('X-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
    subplot(3,1,2),plot(total_acc_y_train(k,:));hold on;ylabel('Y-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
    subplot(3,1,3),plot(total_acc_z_train(k,:));hold on;ylabel('Z-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;

    figure;
    for k=1:size(total_acc_z_train,1)
        if trainActivity(k)=='jump'
            subplot(3,1,1),plot(total_acc_x_train(k,:));hold on;ylabel('X-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
            subplot(3,1,2),plot(total_acc_y_train(k,:));hold on;ylabel('Y-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
            subplot(3,1,3),plot(total_acc_z_train(k,:));hold on;ylabel('Z-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
            j_m.x=[j_m.x; total_acc_x_train(k,:)];
            j_m.y=[j_m.y; total_acc_y_train(k,:)];
            j_m.z=[j_m.z; total_acc_z_train(k,:)];

        end
    end

    figure;
    for k=1:size(total_acc_z_train,1)
        if trainActivity(k)=='rotate'
            subplot(3,1,1),plot(total_acc_x_train(k,:));hold on;ylabel('X-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
            subplot(3,1,2),plot(total_acc_y_train(k,:));hold on;ylabel('Y-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
            subplot(3,1,3),plot(total_acc_z_train(k,:));hold on;ylabel('Z-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;

            r_m.x=[r_m.x; total_acc_x_train(k,:)];
            r_m.y=[r_m.y; total_acc_y_train(k,:)];
            r_m.z=[r_m.z; total_acc_z_train(k,:)];


        end
    end

    figure;
    for k=1:size(total_acc_z_train,1)
        if trainActivity(k)=='wheel'
            subplot(3,1,1),plot(total_acc_x_train(k,:));hold on;ylabel('X-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
            subplot(3,1,2),plot(total_acc_y_train(k,:));hold on;ylabel('Y-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;
            subplot(3,1,3),plot(total_acc_z_train(k,:));hold on;ylabel('Z-Value','FontSize',14);xlabel('Frame Index','FontSize',14);set(gca,'FontSize',14);grid on;


            w_m.x=[w_m.x; total_acc_x_train(k,:)];
            w_m.y=[w_m.y; total_acc_y_train(k,:)];
            w_m.z=[w_m.z; total_acc_z_train(k,:)];


        end
    end

    
end
end