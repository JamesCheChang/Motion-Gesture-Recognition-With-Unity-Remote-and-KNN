%% Human Activity Learning Using Mobile Phone Data




%% Load Training Data

clear
[total_acc_x_train,total_acc_y_train,total_acc_z_train,trainActivity]=collect_all_data;


[trainedClassifier, validationAccuracy] = trainClassifierChatGpt(total_acc_x_train,total_acc_y_train,total_acc_z_train,trainActivity);

