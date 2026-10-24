% Example script to demonstrate the use of reconstructHexFromFragments function

clear;
clc;

%% 1. Define Inputs
% Create a structure to hold all the necessary _f fragments.
% Note: The assembly order requires m1 through m8.
fragments.m1_f = '1010101001010101';
fragments.m2_f = '0000000000000000';
fragments.m3_f = '1110110110110111';
fragments.m4_f = '0001001001001000';
fragments.m5_f = '0001001001001111';
fragments.m6_f = '1110110110110000';
fragments.m7_f = '0000111100001010';
% We create m8_f by reversing the original m8 ('1010111100001111')
fragments.m8_f = '1111000011110101'; 

%% 2. Call the function
% Pass the structure to the function and get the result.
reconstructed_hex = reconstructHexFromFragments(fragments);

%% 3. Display the result
fprintf('The reconstructed hex value is: %s\n', reconstructed_hex);
