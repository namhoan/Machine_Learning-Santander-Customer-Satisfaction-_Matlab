clc
clear all
close all

% Neural network based solution.

load res1;

fprintf('\nReading test data . . . \n')
tedata = csvread('test.csv',1,0);
fprintf('\nTest data loaded. \n')

testdatanum = tedata(:,1);
testinputs = tedata(:,2:end);

res = sim(net,testinputs');
res = res';
res = res >= 0;
res = double(res);

result = uint32([testdatanum,res]);

csvwrite('ann_result.csv',result)
