clear all;

% Exercise 9(a)£ºNaive Regression
load boston.mat;
trainMSE=zeros(20,1);
testMSE=zeros(20,1);
for i = 1:20
    [ndata, D]=size(boston);
    R = randperm(ndata);        
    boston = boston(R(1:ndata),:);
    TrainSet_size=uint32(ndata/3*2);    
    TestSet_size=ndata-TrainSet_size;
    TrainSet = boston(1:TrainSet_size,:);
    TestSet = boston(TrainSet_size+1:ndata,:);

    Training_out = ones(TrainSet_size,1);
    Test_out = ones(TestSet_size,1);

    w = (Training_out'*Training_out)\Training_out'*TrainSet(:,14);

    trainMSE(i) = (TrainSet(:,14)-Training_out*w)'*(TrainSet(:,14)-Training_out*w)/double(TrainSet_size);
    testMSE(i) = (TestSet(:,14)-Test_out*w)'*(TestSet(:,14)-Test_out*w)/double(TestSet_size);
end

disp(['The MSE of the train set is ',num2str(mean(trainMSE))]);
disp(['The MSE of the test set is ',num2str(mean(testMSE))]);

result=[mean(trainMSE) sqrt(var(trainMSE));mean(testMSE) sqrt(var(testMSE))];
disp(result)