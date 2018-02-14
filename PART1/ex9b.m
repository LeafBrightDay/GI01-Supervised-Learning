clear

% Exercise 9(b): Linear Regression with single attributes
load boston.mat;

testMSE = zeros(20,13);
trainMSE = zeros(20,13);

for j = 1:20
    [ndata, D]=size(boston);
    R = randperm(ndata);        
    boston = boston(R(1:ndata),:);

    TrainSet_size=uint32(ndata/3*2);    
    TestSet_size=ndata-TrainSet_size;

    TrainSet = boston(1:TrainSet_size,:);
    TestSet = boston(TrainSet_size+1:ndata,:);
    for i = 1:13
        Train_Set_x = [TrainSet(:,i),ones(TrainSet_size,1)];
        Train_Set_y = TrainSet(:,14);
        Test_Set_x = [TestSet(:,i),ones(TestSet_size,1)];
        Test_Set_y = TestSet(:,14);
        wStar = (Train_Set_x'*Train_Set_x)\(Train_Set_x'* Train_Set_y);
        testMSE(j,i) = testMSE(j,i) + (wStar'*(Test_Set_x)'*Test_Set_x*wStar - 2*Test_Set_y'*Test_Set_x*wStar+Test_Set_y'*Test_Set_y)/double(TestSet_size);
        trainMSE(j,i) = trainMSE(j,i) +(wStar'*(Train_Set_x)'*Train_Set_x*wStar - 2*Train_Set_y'*Train_Set_x*wStar+Train_Set_y'*Train_Set_y)/double(TrainSet_size);
    end
end
disp('The MSE of the train set for 13 different attributes are: ');
num2str(mean(trainMSE))
disp('The MSE of the test set for 13 different attributes are: ');
num2str(mean(testMSE))

num2str(sqrt(var(trainMSE)))
num2str(sqrt(var(testMSE)))
