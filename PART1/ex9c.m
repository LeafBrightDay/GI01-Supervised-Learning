clear

% Exercise 9(c): Linear Regression with all attributes
load boston.mat;

testMSE = zeros(20,1);
trainMSE = zeros(20,1);
for i = 1:20
    [ndata, D]=size(boston);
    R = randperm(ndata);        
    boston = boston(R(1:ndata),:);

    TrainSet_size=uint32(ndata/3*2);    
    TestSet_size=ndata-TrainSet_size;

    TrainSet = boston(1:TrainSet_size,:);
    TestSet = boston(TrainSet_size+1:ndata,:);

    Train_Set_x = [TrainSet(:,1:13),ones(TrainSet_size,1)];
    Train_Set_y = TrainSet(:,14);
    Test_Set_x = [TestSet(:,1:13),ones(TestSet_size,1)];
    Test_Set_y = TestSet(:,14);
    wStar = (Train_Set_x'*Train_Set_x)\(Train_Set_x'* Train_Set_y);
    testMSE(i) = testMSE(i) + (wStar'*(Test_Set_x)'*Test_Set_x*wStar - 2*Test_Set_y'*Test_Set_x*wStar+Test_Set_y'*Test_Set_y)/double(TestSet_size);
    trainMSE(i) = trainMSE(i) +(wStar'*(Train_Set_x)'*Train_Set_x*wStar - 2*Train_Set_y'*Train_Set_x*wStar+Train_Set_y'*Train_Set_y)/double(TrainSet_size);
end

result=[mean(trainMSE) sqrt(var(trainMSE));mean(testMSE) sqrt(var(testMSE))];
disp(result)    