%% q4(a)
clc;clear;close all;format long
x = randn(10, 600);
w = randn(10, 1);
xTrain = x(1:10,1:100);
xTest = x(1:10,101:600);
n = randn(1,600);
yTrain = xTrain' * w + n(1:100)';
yTest = xTest' * w + n(101:600)';
yd = yTrain;
xd = xTrain';
yt = yTest;
xt = xTest';
llog = logspace(-6,3,10);


for i = 1:10
    wStar = (xd'*xd+llog(i)*100*eye(10))\xd'*yd;
    error1(i) = (wStar'*xd'*xd*wStar-2*yd'*xd*wStar+yd'*yd)/100
    error2(i) = (wStar'*xt'*xt*wStar-2*yt'*xt*wStar+yt'*yt)/500
end
figure(1)
loglog(llog, error1,'DisplayName','training set sample 100')
hold on
loglog(llog, error2,'DisplayName','test set sample 100')
legend('show')
%% q4(b)
clc;clear;close all;format long
x = randn(10, 600);
w = randn(10, 1);
xTrain = x(1:10,1:10);
xTest = x(1:10,101:600);
n = randn(1,600);
yTrain = xTrain' * w + n(1:10)';
yTest = xTest' * w + n(101:600)';
yd = yTrain;
xd = xTrain';
yt = yTest;
xt = xTest';
llog = logspace(-6,3,10);
for i = 1:10
    wStar = inv(xd'*xd+llog(i)*10*eye(10))*xd'*yd;
    error1(i) = (wStar'*xd'*xd*wStar-2*yd'*xd*wStar+yd'*yd)/10;
    error2(i) = (wStar'*xt'*xt*wStar-2*yt'*xt*wStar+yt'*yt)/500;
end
figure(2)
loglog(llog, error1,'DisplayName','training set sample 10')
hold on
loglog(llog, error2,'DisplayName','test set sample 10')
legend('show')
%% q4(c)
clc;clear;close all;format long
llog = logspace(-6,3,10);
for i = 1:10
    sum_Train_error_10 = 0;
    sum_Test_error_10 = 0;
    sum_Train_error_100 = 0;
    sum_Test_error_100 = 0;
    for j = 1:200
        x = randn(10, 600);
        w = randn(10, 1);

        xTrain10 = x(1:10,1:10);
        xTrain100 = x(1:10,1:100);
        xTest = x(1:10,101:600);

        n = randn(1,600);

        yTrain10 = xTrain10' * w + n(1:10)';
        yTrain100 = xTrain100' * w + n(1:100)';
        yTest = xTest' * w + n(101:600)';

        xT10 = xTrain10';
        yT10 = yTrain10;
        xT100 = xTrain100';
        yT100 = yTrain100;
        yt = yTest;
        xt = xTest';
        
        wStar10 = (xT10'*xT10+llog(i)*10*eye(10))\xT10'*yT10;
        wStar100 = (xT100'*xT100+llog(i)*100*eye(10))\xT100'*yT100;
        
        Train_error_10 = (wStar10'*xT10'*xT10*wStar10-2*yT10'*xT10*wStar10+yT10'*yT10)/10;
        Test_error_10 = (wStar10'*xt'*xt*wStar10-2*yt'*xt*wStar10+yt'*yt)/500;
        Train_error_100 = (wStar100'*xT100'*xT100*wStar100-2*yT100'*xT100*wStar100+yT100'*yT100)/100;
        Test_error_100 = (wStar100'*xt'*xt*wStar100-2*yt'*xt*wStar100+yt'*yt)/500;
        
        sum_Train_error_10 = sum_Train_error_10 + Train_error_10;
        sum_Test_error_10 = sum_Test_error_10 + Test_error_10;
        sum_Train_error_100 = sum_Train_error_100 + Train_error_100;
        sum_Test_error_100 = sum_Test_error_100 + Test_error_100;
        
    end
    ave_Train_error_10(i) = sum_Train_error_10/200;
    ave_Test_error_10(i) = sum_Test_error_10/200;
    ave_Train_error_100(i) = sum_Train_error_100/200;
    ave_Test_error_100(i) = sum_Test_error_100/200;
end
figure(2)
loglog(llog, ave_Train_error_10,'DisplayName','training set sample 10')
hold on
loglog(llog, ave_Test_error_10,'DisplayName','test set sample 10')
legend('show')
figure(3)
loglog(llog, ave_Train_error_100,'DisplayName','training set sample 100')
hold on
loglog(llog, ave_Test_error_100,'DisplayName','test set sample 100')
legend('show')
