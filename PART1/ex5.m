clear

Dims = 10;
datanumber = 600;
repeat = 200;
noiseSTD=0.034543;
noise=randn(Dims,datanumber);
test = 500;

result=zeros(0,0,0);
total_meantrain =0;
total_meantest= 0;
Utotal_meantrain =0;
Utotal_meantest= 0;
  for gamma = 1:1:10
    for j = 1:repeat  
        x = randn(Dims,datanumber);
        w = randn(Dims,1);
        y = w' * x + noise(j); 
        sample = 20;
            X=x(:,1:sample)';
            Y=y(1,1:sample)';
            Xt=x(:,101:datanumber)';
            Yt=y(1,101:datanumber)';
            
            X1=X';
            Xt1=Xt';
            Y1=Y';
            Yt1=Yt';
            
            UW = esti(X,X1,Y); 
            W = RResti(sample,gamma,X,X1,Y); 
            UW1=UW';
            W1=W';
            
            Umeantest=cost(UW,UW1,Xt,Xt1,Yt,Yt1,sample);
            meantrain=cost(W,W1,X,X1,Y,Y1,sample);
            meantest=cost(W,W1,Xt,Xt1,Yt,Yt1,test);
             
            total_meantrain = total_meantrain + meantrain;
            total_meantest = total_meantest + meantest;
            Utotal_meantest = Utotal_meantest + Umeantest;
          
    end
    result(1,gamma) = Utotal_meantest/repeat;
    result(2,gamma) = total_meantest/repeat;
    result(3,gamma) = total_meantrain/repeat;
  end    

plotdata(result)

function plotdata(result)
    
train = reshape(result(1:2,3,1:10),2,10);
linear = reshape(result(1:2,1,1:10),2,10);
ridge = reshape(result(1:2,2,1:10),2,10);

semilogy(-6:3,ridge(1,:),'LineWidth',2,'Color','r');
hold on;
semilogy(-6:3,linear(1,:),'LineWidth',2,'Color','g');
hold on;
semilogy(-6:3,train(1,:),'LineWidth',2,'Color','b');
hold off;
xlabel('log(\gamma)');
ylabel('log(error)');
legend('100 sample ridge','100 sample linear','100 sample train');
figure('Color',[1 1 1]);
semilogy(-6:3,ridge(2,:),'LineWidth',2,'Color','r');
hold on;
semilogy(-6:3,linear(2,:),'LineWidth',2,'Color','g');
hold on;
semilogy(-6:3,train(2,:),'LineWidth',2,'Color','b');
hold off;
xlabel('log(\gamma)');
ylabel('log(error)');
legend('10 sample ridge','10 sample linear','10 sample train');

end
  
function[W] = esti(X,X1,Y)
    W = (X1*X)\X1*(Y);
end

function[W] = RResti(sample,gamma,X,X1,Y)
    W = ((X1*X+sample*(10^(gamma-7))).*eye(10))\X1*Y;
end

function[mean] = cost(W,W1,X,X1,Y,Y1,a)
    mean =(W1*X1*X*W-2*Y1*X*W+Y1*Y)/a;
end