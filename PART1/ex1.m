clear

Dims = 1;
datanumber = 600;
repeat = 200;
noiseSTD= 0.034543;
test = 500;
result=zeros(2);


for i = 1:2
     total_meantrain =0;
     total_meantest= 0;
    for j = 1:repeat
         
        x = randn(Dims,datanumber);
        w = randn(Dims,1);
        noise=randn(Dims,datanumber);
        y = w' * x + noise; 
    
        if i ==1
            sample = 10;

        else
            sample = 100;
  
        end
            X=x(:,1:sample)';
            Y=y(:,1:sample)';
            Xt=x(:,101:datanumber)';
            Yt=y(:,101:datanumber)';
            
            X1=X';
            Xt1=Xt';
            Y1=Y';
            Yt1=Yt';
            
            W = esti(X,X1,Y); 
            W1=W';
            
            meantrain=cost(W,W1,X,X1,Y,Y1,sample);
            meantest=cost(W,W1,Xt,Xt1,Yt,Yt1,test);
            total_meantrain = total_meantrain + meantrain;
            total_meantest = total_meantest + meantest;  
        
    end
result(i,1) = total_meantrain/repeat;
result(i,2) = total_meantest/repeat;

end     
result
     
function[W] = esti(X,X1,Y)
    W = (X1*X)\X1*(Y);
end

function[mean] = cost(W,W1,X,X1,Y,Y1,a)
    mean =(W1*X1*X*W-2*Y1*X*W+Y1*Y)/a;
end


