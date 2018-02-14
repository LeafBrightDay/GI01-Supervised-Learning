nmax=100;
trials=20;
complexities=zeros(nmax,trials);
g_er_size=5e3;
parfor jj=1:trials % Perform trials
    jj
    m=1;
    for n=1:nmax % Compute complexity for each n
        m=max(1,round(m*0.5)); % start with half previous m        
        while 1
            [X,y]=genData(g_er_size,n);  % Generate large chunk of data
            PerceptronClassifier=myPerceptron(X(1:m,:),y(1:m,:));      % Initiallize a classfier on m samples  
            yy=PerceptronClassifier.evalperceptron(X);% Evaluate on g_er_size samples
            g=sum(yy~=y)/g_er_size  ; %compute generalization
            if g<0.1
                break
            else
                m=m+1;
            end
        end
        complexities(n,jj)=m;
    end
end

varr=sqrt(var(complexities,0,2));
mean_c=sum(complexities,2)./trials;
errorbar( mean_c , varr)
save('Perceptron','complexities')

function [X,y] = genData( m,n )
% Sample m samples from {-1 1}^n set. 
X=2*((rand(m,n)-0.5)>0)-1;
y=X(:,1);
end