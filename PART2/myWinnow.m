classdef myWinnow<handle
    
    properties
        w
    end
    
    methods
        function obj=myWinnow(X,y)
            obj.w=ones(size(X,2),1)';
            X(X<0)=0;
            y(y<0)=0;
            obj.train(X,y);
        end
        
        function train(obj,X,y)
            y_est=zeros(size(y));
            n=size(X,2);
            for ii=1:size(X,1)
                y_est(ii)=double(X(ii,:)*obj.w'>=n);
                if y_est(ii)~=y(ii)                   
                    d1=(y(ii)-y_est(ii))*X(ii,:);
                    d2=2.^d1;
                    obj.w=obj.w.*d2;
                end
            end
        end
        
        function y_est=evalwinnow(obj,X)
            X(X<0)=0;
            n=size(X,2);
            y_est=double(X*obj.w'>=n);
            y_est(y_est<0.5)=-1;
        end
    end
    
end

