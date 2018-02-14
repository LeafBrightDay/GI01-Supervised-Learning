classdef myKNN<handle
    
    properties
        X
        y
    end
    
    methods
        function obj=myKNN(X,y)
            obj.X=X;
            obj.y=y;
        end
        function y_est=eval1nn(obj,X)
            y_est=zeros(size(X,1),1);
            
            for ii=1:size(X,1)
                diff=(repmat(X(ii,:),size(obj.X,1),1)-obj.X).^2;
                dist=sum(diff,2);
                [~,id]=min(dist);
                y_est(ii)=obj.y(id(1));                
            end
        end
    end
    
end

