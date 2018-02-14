classdef myLinear<handle
    properties
        w
    end
    methods
        function obj=myLinear(X,y)
            obj.w=zeros(size(X,2),1);
            obj.train(X,y);
        end
        
        function train(obj,X,y)           
            obj.w = pinv(X)*y;
        end
        
        function y_est=evallinear(obj,X)                  
                y_est=sign(X*obj.w);           
        end
    end
end