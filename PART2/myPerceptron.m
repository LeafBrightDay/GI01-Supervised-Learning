classdef myPerceptron<handle
    properties
        w
    end
    methods
        function obj=myPerceptron(X,y)
            obj.w=zeros(size(X,2),1);
            obj.train(X,y);
        end
        
        function train(obj,X,y)
            y_est=zeros(size(y));
            for t=1:size(X,1)
                y_est(t)=sign(X(t,:)*obj.w);
                if y_est(t)*y(t)<=0
                    obj.w=obj.w+y(t)*X(t,:)';
                else
                    obj.w=obj.w;
                end
            end
        end
        
        function y_est=evalperceptron(obj,X)
            y_est=sign(X*obj.w);
        end
    end
end