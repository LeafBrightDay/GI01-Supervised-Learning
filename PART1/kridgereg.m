function [ alpha] = kridgereg( K,y,gamma )
l=size(y,1);
alpha=(K+gamma*l*eye(l))\y;
end

