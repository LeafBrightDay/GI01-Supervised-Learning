function  mse  = dualcost( K,y,alpha )
l=size(y,1);
mse=(K*alpha-y)'*(K*alpha-y);
mse=mse/l;
end

