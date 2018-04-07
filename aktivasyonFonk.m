function Y=aktivasyonFonk(x, aktivasyon, durum)
%AKTIVASYONFONK: Aktivasyon fonksiyonlarını ve türevlerini hesaplar.
% eğer durum:=1--> ileri besleme
% Y: katman çıkış, x: giriş ve ağırlık iç çarpımı
% eğer durum:=0--> türev (yerel gradient)
% Y: yerel gradient, x: katman çıkış değeri
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                                              %
% %                  AKTIVASYONFONK              %
% %                  Apdullah Yayık, 2016        %
% %                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


switch durum
    case 1
        switch aktivasyon
            case 'sigmoid'
                Y=sigm(x);
            case 'tangentH'
                Y=tanh(x);
            case 'tangentH_opt'
                Y=tan_h(x);
            case 'ReLU'
                Y=relu(x);
            case 'softmax'
                Y=softmax(x);
            case 'linear'
                Y=x;
        end
    case 0
        switch aktivasyon
            case 'sigmoid'
                Y=x.*(1-x);
            case 'tangentH'
                Y=1-(x.^2);
            case 'tangentH_opt'
                Y = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * x.^2); %LeChun,1998
            case 'ReLU'
                Y =1. * (x > 0);
            case 'linear'
                Y=1;
        end
end

function Y = sigm(x)
Y = 1./(1+exp(-x));

function Y=relu(x)
Y=x .* (x > 0);

function Y=tan_h(x)
Y=1.7159*tanh(2/3.*x);

% tanh Matlab kütüphanede mevcut
% function Y=tanh(x)
% Y=(exp(x)-exp(-x))./(exp(x)+exp(-x));

function Y=softmax(x)
shiftx = x - max(x);
exps = exp(shiftx);
Y=(exps)./sum(exps);

% Y=(exp(x - max(x)))./sum(exp(x - max(x)));







