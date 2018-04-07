function  [W, bias, noronSayisi]  = mlptgmAA(egitimVeri, egitimHedef, ogrenmeOrani, momentum, noronSayisi, iterasyonSayisi,...
    sabirSiniri, gosterim, bias, aktivasyon)



% mlpgm: t�revsel azalma ve momentum kullanan �ok katmanl� sinir a�� e�itimi i�lemi
%
%
%��k�� Parametreleri
%         W: Giri� ve hesaplama yap�lan katmanlara ait optimize edilmi� a��rl�klar
%         bias: i� �arp�m�n s�f�r olmas�n� engelleyen parametre
%         noronSayisi: Ara katmanlar�n n�ron say�lar� dizisi
%
%Giri� Parametreleri
%         egitimVeri: egitimde kullan�lacak olan veri (do�rulama verisi bu veri �zerinden al�narak olu�turuluyor)
%         egitimHedef: egitimVeri ye ait s�n�f bilgileri
%         ogrenmeOrani: t�revsel azalma i�lemindeki ad�m miktar�
%         momentum: A��rl�k de�i�imi i�in kullan�lan optimizasyon parametresi
%         noronSayisi: Ara katman n�ron say�lar� dizisi
%         iterasyonSayisi : A��rl�k optimzasyonu i�in t�m e�itim verisinin en fazla e�itilme say�s�
%         sabirSiniri: E�itimi erken durdurma i�in belirlenen, do�rulama verisine ait ortalama toplam hata karelerin s�rekli azald��� iterasyon sayisi.
%         gosterim: Her iterasyon sonunda egitim verileri ile yap�lan a��rl�k optimazasyonlar� sonras� bu a��rl�klar ile e�itim verileri ve
%         do�rulama verilerine ait ortalama hata karelerin grafiksel g�steriminin canl� olarak izlenmesi. (1 a��k, 0 kapali)
%
%
% �rnek Kullan�m
%         mlptgm(egitimVeri, egitimHedef,.4, .01,[20, 20, 10, 30], 1000,20, 1)
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                           E��T�M                             %
% % T�revsel azalma ve momentum kullanan �ok katmanl� sinir a��  %
% %                                                              %
% %                    Apdullah Yay�k, 2016                      %
% %                    apdullahyayik@gmail.com                   %
% %                                                              %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if (numel(aktivasyon)~=numel(noronSayisi))
%     error('Belirlenen katman sayisi ve aktivasyon fonksiyonlar� say�s� uyumsuz.')
% end

warning off;
% sizeoftestVeri=size(testVeri,1);
sizeofdogrulamaVeritemp=size(egitimVeri,1)/3;

class=unique(egitimHedef);
sizeofClass=zeros(1, class);

for j=1:length(class)
    sizeofClass(j)=length(find(egitimHedef==class(j)));
end
sumsizeofClass=sum(sizeofClass);
sizeofClassdogrulamaVeri=zeros(1, length(class));
for ii=1:length(sizeofClass)
    p=sumsizeofClass/sizeofClass(ii);
    sizeofClassdogrulamaVeri(ii)=round(sizeofdogrulamaVeritemp/p);
end
dogrulamaHedef=[];
for s=1:length(class)
    dogrulamaHedef=[dogrulamaHedef, ones(1, sizeofClassdogrulamaVeri(s))*(s-1)];
end
dogrulamaHedef=dogrulamaHedef';

dogrulamaVeri=[];
for pi=1:length(class)
    temp=find(egitimHedef==class(pi));
    
    dogrulamaVeri=[dogrulamaVeri; egitimVeri(temp(1:sizeofClassdogrulamaVeri(pi)),:)];
    egitimVeri(temp(1:sizeofClassdogrulamaVeri(pi)),:)=[];
    egitimHedef(temp(1:sizeofClassdogrulamaVeri(pi)),:)=[];
end

patience=0;
W = cell(1,length(noronSayisi)+1);
W{iterasyonSayisi, length(noronSayisi)+1}=[];
for wi=1:length(noronSayisi)+1
    switch wi
        case 1
            W{1,wi}=rand(noronSayisi(wi), length(egitimVeri(1,:)))*2-1;
        case length(noronSayisi)+1
            W{1,wi}=rand(length(egitimHedef(1,:)), noronSayisi(wi-1))*2-1;
        otherwise
            W{1,wi}=rand(noronSayisi(wi), noronSayisi(wi-1))*2-1;
    end
end
m = 0;
egitimveriSayi = size(egitimVeri,1); dogrulamaveriSayisi=size(dogrulamaVeri,1); % testveriSayi = size(testVeri,1);
MSSE=zeros(1,m);
MSSEval=zeros(1,m);
H = cell(1,length(noronSayisi));
H{1, length(noronSayisi)}=[];
Hval = cell(1,length(noronSayisi));
Hval{1, length(noronSayisi)}=[];

delta = cell(iterasyonSayisi,length(noronSayisi));
delta{iterasyonSayisi, length(noronSayisi)}=[];
switch gosterim
    case 1
        figure,xlabel('iterasyon sayisi'), ylabel('ortalama hata kareler toplam�'), goodplot, hold on,
end
for m=1:iterasyonSayisi
    for i=1:egitimveriSayi
        % E�itim verisi
        I = egitimVeri(i,:)';
        D = egitimHedef(i,:)';
        % Ara katman ��k��� (H) ve ��k�� katman� ��k��� (O) hesaplanmas�
        for li=1:length(noronSayisi)+1
            switch li
                case 1
                    switch aktivasyon{1}
                        case 'sigmoid'
                            H{1,li}=sigm(W{m,li}*I+bias);
                        case 'tangentH'
                            H{1,li}=tanh(W{m,li}*I+bias);
                        case 'tangentH_opt'
                            H{1,li}=tan_h(W{m,li}*I+bias);
                        case 'ReLU'
                            H{1,li}=relu(W{m,li}*I+bias);
                        case 'linear'
                            H{1,li}=(W{m,li}*I+bias);
                    end
                case length(noronSayisi)+1
                    switch aktivasyon{end}
                        case 'sigmoid'
                            O=sigm(W{m,li}*H{1,li-1}+bias);
                        case 'tangentH'
                            O=tanh(W{m,li}*H{1,li-1}+bias);
                        case 'tangentH_opt'
                            O=tan_h(W{m,li}*H{1,li-1}+bias);
                        case 'ReLU'
                            O=relu(W{m,li}*H{1,li-1}+bias);
                        case 'linear'
                            O=(W{m,li}*H{1,li-1}+bias);
                    end
                otherwise
                    switch aktivasyon{li}
                        case 'sigmoid'
                            disp('hello')
                            H{1,li}=sigm(W{m,li}*H{1,li-1}+bias);
                        case 'tangentH'
                            H{1,li}=tanh(W{m,li}*H{1,li-1}+bias);
                        case  'tangentH_opt'
                            H{1,li}=tan_h(W{m,li}*H{1,li-1}+bias);
                        case 'ReLU'
                            H{1,li}=relu(W{m,li}*H{1,li-1}+bias);
                        case 'linear'
                            H{1,li}=(W{m,li}*H{1,li-1}+bias);
                    end
            end
        end
        % yerel t�revlerin hesaplanmas�
        %                     if delta{m,di}<=1e-5   % min gradient for early stopping
        %                         break
        %                     end
        for di=length(noronSayisi)+1:-1:1
            switch di
                case length(noronSayisi)+1
                    switch aktivasyon{end}
                        case 'sigmoid'
                            localgradient_output=O.*(1-O);
                        case 'tangentH'
                            localgradient_output=1-(O.^2);
                        case 'tangentH_opt'
                            localgradient_output = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * O.^2);
                        case 'ReLU'
                            localgradient_output =1. * (O > 0);
                        case 'linear'
                            localgradient_output=1;
                    end
                    delta{m,di}=localgradient_output.*(D-O);
                    deltaW=(ogrenmeOrani.*delta{m, di}*H{1,di-1}'); % Do�al olarak son katmanda momentum uygulanamaz !
                    W{m+1,di}=W{m,di}+deltaW;
                case 1
                    switch aktivasyon{1}
                        case 'sigmoid'
                            localgradient_input=H{1,di}.*(1-H{1,di});
                        case 'tangentH'
                            localgradient_input=1-(H{1,di}.^2);
                        case 'tangentH_opt'
                            localgradient_input = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * H{1,di}.^2);
                        case 'ReLU'
                            localgradient_input =1. * (O > 0);
                        case 'linear'
                            localgradient_input=1;
                    end
                    delta{m,di}=localgradient_input.*(W{m,di+1}.'*delta{m,di+1});
                                        deltaW=(ogrenmeOrani.*delta{m, di}*I');
%                     deltaW=momentum.*(ogrenmeOrani.*delta{m, di}*I')+(ogrenmeOrani.*delta{m, di}*I'); % h�z
                    W{m+1,di}=W{m,di}+deltaW;
                otherwise
                    switch aktivasyon{di}
                        case 'sigmoid'
                            localgradient_hidden=H{1,di}.*(1-H{1,di});
                        case 'tangentH'
                            localgradient_hidden=1-(H{1,di}.^2);
                        case 'tangentH_opt'
                            localgradient_hidden = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * H{1,di}.^2); % LeChun --> Efficient BackProp, 1998
                        case 'ReLU'
                            localgradient_hidden=1. * (O > 0);
                        case 'linear'
                            localgradient_hidden=1;
                    end
                    delta{m,di}=localgradient_hidden.*(W{m,di+1}.'*delta{m,di+1});
                                        deltaW=(ogrenmeOrani.*delta{m, di}*H{1,di-1}');
%                     deltaW=momentum.*(ogrenmeOrani.*delta{m, di-1}*H{1,di-1}')+(ogrenmeOrani.*delta{m, di}*H{1,di-1}');
                    W{m+1,di}=W{m,di}+deltaW;
            end
        end
    end
    SSErr=0;
    SSErrval=0;
    % E�itim verileri i�in Hata Kareler Toplam� (SSE)
    for i=1:egitimveriSayi
        D = egitimHedef(i,:)';
        I = egitimVeri(i,:)';
        for li=1:length(noronSayisi)+1
            switch li
                case 1
                    switch aktivasyon{1}
                        case 'sigmoid'
                            H{1,li}=sigm(W{m+1,li}*I+bias);
                        case  'tangentH'
                            H{1,li}=tanh(W{m+1,li}*I+bias);
                        case 'tangentH_opt'
                            H{1,li}=tan_h(W{m+1,li}*I+bias);
                        case 'ReLU'
                            H{1,li}=relu(W{m+1,li}*I+bias);
                        case 'linear'
                            H{1,li}=(W{m+1,li}*I+bias);
                    end
                case length(noronSayisi)+1
                    switch aktivasyon{end}
                        case 'sigmoid'
                            O=sigm(W{m+1,li}*H{1,li-1}+bias);
                        case 'tangentH'
                            O=tanh(W{m+1,li}*H{1,li-1}+bias);
                        case 'tangentH_opt'
                            O=tan_h(W{m+1,li}*H{1,li-1}+bias);
                        case 'ReLU'
                            O=relu(W{m+1,li}*H{1,li-1}+bias);
                        case 'linear'
                            O=(W{m+1,li}*H{1,li-1}+bias);
                    end
                otherwise
                    switch aktivasyon{li}
                        case 'sigmoid'
                            H{1,li}=sigm(W{m+1,li}*H{1,li-1}+bias);
                        case 'tangentH'
                            H{1,li}=tanh(W{m+1,li}*H{1,li-1}+bias);
                        case 'tangentH_opt'
                            H{1,li}=tan_h(W{m+1,li}*H{1,li-1}+bias);
                        case 'ReLU'
                            H{1,li}=relu(W{m+1,li}*H{1,li-1}+bias);
                        case 'linear'
                            H{1,li}=(W{m+1,li}*H{1,li-1}+bias);
                    end
            end
        end
        SSErr=SSErr+sum(D-O).^2;
    end
    MSSE(m)=SSErr/egitimveriSayi; % MSSE: E�itim verilerinin hata karaler toplam� ortalamas�
    switch gosterim
        case 1
            plot(m,MSSE(m),'g+'), hold on
    end
    % Do�rulama verileri i�in Hata Kareler Toplam� (SSEval)
    for id=1:dogrulamaveriSayisi
        Dval = dogrulamaHedef(id,:)';
        Ival = dogrulamaVeri(id,:)';
        for li=1:length(noronSayisi)+1
            switch li
                case 1
                    switch aktivasyon{1}
                        case 'sigmoid'
                            Hval{1,li}=sigm(W{m+1,li}*Ival+bias);
                        case 'tangentH'
                            Hval{1,li}=tanh(W{m+1,li}*Ival+bias);
                        case 'tangentH_opt'
                            Hval{1,li}=tan_h(W{m+1,li}*Ival+bias);
                        case 'ReLU'
                            Hval{1,li}=relu(W{m+1,li}*Ival+bias);
                        case 'linear'
                            Hval{1,li}=(W{m+1,li}*Ival+bias);
                    end
                case length(noronSayisi)+1
                    switch aktivasyon{end}
                        case 'sigmoid'
                            Oval=sigm(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'tangentH'
                            Oval=tanh(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'tangentH_opt'
                            Oval=tan_h(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'ReLU'
                            Oval=relu(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'linear'
                            Oval=(W{m+1,li}*Hval{1,li-1}+bias);     
                    end
                otherwise
                    switch aktivasyon{li}
                        case 'sigmoid'
                            Hval{1,li}=sigm(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'tangentH'
                            Hval{1,li}=tanh(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'tangentH_opt'
                            Hval{1,li}=tan_h(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'ReLU'
                            Hval{1,li}=relu(W{m+1,li}*Hval{1,li-1}+bias);
                        case 'linear'
                            Hval{1,li}=(W{m+1,li}*Hval{1,li-1}+bias);
                    end
            end
        end
        SSErrval=SSErrval+sum(Dval-Oval).^2;
    end
    MSSEval(m)=SSErrval/dogrulamaveriSayisi;  % MSSEval: Do�rulama verilerinin hata karaler toplam� ortalamas�
    switch gosterim
        case 1
            plot(m,MSSEval(m),'r*'), legend('egitim', 'do�rulama')
            pause(.00001)
    end
    % Erken durdurma
    if m>1
        if MSSEval(m)>MSSEval(m-1)
            patience=patience+1;
        else
            patience=0;
        end
        if patience==sabirSiniri
            save m m
            break
        end
    end
end
save m m
switch gosterim
    case 0
        figure,goodplot, plot(MSSE,'g+'), hold on, plot(MSSEval,'r*'), xlabel('iterasyon sayisi'), ylabel('ortalama hata kareler toplam�')
        legend('egitim', 'do�rulama')
end
% e�itimde ba�ar�
for i=1:egitimveriSayi
    % E�itim verisi
    I = egitimVeri(i,:)';
    D(i)= egitimHedef(i,:)';
    % Ara katman ��k��� (H) ve ��k�� katman� ��k��� (O) hesaplanmas�
    for li=1:length(noronSayisi)+1
        switch li
            case 1
                switch aktivasyon{1}
                    case 'sigmoid'
                        H{1,li}=sigm(W{m+1,li}*I+bias);
                    case 'tangentH'
                        H{1,li}=tanh(W{m+1,li}*I+bias);
                    case 'tangentH_opt'
                        H{1,li}=tan_h(W{m+1,li}*I+bias);
                    case 'ReLU'
                        H{1,li}=relu(W{m+1,li}*I+bias);
                    case 'linear'
                        H{1,li}=(W{m+1,li}*I+bias);
                end
            case length(noronSayisi)+1
                switch aktivasyon{end}
                    case 'sigmoid'
                        O(i)=sigm(W{m+1,li}*H{1,li-1}+bias);
                    case 'tangentH'
                        O(i)=tanh(W{m+1,li}*H{1,li-1}+bias);
                    case 'tangentH_opt'
                        O(i)=tan_h(W{m+1,li}*H{1,li-1}+bias);
                    case 'ReLU'
                        O(i)=relu(W{m+1,li}*H{1,li-1}+bias);
                    case 'linear'
                        O(i)=(W{m+1,li}*H{1,li-1}+bias);
                end
            otherwise
                switch aktivasyon{li}
                    case 'sigmoid'
                        H{1,li}=sigm(W{m+1,li}*H{1,li-1}+bias);
                    case 'tangentH'
                        H{1,li}=tanh(W{m+1,li}*H{1,li-1}+bias);
                    case 'tangentH_opt'
                        H{1,li}=tan_h(W{m+1,li}*H{1,li-1}+bias);
                    case 'ReLU'
                        H{1,li}=relu(W{m+1,li}*H{1,li-1}+bias);
                    case 'linear'
                        H{1,li}=(W{m+1,li}*H{1,li-1}+bias);
                end
        end
    end
end

egitimbasarimtablosu=crosstab(D, round(O))
egitimgeneldogruluk=100*(egitimbasarimtablosu(1,1)+egitimbasarimtablosu(2, 2))/sum(egitimbasarimtablosu(:))
end