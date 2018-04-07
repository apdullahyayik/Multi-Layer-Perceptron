function y=mlptestAA(testVeri, W, bias, noronSayisi, aktivasyon)

% mlptest: t�revsel azalma ve momentum kullanan �ok katmanl� sinir a�� test
% i�lemi
%
%
%��k�� Parametreleri
%         y= n�ron a��n�n ��k�� de�eri
%
%Giri� Parametreleri
%         testVeri: n�ron a�� ile s�n�fland�r�lacak olan veri.
%         W: mlptgm fonksiyonu ile yap�lane�itim sonucunda optimize edilmi� olan a��rl�klar
%         bias: i� �arp�m�n de�erinin s�f�r olmas�n� engelleyen katsay�
%         noronSayisi: Ara katman n�ron say�lar� dizisi
%
%�rnek Kullan�m
%         mlptest(testVeri, W, bias, [20, 10, 20, 45])
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                           TEST                               %
% % T�revsel azalma ve momentum kullanan �ok katmanl� sinir a��  %
% %                                                              %
% %                    Apdullah Yay�k, 2016                      %
% %                    apdullahyayik@gmail.com                   %
% %                                                              %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load m
testveriSayi=size(testVeri,1);
H = cell(1,length(noronSayisi));
H{1, length(noronSayisi)}=[];
for i=1:testveriSayi
    % E�itim verisi
    I = testVeri(i,:)';
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
                        Otest(i)=sigm(W{m+1,li}*H{1,li-1}+bias);
                    case 'tangentH_opt'
                        Otest(i)=tan_h(W{m+1,li}*H{1,li-1}+bias);
                    case 'ReLU'
                        Otest(i)=relu(W{m+1,li}*H{1,li-1}+bias);
                    case 'linear'
                        Otest(i)=(W{m+1,li}*H{1,li-1}+bias);
                end            
            otherwise
                switch aktivasyon{li}
                    case 'sigmoid'
                        H{1,li}=sigm(W{m+1,li}*H{1,li-1}+bias);
                    case 'tangentH'
                        H{1,li}=tanh(W{m+1,li}*H{1,li-1}+bias);
                    case  'tangentH_opt'
                        H{1,li}=tan_h(W{m+1,li}*H{1,li-1}+bias);
                    case 'ReLU'
                        H{1,li}=relu(W{m+1,li}*H{1,li-1}+bias);
                    case 'linear'
                        H{1,li}=(W{m+1,li}*H{1,li-1}+bias);
                end              
        end
    end
end

y=Otest;
end