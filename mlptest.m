function y=mlptestAA(testVeri, W, bias, noronSayisi, aktivasyon)

% mlptest: türevsel azalma ve momentum kullanan çok katmanlý sinir aðý test
% iþlemi
%
%
%Çýkýþ Parametreleri
%         y= nöron aðýnýn çýkýþ deðeri
%
%Giriþ Parametreleri
%         testVeri: nöron aðý ile sýnýflandýrýlacak olan veri.
%         W: mlptgm fonksiyonu ile yapýlaneðitim sonucunda optimize edilmiþ olan aðýrlýklar
%         bias: iç çarpýmýn deðerinin sýfýr olmasýný engelleyen katsayý
%         noronSayisi: Ara katman nöron sayýlarý dizisi
%
%Örnek Kullaným
%         mlptest(testVeri, W, bias, [20, 10, 20, 45])
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                           TEST                               %
% % Türevsel azalma ve momentum kullanan çok katmanlý sinir aðý  %
% %                                                              %
% %                    Apdullah Yayýk, 2016                      %
% %                    apdullahyayik@gmail.com                   %
% %                                                              %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load m
testveriSayi=size(testVeri,1);
H = cell(1,length(noronSayisi));
H{1, length(noronSayisi)}=[];
for i=1:testveriSayi
    % Eðitim verisi
    I = testVeri(i,:)';
    % Ara katman çýkýþý (H) ve çýkýþ katmaný çýkýþý (O) hesaplanmasý
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