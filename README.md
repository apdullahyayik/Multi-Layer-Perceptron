% mlpgm: türevsel azalma ve momentum kullanan çok katmanlı sinir ağı eğitimi işlemi
%
%
%Çıkış Parametreleri
%         W: Giriş ve hesaplama yapılan katmanlara ait optimize edilmiş ağırlıklar
%         bias: iç çarpımın sıfır olmasını engelleyen parametre
%         noronSayisi: Ara katmanların nöron sayıları dizisi
%
%Giriş Parametreleri
%         egitimVeri: egitimde kullanılacak olan veri (doğrulama verisi bu veri üzerinden alınarak oluşturuluyor)
%         egitimHedef: egitimVeri ye ait sınıf bilgileri
%         ogrenmeOrani: türevsel azalma işlemindeki adım miktarı
%         momentum: Ağırlık değişimi için kullanılan optimizasyon parametresi
%         noronSayisi: Ara katman nöron sayıları dizisi
%         iterasyonSayisi : Ağırlık optimzasyonu için tüm eğitim verisinin en fazla eğitilme sayısı
%         sabirSiniri: Eğitimi erken durdurma için belirlenen, doğrulama verisine ait ortalama toplam hata karelerin sürekli azaldığı iterasyon sayisi.
%         gosterim: Her iterasyon sonunda egitim verileri ile yapılan ağırlık optimazasyonları sonrası bu ağırlıklar ile eğitim verileri ve
%         doğrulama verilerine ait ortalama hata karelerin grafiksel gösteriminin canlı olarak izlenmesi. (1 açık, 0 kapali)
%
%
% Örnek Kullanım
%         mlptgm(egitimVeri, egitimHedef,.4, .01,[20, 20, 10, 30], 1000,20, 1)
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                           EĞİTİM                             %
% % Türevsel azalma ve momentum kullanan çok katmanlı sinir ağı  %
% %                                                              %
% %                    Apdullah Yayık, 2016                      %
% %                    apdullahyayik@gmail.com                   %
% %                                                              %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
