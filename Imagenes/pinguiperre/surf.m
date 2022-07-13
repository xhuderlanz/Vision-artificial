clc; clear all; close all;

%1. Leer la imagen de referencia de objeto de interés
I1 = imread('Minion (1).jpeg');
Ir = rgb2gray(I1);
figure, imshow(Ir), title('Goma referencia')

%2. Leer la imagen de objeto en una escena desordenada
I2 = imread('Minion (2).jpeg');
Id = rgb2gray(I2);
figure, imshow(Id), title('Goma por ahí')

%3. Detectar feature points
rP = detectSURFFeatures(Ir);
dP = detectSURFFeatures(Id);
figure,imshow(Ir),title('100 Strongest Feature Points from Box Image');
hold on;
plot(selectStrongest(rP, 100));
figure,imshow(Id),title('300 Strongest Feature Points from Scene Image');
hold on;
plot(selectStrongest(dP, 300));

%4. Extracción de descriptores
[rF, rP] = extractFeatures(Ir, rP);
[dF, dP] = extractFeatures(Id, dP);

%5. Hacer el match usando los descriptores
rM = matchFeatures(rF,dF); %Puntos de la goma
mrp = rP(rM(:, 1), :); %Puntos match de la goma en imagen de referencia
mdp = dP(rM(:, 2), :); %Puntos match de la goma en imagen desordenada
figure;
showMatchedFeatures(Ir, Id, mrp, mdp, 'montage');
title('Puntos match(con error)');

%Filtrado de falsos positivos por agrupación
k=5 ; %Número de clusters
%matchedPoints.Location entrega una matriz con las coordenadas de los
%puntos match en la imagen 1 (la de las 9 galletas)
[Idx,C]=kmeans(mdp.Location,k)

%Idx nos dice a qué cluster pertenece cada punto y C los centoides de cada
%cluster
Grupos=zeros(1,k); %Inicialización de vector que guarda la cantidad de puntos en cada cluster

for j=1:k
    Grupoj=length(find(Idx==j));
    Grupos(j)=Grupoj;
end

Grupo_dominante=find(Grupos==max(Grupos)); %Número del cluster con más puntos

%Eliminación
Puntos_buenos=find(Idx==Grupo_dominante); %índices de los puntos que quedaron en el cluster más grande
%Se cambian los puntos match por lo nuevos ya encontrados. En vez de poner
%todas las filas de matchedPoints, se utilizan las que indique
%Puntos_buenos
mdp=mdp(Puntos_buenos,:); 
mrp=mrp(Puntos_buenos,:);

figure;showMatchedFeatures(Ir,Id,mrp,mdp,'montage');