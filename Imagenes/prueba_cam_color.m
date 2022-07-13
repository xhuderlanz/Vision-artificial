
clc;clear all; close all;
cam=webcam(1);
wb = waitbar(0,'-','Name','Espera..','CreateCancelBtn','delete(gcbf)');
i=0;
while true
    img0=snapshot(cam);
    img=imsubtract(img0(:,:,1),rgb2gray(img0));
    bw=im2bw(img,0.13);
    bw=medfilt2(bw);
    bw=imopen(bw,strel('disk',1));
    bw=bwareaopen(bw,3000);%elimina area menor a 3000px
    bw=imfill(bw,'holes');
    [L N]=bwlabel(bw);
    %-----------------regionprops------------------
    prop=regionprops(L);
    %----------------------------------------------
    imshow(img0);
    for n=1:N
        c=round(prop(n).Centroid); % obtener centroide
        rectangle('Position',prop(n).BoundingBox,'EdgeColor','g','LineWidth',2); % dibujar rectangulo
        %text(0.5,0.5, '{first line \newlinesecond line}')
        text(c(1),c(2),strcat('X:',num2str(c(1)),' \newline',' Y:',num2str(c(2))),'Color','green');%agregar coordenada
        line([640/2 640/2], [0 480],'Color','red','LineWidth',2);%dibujo linea vertical
        line([0 640], [480/2 480/2],'Color','red','LineWidth',2);%dibujo linea hrtl
    end
%____________________________________________________    
    if ~ishandle(wb)
        break
    else
        waitbar(i/10,wb,['num: '  num2str(i)]);
    end
%_____________________________________________________
    i=i+1;
    pause(0.001);
end
clear cam;