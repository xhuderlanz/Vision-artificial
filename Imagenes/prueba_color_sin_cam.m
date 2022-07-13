


    img0=imread("D:\\DocumentosD\\UTP\\brazo\\cubosbrazo.jpeg");
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
    figure();
    subplot(1,3,1);imshow(bw);
    subplot(1,3,2);imshow(img);
    subplot(1,3,3);imshow(img0);
    hold on;
    for n=1:N
        c=round(prop(n).Centroid); % obtener centroide
        rectangle('Position',prop(n).BoundingBox,'EdgeColor','g','LineWidth',2); % dibujar rectangulo
        %text(0.5,0.5, '{first line \newlinesecond line}')
        text(c(1),c(2),strcat('X:',num2str(c(1)),' \newline',' Y:',num2str(c(2))),'Color','green');%agregar coordenada
        line([640/2 640/2], [0 480],'Color','red','LineWidth',2);%dibujo linea vertical
        line([0 640], [480/2 480/2],'Color','red','LineWidth',2);%dibujo linea hrtl
    end
    hold off;
    
    
    %red R=191; G=53; B=41;
    %yw R=221; G=181; B=5;
    %nl R=49; G=63; B=133;
    %gr R=62; G=131; B=83;
    
    
    
    