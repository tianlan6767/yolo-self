YOLOv7


## 修改 
   - 追加训练过程中随机裁剪2048增强方式
        ```python
        def randomcrop(img,labels,channel,crop_size):
            if channel == 3:
                img_rgb = img.copy()
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            else:
                img_rgb = img_gray = img.copy()

            img_h,img_w = img_gray.shape[:2]
            crop_size = np.asarray(crop_size, dtype=np.int32)
            image_size = (img_h,img_w)
            polygons = []
            if len(labels):
                label_index = random.choice(range(len(labels)))
                c,x,y,w,h  = labels[label_index]
                ins_w,ins_h = w*img_w, h*img_h
                center_x,center_y = x*img_w, y*img_h
                center_yx = (center_y,center_x)
                fn = str(random.random())

                min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
                max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
                max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))
                y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
                x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
                if channel == 3:
                    new_im = img_rgb[y0:y0+crop_size[0],x0:x0+crop_size[1],:]
                else:
                    new_im = img_gray[y0:y0+crop_size[0],x0:x0+crop_size[1]]
                jf_im = np.zeros(image_size, np.uint8)
                for label in labels:
                    c,x,y,w,h  = label
                    ins_w,ins_h = w*img_w, h*img_h
                    center_x,center_y = x*img_w, y*img_h
                    x1,x2 = int(center_x-ins_w/2),int(center_x+ins_w/2)
                    y1,y2 = int(center_y-ins_h/2),int(center_y+ins_h/2)
                    # cv2.rectangle(img_copy, (x1,y1),(x2,y2),(0,255,0), 5)
                    counter = [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]
                    cv2.fillConvexPoly(jf_im, np.array(counter), (int(c)+1)*50)        
                jf_im_crop = jf_im[y0:y0+crop_size[0],x0:x0+crop_size[1]]
                contours, _ = cv2.findContours(jf_im_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_SIMPLE
                
                new_labels = []
                for contour in contours:
                    polygons = contour.flatten().tolist()
                    xs, ys = polygons[0::2], polygons[1::2]
                    c = int(jf_im_crop[ys[0]][xs[0]]/50-1)
                    x,y = (max(xs)+min(xs))/2/crop_size[1],(max(ys)+min(ys))/2/crop_size[0]
                    w,h = (max(xs)-min(xs))/crop_size[1],(max(ys)-min(ys))/crop_size[0]
                    new_labels.append([c,x,y,w,h])
                    # cv2.rectangle(new_im, (min(xs),min(ys)),(max(xs),max(ys)),(255,0,0), 5)
                # cv2.imwrite(r"/home/ps/111/s{}.jpg".format(fn),new_im)
                return new_im,np.asarray(new_labels,dtype="float32")
            else:
                x0 = np.random.randint(0,image_size[1]-crop_size[1]+1)
                y0 = np.random.randint(0,image_size[0]-crop_size[0]+1)
                if channel == 3:
                    new_im = img_rgb[y0:y0+crop_size[0],x0:x0+crop_size[1],:]
                else:
                    new_im = img_gray[y0:y0+crop_size[0],x0:x0+crop_size[1]]
                return new_im,np.asarray(labels)
        ```
    
   - 修改推理代码的结果输出为json




## 安装训练等参考原始链接








