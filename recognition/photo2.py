import cv2

from instance import instance_segmentation
segment_image = instance_segmentation() 
segment_image.load_model("mask_rcnn_coco.h5") 


cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    if not ret:
        print("Can't get frame")
        break
    cv2.imshow("I'm working...", frame)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("You've hit the Escape button, closing the app...")
        break

    if k%256 == 32: 
        
        print("Screenshot taken")
                
       
        width = int(len(frame[0])/2)
        
        
        first_image = frame[:, :width-1]
        second_image = frame[:, width:]
        
        cv2.imshow("1", first_image)
        cv2.imshow("2", second_image)

        segment_image.segmentImage(image_path = first_image, segment_target_classes = segment_image.select_target_classes(person=True), output_image_name="image.png", show_bboxes=True)
        segment_image.segmentImage(image_path = second_image, segment_target_classes = segment_image.select_target_classes(person=True), output_image_name="image1.png", show_bboxes=True)        
        # cv2.imwrite(img_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
       


cam.release()



