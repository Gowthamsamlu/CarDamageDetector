import cv2
import numpy as np
import os
import datetime
from os import path

class DamageDetectorWithImages:
    def compareAndHighlightDamages(self, source_image_path):
        print("Start Of Algorithm Log")
        if not path.exists(source_image_path):
            return 0, 0, ""
        # Features
        sift = cv2.xfeatures2d.SIFT_create()
        # Feature Matching
        indexparam = dict(algorithm=0, trees=5)
        searchparam = dict()
        flann = cv2.FlannBasedMatcher(indexparam, searchparam)
        frame = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
        homography = frame
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # car_image
        keypoint_grayframe, descriptor_grayframe = sift.detectAndCompute(grayframe, None)

        dir_name = './masterAlgorithm/scratch_dataset/'
        # dir_name = 'data1a/training/00-damage/'
        scratch_images = os.listdir(dir_name)
        # print(scratch_images)
        for files in scratch_images:
            if not (files.endswith(".JPG") or files.endswith(".jpg") or files.endswith(".jpeg") or files.endswith(".JPEG")):
                scratch_images.remove(files)
        scratch_images_dataset = [dir_name + filenames for filenames in scratch_images]
        # print(scratch_images_dataset)
        highlights_cnt = 0
        # Repeat
        for imgs in scratch_images_dataset:
            img = cv2.imread(imgs, cv2.IMREAD_GRAYSCALE)
            keypoint_image, descriptor_image = sift.detectAndCompute(img, None)
            matches = flann.knnMatch(descriptor_image, descriptor_grayframe, k=2)
            good_points = []
            print("Filename: ", imgs," - ", end="")
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            # Homography
            print("Matches: ", len(good_points), " - ", end="")
            if len(good_points) >= 10:
                highlights_cnt += 1
                query_pts = np.float32([keypoint_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32([keypoint_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()

                # Perspective transform
                h, w = img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                homography = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 2)
                print("Highlighted damages")
            else:
                print("Very Low matching points")

        # Non-Repeat
        # cv2.imshow("Homography", homography)
        ct = str(datetime.datetime.now().timestamp()).replace(".", "")
        absfilename = "web_homography_" + ct + ".jpg"
        output_image_path = os.path.join('./static/output_images', absfilename)
        cv2.imwrite( output_image_path , homography)
        # cv2.waitKey()
        print("End Of Algorithm Log")
        return 1, highlights_cnt, absfilename

# damageDetectorWithImages = DamageDetectorWithImages()
# detection_status, highlights_cnt, output_file_path = damageDetectorWithImages.compareAndHighlightDamages("source_images/audi_a8_2.jpg")
# print("Detected: ", end="")
# if detection_status == 1:
#     print("Yes", end="")
#     if highlights_cnt > 0:
#         print(" - Number Of Damages detected: ", highlights_cnt," - Output_file_path: ",output_file_path, end="")
#     else:
#         print(" - No Damages detected")

# else:
#     print("No")