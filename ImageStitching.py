import numpy as np
import cv2
from matplotlib import pyplot as plt
import random

def read_show():
    # Read images
    pano_1 = cv2.imread("./data/panoramas/pano_1.jpg",cv2.IMREAD_UNCHANGED)
    pano_2 = cv2.imread("./data/panoramas/pano_2.jpg",cv2.IMREAD_UNCHANGED)
    pano_3 = cv2.imread("./data/panoramas/pano_3.jpg",cv2.IMREAD_UNCHANGED)
    pano_4 = cv2.imread("./data/panoramas/pano_4.jpg",cv2.IMREAD_UNCHANGED)
    pano_5 = cv2.imread("./data/panoramas/pano_5.jpg",cv2.IMREAD_UNCHANGED)

    # Plot Original Images
    fig = plt.figure("Original Images")

    plt.subplot(1, 5, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(pano_1, cv2.COLOR_BGR2RGB))
    plt.title("Image 1")

    plt.subplot(1, 5, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(pano_2, cv2.COLOR_BGR2RGB))
    plt.title("Image 2")

    plt.subplot(1, 5, 3)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(pano_3, cv2.COLOR_BGR2RGB))
    plt.title("Image 3")

    plt.subplot(1, 5, 4)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(pano_4, cv2.COLOR_BGR2RGB))
    plt.title("Image 4")

    plt.subplot(1, 5, 5)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(pano_5, cv2.COLOR_BGR2RGB))
    plt.title("Image 5")

    return pano_1, pano_2, pano_3, pano_4, pano_5


def SIFT_detector(img):
    # Convert to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d_SIFT.create()

    # Find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img_gray, None)

    # Draw keypoints
    img_keypoints = np.empty(img.shape, dtype=np.uint8)
    cv2.drawKeypoints(img, kp, img_keypoints)

    return kp, des, img_keypoints


def BFMatcher(img1, des1, kp1, img2, des2, kp2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # Draw total matches
    # cv2.drawMatchesKnn expects list of lists as matches.
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
                           dtype=np.uint8)
    cv2.drawMatchesKnn(img1, kp1, img2, kp2, outImg=img_matches, matches1to2=good, flags=2)

    return good, img_matches


def ransac(src_points, dst_points, ransac_reproj_threshold=1, max_iters=1000, inlier_ratio=0.8):
    """
    Calculate the set of inlier correspondences w.r.t. homography transformation, using the
        RANSAC method.
    :param src_points: numpy.array(float), coordinates of the points in the source image
    :param dst_points: numpy.array(float), coordinates of the points in the destination image
    :param ransac_reproj_threshold: float, maximum allowed reprojection error to treat a point pair
    as an inlier
    :param max_iters: int, the maximum number of RANSAC iterations
    :param inlier_ratio: float, ratio of inliers w.r.t. total number of correspondences
    :return:
    H: numpy.array(float), the estimated homography transformation
    mask: numpy.array(uint8), mask that denotes the inlier correspondences
    """ 
    assert(src_points.shape[0] == dst_points.shape[0])
    assert (src_points.shape[1] == dst_points.shape[1])
    assert(ransac_reproj_threshold >= 0)
    assert(max_iters > 0)
    assert(inlier_ratio >= 0.0 and inlier_ratio <= 1.0)

    max_inliers = 0
    max_H = []
    for interation in range(max_iters):
        random_num_1, random_num_2,random_num_3,random_num_4 = random.sample(range(0, len(src_points)), 4)
        pts1 = np.float32([src_points[random_num_1], src_points[random_num_2], src_points[random_num_3], src_points[random_num_4]])
        pts2 = np.float32([dst_points[random_num_1], dst_points[random_num_2], dst_points[random_num_3], dst_points[random_num_4]])
        temp_mask = np.zeros(src_points.shape[0], dtype=np.uint8)

        H_temp = cv2.getPerspectiveTransform(src=pts1, dst=pts2)

        projected_src = []
        src_points_homogeneous = []
        dst_points_homogeneous = []

        # Homogeneous
        for i in range(src_points.shape[0]):
            src_points_homogeneous.append(list(np.append(src_points[i], 1)))
            dst_points_homogeneous.append(list(np.append(dst_points[i], 1)))

        for i in range(src_points.shape[0]):
            projected_src.append(list(np.matmul(H_temp, src_points_homogeneous[i])))

        sum = 0
        inliers = 0
        for i in range(dst_points.shape[0]):
            for j in range(3):
                sum += (dst_points_homogeneous[i][j] - projected_src[i][j]/projected_src[i][2])**2
            distance = np.square(sum)
            # print(distance)
            sum = 0
            if distance < ransac_reproj_threshold:
                inliers += 1
                temp_mask[i] = 1


        if inliers > max_inliers:
            max_inliers = inliers
            mask = np.copy(temp_mask)
            H = np.copy(H_temp)
        if inliers > inlier_ratio*src_points.shape[0]:
            print(max_inliers)
            return H, mask


    return H, mask


def draw_inliers(img1, kp1, img2, kp2, good, masked):
    # Draw inliers
    img_inliers = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    good = np.array(good)
    inliers = good[np.where(np.squeeze(masked) == 1)[0]]
    cv2.drawMatchesKnn(img1, kp1, img2, kp2, outImg=img_inliers, matches1to2=inliers, flags=2)
    return img_inliers


def stiching_blending(pano_1, pano_2, pano_3, pano_4, pano_5, H_1_2, H_2_3, H_3_4, H_4_5):
    # Stitch images
    panorama_height = np.maximum(pano_1.shape[0], pano_2.shape[0])
    panorama_width = pano_1.shape[1] + pano_2.shape[1]
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama[0:pano_1.shape[0], 0:pano_1.shape[1]] = pano_1
    warped_img = cv2.warpPerspective(pano_2, H_1_2, (panorama_width, panorama_height),
                                     flags=cv2.WARP_INVERSE_MAP)

    # Blending
    temp_panorama = np.round(0.5 * panorama + 0.5 * warped_img).astype(np.uint8)
    temp_panorama[warped_img == [0, 0, 0]] = panorama[warped_img == [0, 0, 0]]
    temp_panorama[panorama == [0, 0, 0]] = warped_img[panorama == [0, 0, 0]]
    panorama1 = temp_panorama.copy()

    # Stitch images
    panorama_height = np.maximum(panorama1.shape[0], pano_3.shape[0])
    panorama_width = panorama1.shape[1] + pano_3.shape[1]
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama[0:panorama1.shape[0], 0:panorama1.shape[1]] = panorama1
    warped_img = cv2.warpPerspective(pano_3, np.matmul(H_2_3, H_1_2), (panorama_width, panorama_height),
                                     flags=cv2.WARP_INVERSE_MAP)

    # Blending
    temp_panorama = np.round(0.5 * panorama + 0.5 * warped_img).astype(np.uint8)
    temp_panorama[warped_img == [0, 0, 0]] = panorama[warped_img == [0, 0, 0]]
    temp_panorama[panorama == [0, 0, 0]] = warped_img[panorama == [0, 0, 0]]
    panorama2 = temp_panorama.copy()

    # Stitch images
    panorama_height = np.maximum(panorama2.shape[0], pano_4.shape[0])
    panorama_width = panorama2.shape[1] + pano_4.shape[1]
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama[0:panorama2.shape[0], 0:panorama2.shape[1]] = panorama2
    warped_img = cv2.warpPerspective(pano_4, np.matmul(H_3_4, np.matmul(H_2_3, H_1_2)),
                                     (panorama_width, panorama_height),
                                     flags=cv2.WARP_INVERSE_MAP)

    # Blending
    temp_panorama = np.round(0.5 * panorama + 0.5 * warped_img).astype(np.uint8)
    temp_panorama[warped_img == [0, 0, 0]] = panorama[warped_img == [0, 0, 0]]
    temp_panorama[panorama == [0, 0, 0]] = warped_img[panorama == [0, 0, 0]]
    panorama3 = temp_panorama.copy()

    # Stitch images
    panorama_height = np.maximum(panorama3.shape[0], pano_5.shape[0])
    panorama_width = panorama3.shape[1] + pano_5.shape[1]
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama[0:panorama3.shape[0], 0:panorama3.shape[1]] = panorama3
    warped_img = cv2.warpPerspective(pano_5, np.matmul(H_4_5, np.matmul(H_3_4 ,np.matmul(H_2_3, H_1_2))),
                                     (panorama_width, panorama_height),
                                     flags=cv2.WARP_INVERSE_MAP)

    # Blending
    temp_panorama = np.round(0.5 * panorama + 0.5 * warped_img).astype(np.uint8)
    temp_panorama[warped_img == [0, 0, 0]] = panorama[warped_img == [0, 0, 0]]
    temp_panorama[panorama == [0, 0, 0]] = warped_img[panorama == [0, 0, 0]]
    panorama4 = temp_panorama.copy()

    return panorama4


def main():
    # Load 5 images
    pano_1, pano_2, pano_3, pano_4, pano_5 = read_show()

    print("Run Image Stitching ...")
    print("Find Keypoints and Descriptors using SIFT ...")
    # Feature Detection
    kp1, des1, img_keypoints1 = SIFT_detector(pano_1)
    kp2, des2, img_keypoints2 = SIFT_detector(pano_2)
    kp3, des3, img_keypoints3 = SIFT_detector(pano_3)
    kp4, des4, img_keypoints4 = SIFT_detector(pano_4)
    kp5, des5, img_keypoints5 = SIFT_detector(pano_5)

    print("BFMatcher ...")
    # Feature matching
    good_1_2, img_matches_1_2 = BFMatcher(pano_1, des1, kp1, pano_2, des2, kp2)
    good_2_3, img_matches_2_3 = BFMatcher(pano_2, des2, kp2, pano_3, des3, kp3)
    good_3_4, img_matches_3_4 = BFMatcher(pano_3, des3, kp3, pano_4, des4, kp4)
    good_4_5, img_matches_4_5 = BFMatcher(pano_4, des4, kp4, pano_5, des5, kp5)

    print("My RANSAC and OpenCV RANSAC ...")
    # Estimate homography using RANSAC algorithm
    src_1_2 = np.float32([kp1[g[0].queryIdx].pt for g in good_1_2])
    dst_1_2 = np.float32([kp2[g[0].trainIdx].pt for g in good_1_2])
    print("Images: 1, 2 ...")
    # Mine Ransac Image1, Image2
    H_1_2, masked_1_2 = ransac(src_1_2,dst_1_2)
    # OpenCv Ransac Image1, Image2
    cv2_H_1_2, cv2_masked_1_2 = cv2.findHomography(srcPoints=src_1_2, dstPoints=dst_1_2, method=cv2.RANSAC,
                                   ransacReprojThreshold=1.0, maxIters=1000)
    # Inliers
    my_img_inliers_1_2 = draw_inliers(pano_1, kp1, pano_2, kp2, good_1_2, masked_1_2)
    cv2_img_inliers_1_2 = draw_inliers(pano_1, kp1, pano_2, kp2, good_1_2, cv2_masked_1_2)

    # Estimate homography using RANSAC algorithm
    src_2_3 = np.float32([kp2[g[0].queryIdx].pt for g in good_2_3])
    dst_2_3 = np.float32([kp3[g[0].trainIdx].pt for g in good_2_3])
    print("Images: 2, 3 ...")
    # Mine Ransac Image2, Image3
    H_2_3, masked_2_3 = ransac(src_2_3, dst_2_3)
    # OpenCv Ransac Image2, Image3
    cv2_H_2_3, cv2_masked_2_3 = cv2.findHomography(srcPoints=src_2_3, dstPoints=dst_2_3, method=cv2.RANSAC,
                                                   ransacReprojThreshold=1.0, maxIters=1000)
    # Inliers
    my_img_inliers_2_3 = draw_inliers(pano_2, kp2, pano_3, kp3, good_2_3, masked_2_3)
    cv2_img_inliers_2_3 = draw_inliers(pano_2, kp2, pano_3, kp3, good_2_3, cv2_masked_2_3)

    # Estimate homography using RANSAC algorithm
    src_3_4 = np.float32([kp3[g[0].queryIdx].pt for g in good_3_4])
    dst_3_4 = np.float32([kp4[g[0].trainIdx].pt for g in good_3_4])
    print("Images: 3, 4 ...")
    # Mine Ransac Image3, Image4
    H_3_4, masked_3_4 = ransac(src_3_4, dst_3_4)
    # OpenCv Ransac Image3, Image4
    cv2_H_3_4, cv2_masked_3_4 = cv2.findHomography(srcPoints=src_3_4, dstPoints=dst_3_4, method=cv2.RANSAC,
                                                   ransacReprojThreshold=1.0, maxIters=1000)
    # Inliers
    my_img_inliers_3_4 = draw_inliers(pano_3, kp3, pano_4, kp4, good_3_4, masked_3_4)
    cv2_img_inliers_3_4 = draw_inliers(pano_3, kp3, pano_4, kp4, good_3_4, cv2_masked_3_4)

    # Estimate homography using RANSAC algorithm
    src_4_5 = np.float32([kp4[g[0].queryIdx].pt for g in good_4_5])
    dst_4_5 = np.float32([kp5[g[0].trainIdx].pt for g in good_4_5])
    print("Images: 4, 5 ...")
    # Mine Ransac Image4, Image5
    H_4_5, masked_4_5 = ransac(src_4_5, dst_4_5)
    # OpenCv Ransac Image4, Image5
    cv2_H_4_5, cv2_masked_4_5 = cv2.findHomography(srcPoints=src_4_5, dstPoints=dst_4_5, method=cv2.RANSAC,
                                                   ransacReprojThreshold=1.0, maxIters=1000)
    # Inliers
    my_img_inliers_4_5 = draw_inliers(pano_4, kp4, pano_5, kp5, good_4_5, masked_4_5)
    cv2_img_inliers_4_5 = draw_inliers(pano_4, kp4, pano_5, kp5, good_4_5, cv2_masked_4_5)

    print("Stitching and blending Images ...")
    my_panorama = stiching_blending(pano_1, pano_2, pano_3, pano_4, pano_5, H_1_2, H_2_3, H_3_4, H_4_5)
    cv2_panorama = stiching_blending(pano_1, pano_2, pano_3, pano_4, pano_5, cv2_H_1_2, cv2_H_2_3, cv2_H_3_4, cv2_H_4_5)


    ####################### Printing ###########################

    # Plot Feature detection
    fig = plt.figure("Image SIFT Keypoints")

    plt.subplot(1, 5, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_keypoints1, cv2.COLOR_BGR2RGB))
    plt.title("Image 1")

    plt.subplot(1, 5, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_keypoints2, cv2.COLOR_BGR2RGB))
    plt.title("Image 2")

    plt.subplot(1, 5, 3)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_keypoints3, cv2.COLOR_BGR2RGB))
    plt.title("Image 3")

    plt.subplot(1, 5, 4)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_keypoints4, cv2.COLOR_BGR2RGB))
    plt.title("Image 4")

    plt.subplot(1, 5, 5)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_keypoints5, cv2.COLOR_BGR2RGB))
    plt.title("Image 5")

    # Plot Feature detection
    fig = plt.figure("SIFT Feature matching")

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_matches_1_2, cv2.COLOR_BGR2RGB))
    plt.title("Matches 1")

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_matches_2_3, cv2.COLOR_BGR2RGB))
    plt.title("Matches 2")

    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_matches_3_4, cv2.COLOR_BGR2RGB))
    plt.title("Matches 3")

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_matches_4_5, cv2.COLOR_BGR2RGB))
    plt.title("Matches 4")

    # Plot Feature detection
    fig = plt.figure("My implementation of RANSAC Inliers")

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(my_img_inliers_1_2, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 1")

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(my_img_inliers_2_3, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 2")

    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(my_img_inliers_3_4, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 3")

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(my_img_inliers_4_5, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 4")

    # Plot Feature detection
    fig = plt.figure("RANSAC Inliers")

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(cv2_img_inliers_1_2, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 1")

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(cv2_img_inliers_2_3, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 2")

    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(cv2_img_inliers_3_4, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 3")

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(cv2_img_inliers_4_5, cv2.COLOR_BGR2RGB))
    plt.title("Inliers 4")

    # Plot Feature detection
    fig = plt.figure("Image Stitching")

    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(my_panorama, cv2.COLOR_BGR2RGB))
    plt.title("My Panorama")

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(cv2_panorama, cv2.COLOR_BGR2RGB))
    plt.title("Cv2 Panorama")

    plt.show()




if __name__ == "__main__":
    main()
