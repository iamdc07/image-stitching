import numpy as np
import cv2
import math, random

inlier_threshold = 10
iterations = 40


def project(x1, y1, h):
    x = np.array((x1, y1, 1), np.float)
    result = h.dot(x)
    result /= result[2]
    x2 = result[0]
    y2 = result[1]

    return x2, y2


def compute_inlier_count(h, matches, descriptor1, descriptor2):
    inlier1 = []
    inlier2 = []

    for each in matches:
        x, y, d = descriptor1[each.queryIdx]
        x1, y1, d1 = descriptor2[each.trainIdx]
        x2, y2 = project(x, y, h)
        distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

        if distance < inlier_threshold:
            inlier1.append((x, y))
            inlier2.append((x2, y2))

    return len(inlier1), inlier1, inlier2


def ransac(matches, sift_descriptor1, sift_descriptor2):
    max_count = 0
    # max_inliers1 = []
    # max_inliers2 = []
    best_h = 0
    j = 0
    src_points = []
    dst_points = []

    while j <= iterations:
        for i in range(0, 4):
            r = random.randint(0, len(matches) - 1)
            pair = matches[r]

            # print(pair.queryIdx, " | Image2 index:", pair.trainIdx)
            x, y, d = sift_descriptor1[pair.queryIdx]
            x1, y1, d1 = sift_descriptor2[pair.trainIdx]
            # print("X:", x, " Y:", y)
            src_points.append([x, y])
            dst_points.append([x1, y1])

        the_tuple = cv2.findHomography(np.float32(src_points), np.float32(dst_points), 0)
        h = np.array(the_tuple[0], np.float32)

        count, inliers1, inliers2 = compute_inlier_count(h, matches, sift_descriptor1, sift_descriptor2)
        # print("COUNT:", count)

        if count > max_count:
            max_count = count
            # max_inliers1 = inliers1
            # max_inliers2 = inliers2
            best_h = h

        src_points.clear()
        dst_points.clear()
        j += 1

    count, max_inliers1, max_inliers2 = compute_inlier_count(best_h, matches, sift_descriptor1, sift_descriptor2)
    # print("BEST COUNT1:", max_count, " Best count2:", len(max_inliers2))

    the_tuple = cv2.findHomography(np.float32(max_inliers1), np.float32(max_inliers2), 0)
    h = np.array(the_tuple[0], np.float32)

    kp1 = []
    for x, y in max_inliers1:
        kp1.append(cv2.KeyPoint(y, x, 1))

    kp2 = []
    for x, y in max_inliers2:
        kp2.append(cv2.KeyPoint(y, x, 1))

    # coloured_img = cv2.imread("project_images/Rainier1.png")
    # coloured_img2 = cv2.imread("project_images/Rainier2.png")
    # coloured_img = cv2.imread("project_images/MelakwaLake1.png")
    # coloured_img2 = cv2.imread("project_images/MelakwaLake2.png")
    # coloured_img = cv2.imread("project_images/MelakwaLake3.png")
    # coloured_img2 = cv2.imread("project_images/MelakwaLake4.png")
    coloured_img = cv2.imread("project_images/pano1_0008.jpg")
    coloured_img2 = cv2.imread("project_images/pano1_0009.jpg")

    new_matches = link_matches(max_inliers1, max_inliers2)
    print("After RANSAC:", len(new_matches), " matches")

    result = cv2.drawMatches(coloured_img, kp1, coloured_img2, kp2, new_matches, None)
    cv2.imshow('After RANSAC', result)
    cv2.waitKey()


def link_matches(inliers1, inliers2):
    new_matches = []

    for i in range(0, len(inliers1)):
        x1, y1 = inliers1[i]
        x2, y2 = inliers2[i]
        distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

        each_match = cv2.DMatch(i, i, distance)
        new_matches.append(each_match)

    return new_matches



