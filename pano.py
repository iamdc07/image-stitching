import cv2
import numpy as np
import random
import math

inlier_threshold = 10
iterations = 750
img = 0
img2 = 0
coloured_img = 0
coloured_img2 = 0


def project(x, y, h):
    point_array = np.array([[x], [y], [1]], dtype=np.float64)
    result = np.dot(h, point_array)
    if result[2] != 0:
        x2 = result[0][0] / result[2][0]
        y2 = result[1][0] / result[2][0]
    else:
        x2 = result[0][0]
        y2 = result[1][0]
    return x2, y2


def compute_inlier_count(h, matches, descriptor1, descriptor2):
    inlier1 = []
    inlier2 = []

    for each_match in matches:
        x, y, d = descriptor1[each_match.queryIdx]
        x1, y1, d1 = descriptor2[each_match.trainIdx]
        x2, y2 = project(x, y, h)
        distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

        if distance < inlier_threshold:
            inlier1.append([x, y])
            inlier2.append([x1, y1])

    return len(inlier1), inlier1, inlier2


def ransac(matches, descriptor1, descriptor2):
    max_count = 0
    src_points = []
    dst_points = []
    best_h = 0
    j = 0

    while j <= iterations:
        if len(matches) < 4:
            r = (random.sample(range(len(matches)), len(matches)))
        elif len(matches) == 1:
            x, y, d = descriptor1[matches[0].queryIdx]
            x1, y1, d1 = descriptor2[matches[0].trainIdx]
            src_points.append([x, y])
            dst_points.append([x1, y1])
            the_tuple = cv2.findHomography(np.float32(src_points), np.float32(dst_points), 0)
            h = the_tuple[0]
            best_h = h
            break
        else:
            r = (random.sample(range(len(matches)), 4))

        for i in r:
            pair = matches[i]

            x, y, d = descriptor1[pair.queryIdx]
            x1, y1, d1 = descriptor2[pair.trainIdx]

            src_points.append([x, y])
            dst_points.append([x1, y1])

        the_tuple = cv2.findHomography(np.float32(src_points), np.float32(dst_points), 0)
        h = the_tuple[0]

        count, inliers1, inliers2 = compute_inlier_count(h, matches, descriptor1, descriptor2)

        if count > max_count:
            max_count = count
            best_h = h

        src_points.clear()
        dst_points.clear()
        j += 1

    count, max_inliers1, max_inliers2 = compute_inlier_count(best_h, matches, descriptor1, descriptor2)

    the_tuple = cv2.findHomography(np.float32(max_inliers1), np.float32(max_inliers2), 0)
    h = the_tuple[0]

    kp1 = []
    for x, y in max_inliers1:
        kp1.append(cv2.KeyPoint(y, x, 1))

    kp2 = []
    for x, y in max_inliers2:
        kp2.append(cv2.KeyPoint(y, x, 1))

    new_matches = link_matches(max_inliers1, max_inliers2)
    print("After RANSAC:", len(new_matches), " matches")

    result = cv2.drawMatches(coloured_img, kp1, coloured_img2, kp2, new_matches, None)
    cv2.imshow('After RANSAC', result)
    cv2.waitKey(5000)

    h_inv = np.linalg.inv(h)

    return stitch(h, h_inv)

def stitch(h, h_inv):
    corners = []
    new_corners = []
    height1, width1 = img2.shape

    mat = np.array([[0, 0],
                    [0, img.shape[1]],
                    [img.shape[0], 0],
                    [img.shape[0], img.shape[1]]])

    corners.append((0, 0))
    corners.append((0, img2.shape[1]))
    corners.append((img2.shape[0], 0))
    corners.append((img2.shape[0], img2.shape[1]))

    for i in range(0, 4):
        new_corners.append(project(corners[i][0], corners[i][1], h_inv))

    print("NEW CORNERS:", new_corners)
    print("HERE")
    new_corners = tuple(tuple(map(int, tup)) for tup in new_corners)
    print("HERE1")
    new_corners = [list(elem) for elem in new_corners]
    print("HERE2")

    for i in new_corners:
        mat = np.append(mat, [i], axis=0)

    val = cv2.boundingRect(mat)
    print(val)
    height = val[2] - val[0]
    width = val[3] - val[1]
    print("H:", height, " W:", width)

    stitched = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            stitched[i - val[0], j - val[1]] = coloured_img[i, j]

    for i in range(val[0], height):
        for j in range(val[1], width):
            x2, y2 = project(i, j, h)
            if 0 <= x2 < height1 and 0 <= y2 < width1:
                if (i - val[0] < height) and (j - val[1] < width):
                    if isinstance(x2, int) and isinstance(y2, int):
                        patch = coloured_img2[x2, y2]
                        point = patch
                    else:
                        patch = cv2.getRectSubPix(coloured_img2, (3, 3), (y2, x2))
                        point = patch[1, 1]

                    stitched[i - val[0], j - val[1]] = np.ravel(point)
    return stitched


def link_matches(inliers1, inliers2):
    new_matches = []

    for i in range(0, len(inliers1)):
        x1, y1 = inliers1[i]
        x2, y2 = inliers2[i]
        distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

        each_match = cv2.DMatch(i, i, distance)
        new_matches.append(each_match)

    return new_matches


def set_images(im, im2, im3, im4):
    global img, img2, coloured_img, coloured_img2
    img = im
    img2 = im2
    coloured_img = im3
    coloured_img2 = im4


def blend(new_corners, height, width, h):
    height1, width1 = img2.shape

    # height = new_corners[2][0] - new_corners[0][0]
    # width = new_corners[3][1] - new_corners[1][1]
    # print("H:", height, " W:", width)

    mat = np.array([[0, 0]])

    new_corners = tuple(tuple(map(int, tup)) for tup in new_corners)
    new_corners = [list(elem) for elem in new_corners]

    for i in new_corners:
        mat = np.append(mat, [i], axis=0)

    mat = mat[1:]
    val = cv2.boundingRect(mat)
    print(val)
    height = val[2] - val[0]
    width = val[3] - val[1]
    print("H:", height, " W:", width)

    stitched = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(val[0], height):
        for j in range(val[1], width):
            x2, y2 = project(i, j, h)
            if 0 <= x2 < height1 and 0 <= y2 < width1:
                # print("X:", x2, " | Y:", y2)
                # print("X:", x2, " | Y:", y2, " I:", i, " J:", j)
                # if x2 < 0 and  y2 < 0:
                #     stitched[x2, y2] = coloured_img2[int(math.floor(x2)), int(math.floor(y2))]
                # else:
                if (i - val[0] < height) and (j - val[1] < width):
                    if isinstance(x2, int) and isinstance(y2, int):
                        # stitched[i - val[0], j - val[1]] = coloured_img2[x2, y2]
                        patch = coloured_img2[x2, y2]
                        point = patch
                    else:
                        patch = cv2.getRectSubPix(coloured_img2, (3, 3), (y2, x2))
                        point = patch[1, 1]

                stitched[i - val[0], j - val[1]] = np.ravel(point)

    # cv2.imshow('Final', stitched)
    # cv2.waitKey()
