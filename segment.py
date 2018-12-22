import scipy

import cv2
import numpy as np
import time
from scipy import stats

from collections import defaultdict


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


from collections import defaultdict


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def createLines(img):
    img = cv2.resize(img, (1500, 1500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    return lines, img


def segregatePoints(sortedPoints):
    linesDic = defaultdict(list)
    for i in range(len(sortedPoints)):
        linesDic[sortedPoints[i][1]].append(sortedPoints[i])

    keysToMerge = [[]]
    lastKey = 999999
    for i in linesDic:
        if (i > lastKey + 20):  ### < -------- tu usuń +20
            keysToMerge.append([i])
        else:
            keysToMerge[len(keysToMerge) - 1].append(i)
        lastKey = i
    linesDic2 = defaultdict(list)
    for i in range(len(keysToMerge)):
        for j in keysToMerge[i]:
            linesDic2[i] = linesDic2[i] + (linesDic[j])
    for i in linesDic2:
        linesDic2[i] = sorted(linesDic2[i], key=lambda k: [k[0]])
        indToRemove = []
        for j in range(len(linesDic2[i]) - 1):
            if (linesDic2[i][j][0] > linesDic2[i][j + 1][0] - 50):  ###<---- tu usuń -50
                indToRemove.append(j + 1)
        linesDic2[i] = [i for j, i in enumerate(linesDic2[i]) if j not in indToRemove]

    return linesDic2


args = "--output D:\\doWywaleniaUczelnia\\test.txt --twitter-source.consumerKey D3RoDElXkKzypiyoaJmUXZDdW --twitter-source.consumerSecret U4Ogi4lOjCaJ3gnlvor7ZN3Eqoyih0do5KV9C8rrG66D8OLKrw --twitter-source.token 743995742-5gWpCMulAKtcT12l3B0avoXNBOioHeFm4NoSprAF --twitter-source.tokenSecret na7HPUONFJDJXhckgMKSoKu4p0FLg4GZ4vtXtyUPOGyzd"


def createStrongLines(lines):
    strong_lines = np.zeros([len(lines), 1, 2])
    n2 = 0

    for n1 in range(0, len(lines)):
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                if rho < 0:
                    rho *= -1
                    theta -= np.pi
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=10)
                closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness):
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
    strong_lines = strong_lines[0:n2]
    return strong_lines


def cropPolygon(pts, img):
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y:y + h, x:x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    return croped, mask, dst, dst2


def cropBox(bbox, img):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def cropFields(segregatedPoints, img, dir='test'):
    cnt = 0
    resImg = []
    for i in range(1, len(segregatedPoints[1])):
        ptsL = segregatedPoints[1][i - 1].copy()
        ptsL[1] = 0
        ptsR = segregatedPoints[1][i].copy()
        ptsL.extend(ptsR)
        cropped = cropBox(ptsL, img)
        resImg.append(cropped)
        cv2.imwrite(dir + "/cropped" + str(cnt) + ".jpg", cropped)
        cnt += 1

    for i in range(2, len(segregatedPoints)):
        for j in range(1, len(segregatedPoints[i])):
            ptsL = segregatedPoints[i - 2][j - 1]
            ptsR = segregatedPoints[i][j]
            ptsL.extend(ptsR)
            cropped = cropBox(ptsL, img)
            resImg.append(cropped)
            cv2.imwrite(dir + "/cropped" + str(cnt) + ".jpg", cropped)
            cnt += 1
    return resImg


def distance(point, center):
    diffs = [(center[i] - point[i]) ** 2 for i in range(2)]
    return np.sqrt(np.sum(diffs))


def filterPoints(segregatedPoints, img):
    pass
    linesCount = len(segregatedPoints)
    middleLine = segregatedPoints[linesCount / 2]

    lineLength = len(middleLine)

    print(lineLength / 2, middleLine, lineLength)
    diffs = []
    prev = [0, 0]
    for i in range(lineLength):
        curr = middleLine[i]
        diffs.append(curr[0] - prev[0])
        prev = curr
    print(diffs)


def filter_vertical_lines(segregated_points):
    # for key in segregated_points:
    #     print(segregated_points[key])

    all_diffs = []
    for key in segregated_points:
        line = segregated_points[key]
        prev = [0, 0]
        diffs = []
        for i in range(len(line)):
            curr = line[i]
            diffs.append(curr[0] - prev[0])
            prev = curr
        # print(diffs)
        all_diffs.append(diffs)

    starting_points = []
    for row in all_diffs:
        starting_points.append(find_first_important_vertical_line(row))
    # print(starting_points)
    starting_column = scipy.stats.mode(starting_points)[0][0]
    # print(starting_column)
    for key in segregated_points:
        end = max(min(len(segregated_points[key]), starting_column + 8), 8)
        segregated_points[key] = segregated_points[key][max(0, starting_column - 1):end]
    # for key in segregated_points:
    #     print(segregated_points[key])
    return starting_column


def find_first_important_vertical_line(row):
    maxStartingPosition = len(row) - 7
    max_differences = []
    for i in range(0, maxStartingPosition):
        min_ = 1500
        max_ = -1500
        for point in range(8):
            if row[i + point] < min_:
                min_ = row[i + point]
            if row[i + point] > max_:
                max_ = row[i + point]
        max_differences.append(abs(max_ - min_))
    return np.argmin(max_differences)


def filter_horizontal_lines(segregated_points):
    arr = [segregated_points[key] for key in segregated_points]
    keys = [key for key in segregated_points]
    if len(keys) < 10:
        return
    # print(keys)
    middle = len(arr[0]) // 2
    ys = [row[middle][1] for row in arr]
    prs = []
    prev = 0
    for el in ys:
        prs.append(el - prev)
        prev = el
    # print(prs)
    eighth = -1
    for i in range(len(prs)):
        if prs[i] > 130:
            eighth = i
            break
    print(eighth)

    segregated_points.clear()
    # print(keys)
    subtractor = 2
    if eighth == 1:
        subtractor = 1
    for i in range(8 + subtractor):
        segregated_points[i] = arr[i + eighth - subtractor]


def getFields(crops):
    fields = crops[-64:]
    print(len(fields))

    cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
    rows = [8, 7, 6, 5, 4, 3, 2, 1]
    dict = {}
    for row in range(8):
        for col in range(8):
            name = cols[col] + str(rows[row])
            dict[name] = fields[8 * row + col]

    test = ["a8", "b7", "c6", "d5", "e4", "f3", "g2", "h1"]
    for f in test:
        cv2.imshow(f, dict[f])
    return dict


def createSampImage(img, dir):
    start = time.time()
    lines, img = createLines(img)
    strong = createStrongLines(lines)

    segmented = segment_by_angle_kmeans(strong)
    intersections = segmented_intersections(segmented)

    s = sorted(intersections, key=lambda k: [k[1], k[0]])
    segregatedPoints = segregatePoints(s)

    img2 = img.copy()
    for x in strong:
        for rho, theta in x:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1500 * (-b))
            y1 = int(y0 + 1500 * (a))
            x2 = int(x0 - 1500 * (-b))
            y2 = int(y0 - 1500 * (a))

            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for i in segregatedPoints:
        for j in segregatedPoints[i]:
            try:
                cv2.circle(img2, (j[0], j[1]), 5, (0, 255, 255), -1)

            except:
                pass
    cr = cropBox([753, -400, 905, 259], img)
    # filterPoints(segregatedPoints, img)
    filter_vertical_lines(segregatedPoints)
    filter_horizontal_lines(segregatedPoints)
    for key in segregatedPoints:
        print(segregatedPoints[key])
    crops = cropFields(segregatedPoints, img, dir)
    fields = getFields(crops)
    stop = time.time()
    print(stop - start)

    return img2


import os


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


for i in {1, 3, 4, 5, 6, 7}:
# for i in {6}:
    print("Processing img {}".format(i))
    img = cv2.imread('samp' + str(i) + '.jpg')
    ensure_dir('testSample' + str(i))
    test = createSampImage(img, 'testSample' + str(i))
    cv2.imwrite('testSample' + str(i) + '/samp' + str(i) + 'Processed.jpg', test)
cv2.waitKey(0)
