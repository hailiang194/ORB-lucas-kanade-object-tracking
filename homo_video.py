import cv2
import numpy as np
import sys
import time

def pre_process(frame):
    process_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    process_frame = cv2.medianBlur(process_frame, 3)
    process_frame = cv2.GaussianBlur(process_frame, (5, 5), 0)

    return process_frame

def is_valid_homography(dst, image):
    edge = np.float32([
        dst[1][0] - dst[0][0],
        dst[2][0] - dst[1][0],
        dst[3][0] - dst[2][0],
        dst[0][0] - dst[3][0]
    ])

    edge_len = np.linalg.norm(edge, axis=1)
    if abs((np.amax(edge_len) / np.amin(edge_len)) - (max(image.shape[:2]) / min(image.shape[:2]))) >= 0.4:
        return False

    get_cos = lambda x, y: (np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

    cos = np.float32([
        get_cos(edge[0], -edge[3]),
        get_cos(-edge[0], edge[1]),
        get_cos(-edge[1], edge[2]),
        get_cos(-edge[2], edge[3])
    ])

    corner = np.arccos(cos) * 180 / np.pi
    if abs(np.sum(corner) - 360.0) > 2:
        return False
    if np.min(corner) < 40:
        return False
    elif np.max(corner) >= 100:
        return False

    return True


def get_homography(frame, query_pts, train_pts, image):

    currentPts = len(query_pts)
    if query_pts.shape[0] <= 25:
        MIN_DISTANCE_OF_MIDPOINT = min(frame.shape[0] * 0.05, frame.shape[1] * 0.05)
        for i in range(1, currentPts):
            for j in range(i):
                if np.linalg.norm(train_pts[i] - train_pts[j]) > MIN_DISTANCE_OF_MIDPOINT:
                    query_pts = np.append(query_pts, [(query_pts[i] + query_pts[j]) / 2.0], axis=0)
                    train_pts = np.append(train_pts, [(train_pts[i] + train_pts[j]) / 2.0], axis=0)

    if train_pts.shape[0] < 4:
        return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), None

    mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    if mat is None:
        return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), None

    real_train_pts = cv2.perspectiveTransform(query_pts, mat)
    distance_filtered_query_pts = []
    distance_filtered_train_pts = []


    for i in range(len(query_pts)):
        distance = np.linalg.norm(real_train_pts[i][0] - train_pts[i][0])
        if distance <=  5.0:
            distance_filtered_query_pts.append(query_pts[i])
            distance_filtered_train_pts.append(train_pts[i])

    query_pts = np.float32(distance_filtered_query_pts)
    train_pts = np.float32(distance_filtered_train_pts)
    if len(good_points) > 10 and ((query_pts.shape[0]) >= 4): 
        mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        if mat is None:
            return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), None

        h, w = image.shape[:2]
        pts = np.float32([
            [0, 0], [0, h], [w, h], [w, 0]
        ]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, mat)

        if is_valid_homography(dst, image):
            return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), dst
    return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), None

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(sys.argv[2] if len(sys.argv) == 3 else 0)

    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    kp_image, desc_image = orb.detectAndCompute(img, None)
    corners = []
    lk_params = dict(winSize=(15, 15), 
                     maxLevel = 3,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    old_frame = None
    old_points = None
    query_points = None
    dst = None

    FRAMES_BUFFER = 15
    
    buffer = 0

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        process_frame = pre_process(frame)


        if buffer == FRAMES_BUFFER:
            old_frame = None
            old_points = None
            query_points = None
            dst = None
            buffer = 0

        if old_frame is None:
            kp_frame, desc_frame = orb.detectAndCompute(process_frame, None)

            matches = bf.knnMatch(desc_image, desc_frame, k=2)

            good_points = []
            #distance is the difference of 2 vector
            for m, n in matches:
                if m.distance < 0.7  * n.distance:
                    # if True:
                    good_points.append(m)

            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts =np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            query_points, points, dst = get_homography(process_frame, query_pts, train_pts, img)
            if not dst is None:
                old_points = points
                old_frame = process_frame

        else:
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, process_frame, old_points, None, **lk_params)
            mat, mask = cv2.findHomography(query_points.reshape(-1, 1, 2), new_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
            h, w = img.shape[:2]
            pts = np.float32([
                [0, 0], [0, h], [w, h], [w, 0]
            ]).reshape(-1, 1, 2)

            # print(np.amax(np.abs(new_points - old_points)))
            # print(np.amax(np.linalg.norm(old_points - new_points, axis=1)))
            # print(old_points.shape[0])
            for i in range(old_points.shape[0]):
                cv2.line(frame, (int(old_points[i, 0]), int(old_points[i, 1])), (int(new_points[i, 0]), int(new_points[i, 1])), (0, 255, 0), 3)
            dst = cv2.perspectiveTransform(pts, mat)
            old_frame = process_frame
            old_points = new_points
            buffer = buffer + 1
            

        if dst is None or not is_valid_homography(dst, img):
            cv2.imshow("frame", frame)
        else:
            test = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("frame", test)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
