import cv2
import numpy as np
import sys
import time


def mapping_image(first_image, first_kp, second_image, second_kp, dst=None):
    first_h, first_w = first_image.shape[:2]
    second_h, second_w = second_image.shape[:2]

    match_w = first_w + second_w
    match_h = max(first_h, second_h)

    match = np.zeros((match_h, match_w), dtype=np.uint8)

    match[0:first_h, 0:first_w] = first_image[:]
    match[0: second_h, -second_w - 1: -1] = second_image[:]

    match = np.dstack([match] * 3)

    for f_pt, s_pt in zip(first_kp, second_kp):
        start_pt = (int(f_pt[0][0]), int(f_pt[0][1]))
        end_pt = (int(s_pt[0][0]) + first_w, int(s_pt[0][1]))
        cv2.circle(match, tuple(start_pt), 3, (0, 255, 0), 1)
        cv2.circle(match, tuple(end_pt), 3, (0, 255, 0), 1)
        cv2.line(match, tuple(start_pt), tuple(end_pt), (0, 0, 255), 1)
    
    if not dst is None:
        map_dst = dst.copy()
        map_dst[:,:,0] = map_dst[:,:, 0] + first_w
        cv2.polylines(match, [np.int32(map_dst)], True, (255, 0, 0))
    return match

def pre_process(frame):
    process_frame = frame.copy()
    if len(frame.shape) == 3:
        process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
    # process_frame = cv2.medianBlur(process_frame, 3)
    process_frame = cv2.GaussianBlur(process_frame, (7, 7), 0)

    return process_frame

def is_valid_homography(dst, image, MIN_EDGE_RATIO=0.4, debug=False):
    edge = np.float32([
        dst[1][0] - dst[0][0],
        dst[2][0] - dst[1][0],
        dst[3][0] - dst[2][0],
        dst[0][0] - dst[3][0]
    ])

    edge_len = np.linalg.norm(edge, axis=1)
    if abs((np.amax(edge_len) / np.amin(edge_len)) - (max(image.shape[:2]) / min(image.shape[:2]))) >= MIN_EDGE_RATIO:
        return False

    get_cos = lambda x, y: (np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

    cos = np.float32([
        get_cos(edge[0], -edge[3]),
        get_cos(-edge[0], edge[1]),
        get_cos(-edge[1], edge[2]),
        get_cos(-edge[2], edge[3])
    ])

    corner = np.arccos(cos) * 180 / np.pi
    if debug:
        print(corner)
    if np.any(np.isnan(corner)):
        return False
    if abs(np.sum(corner) - 360.0) > 2:
        return False
    if np.min(corner) < 40:
        return False
    elif np.max(corner) >= 110:
        return False

    return True


def get_homography(frame, query_pts, train_pts, image, LIMIT_POINTS_FOR_MIDDLE=25, REMAP_ACCEPTED_DISTANCE=5.0, MIN_EDGE_RATIO=0.4, MIN_POINTS=10, debug=False):
    
    #add middle points for pairs of points which is too far
    dst = None
    len_points = len(query_pts)
    if query_pts.shape[0] <= LIMIT_POINTS_FOR_MIDDLE:
        MIN_DISTANCE_OF_MIDPOINT = min(frame.shape[0] * 0.05, frame.shape[1] * 0.05)
        for i in range(1, len_points):
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
        if distance <= REMAP_ACCEPTED_DISTANCE:
            distance_filtered_query_pts.append(query_pts[i])
            distance_filtered_train_pts.append(train_pts[i])

    # print("REMOVE TOO FAR POINT")
    query_pts = np.float32(distance_filtered_query_pts)
    train_pts = np.float32(distance_filtered_train_pts)
    # if debug: cv2.imshow("Mapping debug", mapping_image(image, query_pts, frame, train_pts, dst))
    if len(good_points) > MIN_POINTS and ((query_pts.shape[0]) >= 4):
        mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        if mat is None:
            return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), None

        h, w = image.shape[:2]
        pts = np.float32([
            [0, 0], [0, h], [w, h], [w, 0]
        ]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, mat)

        if debug: cv2.imshow("Mapping debug", mapping_image(image, query_pts, frame, train_pts, dst))
        if is_valid_homography(dst, image, MIN_EDGE_RATIO):
            return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), dst
    return query_pts.reshape(-1, 2), train_pts.reshape(-1, 2), None

def get_homo_by_optical_flow_and_goodFeatureToTrack(frame, old_frame, old_points, old_dst, image, lk_params, MAX_MOVEMENT_DISTANCE = -1 , debug=False):
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, frame, old_points, None, **lk_params)
    if debug: print(np.amax(np.linalg.norm(new_points - old_points, axis=1)))
    if MAX_MOVEMENT_DISTANCE > 0:
        if np.amax(np.linalg.norm(new_points - old_points, axis=1)) > MAX_MOVEMENT_DISTANCE:
            return None, None

    mat, _ = cv2.findHomography(old_points.reshape(-1, 1, 2), new_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    h, w = image.shape[:2]
    pts = np.float32([
                [0, 0], [0, h], [w, h], [w, 0]
            ]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(old_dst, mat)
    if debug:
        print(dst)
        for i in range(old_points.shape[0]):
            cv2.circle(frame, (int(old_points[i, 0]), int(old_points[i, 1])), 2, (255, 0, 0), 3)
            cv2.circle(frame, (int(new_points[i, 0]), int(new_points[i, 1])), 2, (0, 0, 255), 3)
            cv2.line(frame, (int(old_points[i, 0]), int(old_points[i, 1])), (int(new_points[i, 0]), int(new_points[i, 1])), (0, 255, 0), 3)
        cv2.imshow("point", frame)

    if dst is not None and is_valid_homography(dst, image, debug=debug):
        h, status = cv2.findHomography(pts, dst)
                
        mask = cv2.warpPerspective(image, h, (frame.shape[1], frame.shape[0]))
        if debug:
            cv2.imshow("mask", mask)
        old_points = np.float32(cv2.goodFeaturesToTrack(frame, 200000, 0.01, 10,mask=mask)).reshape(-1, 2)

        return old_points, dst
    else:
        return None, None

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img = pre_process(img)

    cap = cv2.VideoCapture(sys.argv[2] if len(sys.argv) == 3 else 0)

    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nfeatures=2000000)
    # orb = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # bf = cv2.BFMatcher()
    kp_image, desc_image = orb.detectAndCompute(img, None)
    lk_params = dict(winSize=(45, 45), 
                     maxLevel = 9,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    old_frame = None
    old_points = None
    query_points = None
    new_points = None
    dst = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    FRAMES_BUFFER = 100000 * fps
    # FRAMES_BUFFER = 5
    total_frame = 0
    optical = 0
    matching = 0
    not_found = 0

    buffer = 0

    while cap.isOpened():
        # clock = time.perf_counter()
        _, frame = cap.read()
        if frame is None:
            break
        process_frame = pre_process(frame)
        total_frame = total_frame + 1

        if buffer >= FRAMES_BUFFER:
            old_frame = None
            old_points = None
            query_points = None
            dst = None
            buffer = 0
            new_points = None

 
        if not old_frame is None:
            # clock = time.perf_counter()
            # print("Optical")
            old_points, dst = get_homo_by_optical_flow_and_goodFeatureToTrack(process_frame, old_frame, old_points, dst, img, lk_params)
            if dst is None:
                old_frame = None
            else:
                # print("DONE")
                is_valid_homography(dst, img, debug=True)
                old_frame = process_frame
                optical = optical + 1

            buffer = buffer + 1

        if old_frame is None:
        # if True:
            # print("Matching")
            kp_frame, desc_frame = orb.detectAndCompute(process_frame, None)
            # print(len(kp_frame))

            matches = bf.knnMatch(desc_image, desc_frame, k=2)

            good_points = []
            #distance is the difference of 2 vector
            for m, n in matches:
                if m.distance < 0.70 * n.distance:
                    # if True:
                    good_points.append(m)

            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts =np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            query_points, points, dst = get_homography(process_frame, query_pts, train_pts, img)
            if not dst is None:
                is_valid_homography(dst, img, debug=True)
            else:
                print("None")
            # print(query_pts.shape)
            cv2.imshow("Mapping debug", mapping_image(img, query_points.reshape(-1, 1, 2), process_frame, points.reshape(-1, 1, 2), dst))
            # print(query_pts.shape)
            # print()
            if not dst is None:
                matching = matching + 1
                old_points = points
                old_frame = process_frame



        # print(time.perf_counter() - clock)
        # clock = time.perf_counter()
        if dst is None:
            # print("None dst")
            not_found = not_found + 1
            buffer = FRAMES_BUFFER
        if not dst is None and is_valid_homography(dst, img):
            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        
        # if not query_points is None and not new_points is None:
            # cv2.imshow("Mapping", mapping_image(img, query_points.reshape(-1, 1, 2), process_frame, new_points.reshape(-1, 1, 2), dst))
            # cv2.imshow("Mapping debug", mapping_image(img, query_points.reshape(-1, 1, 2), process_frame, new_points.reshape(-1, 1, 2), dst))
        cv2.imshow("frame", frame)


        key = cv2.waitKey(int(fps))
        # key = cv2.waitKey(0)
        if key == 27:
            break
    # print("[%d %d %d %d]" % (optical, matching, not_found, total_frame))
    cap.release()
    cv2.destroyAllWindows()
