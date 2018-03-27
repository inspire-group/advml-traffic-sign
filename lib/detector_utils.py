from lib.utils import *


CONF_THRES = 0.95   # Threshold for confidence score
IOU_THRES = 0.5     # Threshold for IoU for evaluating detector
# Labels of circlur signs
cir_cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16,
           17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]


def bb_iou(boxA, boxB):
    """
    Calculate IoU from two bounding boxes. Code adapted from:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def read_bb(filename):
    """Read ground truth bounding boxes from file"""

    gt_bboxes = {}
    with open(filename) as file:
        for line in file:
            frame = str(int(line.strip().split(".")[0]))
            bb = line.strip().split(";")[1:6]
            bb = list(map(int, bb))
            if frame in gt_bboxes:
                gt_bboxes[frame].append(bb)
            else:
                gt_bboxes[frame] = [bb]
    return gt_bboxes


def find_circles(img, mg_ratio=0.4, n_circles=1):
    """
    Find circular objects and return bounding boxes in the format
    [x1, y1, x2, y2]
    """

    targetImg = np.copy(img)
    targetImg = np.uint8(targetImg * 255)
    # Apply Gaussian blur if needed
    n = 13
    targetImg = cv2.GaussianBlur(targetImg, (n, n), 0)

    # Convert to grayscale
    grayImg = np.uint8(rgb2gray(targetImg))
    # param1 is canny threshold, param2 is circle accumulator threshold
    # Set of parameters for GTSDB testing
    # (because of different frame size and recording device)
    # circles = cv2.HoughCircles(grayImg, cv2.HOUGH_GRADIENT, 1, 200,
    #                            param1=60, param2=50, minRadius=5,
    #                            maxRadius=100)
    circles = cv2.HoughCircles(grayImg, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=50, param2=30, minRadius=20,
                               maxRadius=250)

    bboxes = []
    try:
        cir = circles.astype(np.uint16)
        for c in cir[0, :n_circles]:
            r = int(c[2])
            mg = int(r * mg_ratio)
            bboxes.append([c[0] - r - mg, c[1] - r - mg,
                           c[0] + r + mg, c[1] + r + mg])
    except AttributeError:
        pass
    except:
        raise
    return bboxes


def draw_bb(img, bbox, color=(0, 1, 0)):
    """Draw bounding box"""
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  color, 2)
    return img


def crop_bb(img, bbox):
    """Crop image by specifying bounding box"""
    bb = np.array(bbox)
    bb = bb * (bb > 0)
    return img[bb[1]:bb[3], bb[0]:bb[2], :]


def detect(model, im):
    """Run detection, returns bounding box"""

    img = np.copy(im)
    bboxes = find_circles(img, mg_ratio=0.4, n_circles=3)
    bb = []
    for bbox in bboxes:
        crop = crop_bb(im, bbox)
        resized_im = resize(crop)
        y_pred = predict(model, resized_im)
        conf = np.max(softmax(model.predict(
            resized_im.reshape(INPUT_SHAPE))[0]))
        # Consider detection only if confidence is larger than threshold
        if conf > CONF_THRES:
            bbox.append(y_pred)
            bb.append(bbox)
    return bb


def detect_mAP(model, images, gt_bboxes):
    """Run detection and evaluate mAP"""

    n_pos = np.zeros(NUM_LABELS)
    n_tp = np.zeros(NUM_LABELS)

    for i, im in enumerate(images):
        if str(i) not in gt_bboxes:
            continue
        bboxes = detect(model, im)
        for bbox in bboxes:
            y_pred = bbox[4]
            n_pos[y_pred] += 1
            # Check the detected bb against all ground truth bb's
            for bb in gt_bboxes[str(i)]:
                if y_pred == bb[4] and bb_iou(bb, bbox) > IOU_THRES:
                    n_tp[y_pred] += 1
                    break

    return n_tp, n_pos
