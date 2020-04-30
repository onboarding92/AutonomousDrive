from models import *
from utils.utils import load_classes, non_max_suppression, label_img_to_color
import argparse

from PIL import Image
import cv2

import torch
import torch.utils.data
from torchvision import transforms
from torch.autograd import Variable

from segmentation_model.deeplabv3 import DeepLabV3
from moving_object import MovingObject, MODictionary
from sort import *


# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
FPS = 25
min_frame = 10

# classes to which evaluate the speed
admitted_classes = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "bus",
    "truck",
    "cat",
    "dog",
]

# load model for detection and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor

# load model for segmentation
segmodel = DeepLabV3("eval_seq", project_dir=".").cuda()
segmodel.load_state_dict(torch.load("pretrained_models/model_13_2_2_2_epoch_580.pth"))
segmodel.eval()

# Dimensions of the frames needed for the segmentation network
seg_img_w = 1024
seg_img_h = 512

det_colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

videopath = None
coursepath = None

frames_to_wait = 0
frames_to_end = 7500


# Create the homographic matrix
def create_M():
    src = np.float32([[0, 790], [1919, 790], [0, 500], [1919, 500]])
    dst = np.float32([[853, 290], [1067, 290], [0, 0], [1919, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


# Perform the detection of the given frame
def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2), 0), max(int((imw-imh)/2), 0), max(int((imh-imw)/2), 0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


# Perform the segmentation of the given frame.
def segment_image(img, seg_img_w, seg_img_h):
    img = cv2.resize(img, (seg_img_w, seg_img_h),
                     interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

    # normalize the img (with the mean and std for the pretrained ResNet):
    img = img / 255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
    img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
    img = img.astype(np.float32)

    # convert numpy -> torch:
    img = img[np.newaxis]
    img = torch.from_numpy(img)  # (shape: (1, 3, 512, 1024))
    with torch.no_grad():
        img = Variable(img).cuda()  # img = Variable(img).cuda()
        output = segmodel(img) # (shape: (1, num_classes, img_h, img_w))
        output = output.data.cpu().numpy()  # (shape: (1, num_classes, img_h, img_w))
        return output


# pointA == Centroid
# pointB == Car Point
# Return the x and y distances between 2 points
def dist_component(pointA, pointB):
    change_Y = pointB[1] - pointA[1]
    change_X = pointA[0] - pointB[0]
    return change_Y, change_X


# Returns True if the 10x10 neighborhood of the centroid is on the given road mask
def centroid_on_street(centroid, mask):
    w = centroid[0]
    h = centroid[1]
    if True in mask[h-5:h+5, w-5:w+5]:
        return True
    else:
        return False


def parse():
    global videopath, coursepath, FPS, min_frame, frames_to_wait, frames_to_end

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="source path of the video directory")
    parser.add_argument("-fps", "--framepersecond", help="Frame Per Second for the video",
                        type=int, default=25)
    parser.add_argument("-mf", "--minframe", help="Number of frame to compute speed", type=int,
                        default=10)
    parser.add_argument("-sb", "--seconds", help="Second of the video in which start the computation",
                        type=int, default=0)
    parser.add_argument("-se", "--secondsend", help="Second of the video in which end the computation",
                        type=int, default=300)

    args = parser.parse_args()

    if args.path:
        path = args.path
    else:
        path = "videos/02/"

    videopath = path + "video_garmin.avi"
    coursepath = path + "speed_course_coord.txt"

    if args.framepersecond:
        FPS = args.framepersecond

    if args.minframe:
        min_frame = args.minframe

    if args.seconds:
        frames_to_wait = args.seconds * FPS

    if args.secondsend:
        frames_to_end = args.secondsend * FPS

    if frames_to_end <= frames_to_wait:
        frames_to_wait = 0
        frames_to_end = 7500


if __name__ == "__main__":
    parse()

    # Reading the video
    vid = cv2.VideoCapture(videopath)

    # Initiate the tracker
    mot_tracker = Sort()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = vid.read()

    vw = frame.shape[1]
    vh = frame.shape[0]

    outvideo = cv2.VideoWriter(videopath.replace(".avi", "-det.avi"), fourcc, FPS, (vw, vh))
    outseg = cv2.VideoWriter(videopath.replace(".avi", "-seg.avi"), fourcc, FPS, (vw, vh))

    car_point = (int(vw / 2), int(vh - 330))

    # Creating the homographic matrix
    M, Minv = create_M()

    # Computing the transformed centroid of our car to compute the distances.
    cp_transformed = np.array([[car_point[0], car_point[1]]], np.float32)
    cp_transformed = np.array([cp_transformed])
    cp_transformed = cv2.perspectiveTransform(cp_transformed, M)
    cp_transformed = cp_transformed[0][0]

    frames = 0

    # Opening the file for our speeds.
    with open(coursepath, 'r') as f:
        myLines = [line.strip() for line in f]

    # meter per pixel ratio
    px_conv = 2.75/28.3

    # Dictionary of the moving objects
    obj_dict = MODictionary()
    while ret:
        mo_on_street = 0  # moving objects on the road
        mo_out_street = 0  # moving objects not on the road

        if frames % 1 == 0:
            print("COMPUTING FRAME: {}".format(frames))

        if not ret:
            break

        # Reading the speed of our camera in the current frame
        camera_speed = int(myLines[frames].split()[1])

        frames += 1
        # Computing the Detection and Segmentation only in the range of seconds the user specifies.
        if frames > frames_to_wait:
            if frames == frames_to_end:
                break

            # SEGMENTATION
            segout = segment_image(frame, seg_img_w, seg_img_h)
            pred_label_img = np.argmax(segout, axis=1)  # (shape: (1, img_h, img_w))
            pred_label_img = pred_label_img.astype(np.uint8)[0] # (shape: (img_h, img_w))

            pred_label_img_color = label_img_to_color(pred_label_img).astype(np.uint8)
            pred_label_img_color = cv2.resize(pred_label_img_color, (1920, 1080), interpolation=cv2.INTER_NEAREST)

            # OBTAINING THE ROAD IN THE FRAME
            pred_label_img = cv2.resize(pred_label_img, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            mask = pred_label_img == 0
            street = np.zeros_like(pred_label_img_color)
            street[mask] = [128, 64, 128]

            # COMBINATION OF OUR FRAME AND THE SEGMENTED ROAD.
            combined = 0.65*frame + 0.35*street
            combined = combined.astype(np.uint8)

            # DETECTION
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)
            detections = detect_image(pilimg)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
            unpad_h = img_size - pad_y
            unpad_w = img_size - pad_x
            mov_obj = 0
            if detections is not None:
                # Update the tracker.
                tracked_objects = mot_tracker.update(detections.cpu())

                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:

                    cls = classes[int(cls_pred)]
                    mov_obj += 1

                    # compute the speed only of the admitted classes.
                    compute_speed = False
                    if cls in admitted_classes:
                        compute_speed = True

                    # bbox computation in the frame
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                    # centroid of the moving object
                    centroid = (int(x1 + box_w / 2), int(y1 + box_h))

                    # evaluating the position of the moving object on the road.
                    if centroid_on_street(centroid, mask):
                        mo_on_street += 1
                    else:
                        mo_out_street += 1

                    obj = None
                    km_h_from_camera = 0  # speed of the moving objects in km/h
                    x_s = 0  # x speed for debugging
                    y_s = 0  # y speed for debugging
                    if compute_speed:
                        # Finding the coordinates of the centroid of our car in the transformed frame.
                        centroid_transformed = np.array([[centroid[0], centroid[1]]], np.float32)
                        centroid_transformed = np.array([centroid_transformed])
                        centroid_transformed = cv2.perspectiveTransform(centroid_transformed, M)[0][0]

                        # Finding the distances in pixels.
                        dist_y, dist_x = dist_component(centroid_transformed, cp_transformed)

                        # Distances in meters.
                        met_dist_y = px_conv * dist_y
                        met_dist_x = px_conv * dist_x

                        # Computing the trajectory point.
                        trajectory_point = (int(x1 + box_w / 2), int(y1 + box_h / 2))

                        if not obj_dict.id_in(obj_id):
                            # Creating the instance of MovingObject class and adding it to the dictionary.
                            obj = MovingObject(obj_id, met_dist_x, met_dist_y)
                            obj_dict.add(obj)

                        obj = obj_dict.get(obj_id)

                        # Compute the speeds each min_frame frames.
                        if obj.frame_counter % min_frame == 0:
                            prev_distance_x = obj.dist_x
                            prev_distance_y = obj.dist_y

                            # computing the y speed and summing our velocity
                            speed_from_camera_y = (met_dist_y - prev_distance_y)/(min_frame/FPS)
                            obj.speed_y = speed_from_camera_y
                            speed_from_camera_y += (camera_speed / 3.6)

                            # Computing the x speed
                            speed_from_camera_x = (met_dist_x - prev_distance_x)/(min_frame/FPS)
                            obj.speed_x = speed_from_camera_x

                            # Update the distances
                            obj.dist_x = met_dist_x
                            obj.dist_y = met_dist_y

                            # Update the speed of the object
                            speed_from_camera = np.sqrt(speed_from_camera_x ** 2 + speed_from_camera_y ** 2)

                            if obj.speed_counter == 5:
                                obj.speed = speed_from_camera
                                obj.speed_counter = 0
                            else:
                                obj.speed += speed_from_camera

                            obj.speed_counter += 1

                            if speed_from_camera_y < 1:
                                obj.speed = 0

                            if obj.speed != 0:
                                obj.trajectory.append(trajectory_point)

                        # Aggiungo un frame
                        obj.frame_counter += 1

                        # Drawing the trajectory for 5 seconds
                        if 1 * (FPS / min_frame) < len(obj.trajectory) <= 5 * (FPS / min_frame):
                            points = obj.trajectory
                            for i in range(len(points) - 1):
                                prec = i - 1
                                cv2.circle(combined, points[prec + 1], 4, (255, 0, 0), thickness=5)
                                cv2.circle(combined, points[i + 1], 4, (255, 0, 0), thickness=5)
                                cv2.line(combined, points[prec + 1], points[i + 1], color, thickness=3)

                        # m/s to km/h
                        km_h_from_camera = (obj.speed / obj.speed_counter) * 3.6
                        x_s = obj.speed_x
                        y_s = obj.speed_y

                        # Drawing the centroid of the moving object
                        cv2.circle(combined, centroid, 2, (0, 0, 255), thickness=5)

                    color = det_colors[int(obj_id) % len(det_colors)]

                    # PRINTING THE BOUNDING BOX AND SPEED
                    cv2.rectangle(combined, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(combined, (x1, y1-35), (x1+len(cls)*19+390, y1), color, -1)

                    cv2.putText(combined, cls + "-" + str(int(obj_id)) + " speed: "
                                + str(round(km_h_from_camera, 3)) + "km/h",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # WRITING INFO ON THE FRAME
            cv2.circle(combined, car_point, 5, (0, 0, 255), thickness=5)
            cv2.rectangle(combined, (1770, 1000), (1920, 1065), (0, 255, 0), -1)
            cv2.putText(combined, str(camera_speed) + "km/h", (1780, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            cv2.putText(combined, "Moving Objects: {}".format(mov_obj), (0, 50), 0, 2, (0, 0, 0), thickness=4)
            cv2.putText(combined, "Obj on Road: {}".format(mo_on_street), (0, 150), 0, 2, (0, 0, 0), thickness=4)
            cv2.putText(combined, "Obj not on Road: {}".format(mo_out_street), (0, 250), 0, 2, (0, 0, 0), thickness=4)

            cv2.putText(combined, "Moving Objects: {}".format(mov_obj), (0, 50), 0, 2, (255, 255, 255), thickness=2)
            cv2.putText(combined, "Obj on Road: {}".format(mo_on_street), (0, 150), 0, 2, (255, 255, 255), thickness=2)
            cv2.putText(combined, "Obj not on Road: {}".format(mo_out_street), (0, 250), 0, 2, (255, 255, 255), thickness=2)

            outvideo.write(combined)
            outseg.write(pred_label_img_color)

        ret, frame = vid.read()

    outvideo.release()
    outseg.release()
