from models.facealignment.FastFaceAlignemt import generateFastFan
import cv2
import argparse
import time

from models.facedetect.vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from models.facedetect.vision.ssd.mb_tiny_fd import create_mb_tiny_fd_predictor
# Run the 3D face alignment on a test image, without CUDA.

parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--model_path', default="./models/facealignment/FastFAN.pth", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--video_path', default="./data/orin/Oscars1.mp4", type=str,
                    help='video dir')
parser.add_argument('--save_path', default="./data/results/Oscars.mp4", type=str,
                    help='ouput video dir')
parser.add_argument('--test_device', default=0, type=int,
                    help='cuda:0')
args = parser.parse_args()


if __name__ == '__main__':


    landmarksDetector = generateFastFan(modelPath=args.model_path, deviceID=args.test_device)

    #face detector
    candidate_size = 1500
    threshold = 0.98
    label_path = "./models/facedetect/voc-model-labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=0)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=args.test_device)
    model_path = "./models/facedetect/version-slim-320.pth"
    net.load(model_path)


    cap = cv2.VideoCapture(args.video_path)
    count=0
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(args.save_path, -1, fps, size)
    timec = 0
    timep = 0
    count = 0
    while cap.isOpened():
        ret, img = cap.read()
        #detect faces
        start = time.time()

        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(img, candidate_size / 2, threshold)



        end = time.time()
        #print("face Execution Time: ", end - start)

        faces = []
        for box in boxes:
            w = box[2]-box[0]
            faces.append([box[0], box[1],box[2], box[3]])
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

        #detect landmarks
        start = time.time()
        landmarks = landmarksDetector.getLandmarksFromFrame(img,faces)
        end = time.time()
        count = count + 1
        timec = timec + (end - start)
        if count % 25 == 0:
            timep = timec
            timec = 0
        if count > 25:
            t = '{0:.2f}'.format(timep)
            text = '{}{}{}'.format('25-frame inference uses ', t, ' s.')
            cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        for facel in landmarks:
            for point in facel:
                cv2.circle(img, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=1)
        videoWriter.write(img)






