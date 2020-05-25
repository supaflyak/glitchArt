import cv2 as cv
import numpy as np
import imutils


class Image_Container():
    def __init__(self, frame=None, buffer_size=3, fade_factor=.4):
        self.column_glitch = False
        self.fade = False
        self.detection = False
        self.straighten_on_face = False
        self.frame = frame
        self.frame_count = 0
        self.frame_buffer = []
        self.processed_frame_buffer = []
        self.fade_factor = fade_factor
        self.buffer_size = buffer_size
        self.col_glitch_random_arr = []
        self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
        self.rotation_angle = 0
        self.max_rotation_angle = np.pi/12

    def process_images(self):
        if self.detection:
            self.face_detect(self.frame)
        if self.straighten_on_face:
            self.straighten_process(self.frame)
        if self.column_glitch:
            self.column_glitch_process()
        if self.fade:
            self.fade_process()
        return self.frame

    def increment_frame_count(self):
        self.frame_count += 1
        if len(self.frame_buffer) < self.buffer_size:
            self.frame_buffer.append(self.frame)
            self.processed_frame_buffer.append(
                (self.frame * (1 - self.fade_factor) / len(self.frame_buffer)).astype(np.uint8))
        else:
            self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
            self.frame_buffer[-1] = self.frame
            self.processed_frame_buffer = np.roll(self.processed_frame_buffer, -1, axis=0)
            self.processed_frame_buffer[-1] = (self.frame * (1 - self.fade_factor) / len(self.frame_buffer))
        return self.frame_count

    def column_glitch_process(self, change_freq=25, col_chunk=30):
        copy_frame = self.frame.copy()
        # Use the previous random columns
        if self.frame_count % change_freq > 0 and self.col_glitch_random_arr.__len__() > 0:
            for col in range(0, int(self.frame.shape[1]/col_chunk)):
                glitch_offset = self.col_glitch_random_arr[col]
                if glitch_offset >=0:
                    actual_base_col = col * col_chunk
                    for sub_col in range(0, col_chunk):
                        self.frame[:, actual_base_col + sub_col, :] = copy_frame[:, glitch_offset + sub_col, :]
        # Create new random columns
        else:
            self.col_glitch_random_arr = []
            for col in range(0, int(self.frame.shape[1]/col_chunk)):
                if np.random.randint(0, 2) == 0:
                    actual_base_col = col * col_chunk
                    glitch_offset = np.random.randint(1, int(self.frame.shape[1] - col_chunk))
                    self.col_glitch_random_arr.append(glitch_offset)
                    for sub_col in range(0, col_chunk):
                        self.frame[:, actual_base_col + sub_col, :] = copy_frame[:, glitch_offset + sub_col, :]
                else:
                    self.col_glitch_random_arr.append(-1)
        return self.frame

    def fade_process(self, base_fade=.6):
        if len(self.frame_buffer) > 1:
            self.frame = (self.frame * base_fade).astype(np.uint16)
            for prev_frame in self.processed_frame_buffer:
                self.frame = np.clip(self.frame + prev_frame.astype(np.uint16), 0, 255)
            self.frame = self.frame.astype(np.uint8)
            self.frame_buffer[-1] = self.frame
        return self.frame

    def face_detect(self, image):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            self.frame = cv.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = self.frame[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    def straighten_process(self, image):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # self.frame = cv.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            # roi_color = self.frame[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            # If there are two eyes, update the rotation angle
            if len(eyes) == 2:
                new_angle = -1 * np.arctan((eyes[1][1] - eyes[0][1]) / (eyes[1][0] - eyes[0][0]))
                if new_angle > np.absolute(np.absolute(self.rotation_angle) - np.absolute(self.max_rotation_angle)):
                    if new_angle > 0:
                        self.rotation_angle += self.max_rotation_angle
                    if new_angle < 0:
                        self.rotation_angle += self.max_rotation_angle
                else:
                    self.rotation_angle = new_angle
            # apply the new or old rotation angle to the image
            self.frame = imutils.rotate(self.frame, self.rotation_angle)

def main():
    cap = cv.VideoCapture(0)
    image = Image_Container()

    while True:
        retval, image.frame = cap.read()
        if retval:
            image.increment_frame_count()
        image.process_images()
        cv.imshow("Display", image.frame)
        key = cv.waitKey(1)
        if key:
            key = key & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                image.column_glitch = not image.column_glitch
                print("column glitches: " + str(image.column_glitch))
            elif key == ord('2'):
                image.fade = not image.fade
                print("fade: " + str(image.fade))
            elif key == ord('3'):
                image.detection = not image.detection
                print("face detection: " + str(image.detection))
                if image.straighten_on_face:
                    image.straighten_on_face = not image.straighten_on_face
                    print("straighten on face: " + str(image.straighten_on_face))
            elif key == ord('4'):
                image.straighten_on_face = not image.straighten_on_face
                print("straighten on face: " + str(image.straighten_on_face))
                if image.detection:
                    image.detection = not image.detection
                    print("face detection: " + str(image.detection))
    cap.release()
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
