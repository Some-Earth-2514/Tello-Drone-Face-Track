from pyimagesearch.objcenter import ObjCenter
import cv2
from pyimagesearch.pid import PID
from djitellopy import Tello, TelloException
import signal
import imutils
import time
from datetime import datetime
from multiprocessing import Manager, Process, Pipe, Event

tello = None
video_writer = None

def signal_handler(sig, frame, exit_event_obj):
    print("Signal Handler Triggered")
    exit_event_obj.set()

    global tello, video_writer
    if tello:
        try:
            print("Attempting Tello cleanup...")
            tello.streamoff()
            tello.land()
            print("Tello cleanup commands sent.")
        except Exception as e:
            print(f"Error during Tello cleanup: {e}")

    if video_writer:
        try:
            print("Attempting VideoWriter release...")
            video_writer.release()
            print("VideoWriter released.")
        except Exception as e:
            print(f"Error releasing VideoWriter: {e}")

def track_face_in_video_feed(exit_event, show_video_conn, video_writer_conn, run_pid, track_face, fly=False,
                             max_speed_limit=40):
    """

    :param exit_event: Multiprocessing Event.  When set, this event indicates that the process should stop.
    :type exit_event:
    :param show_video_conn: Pipe to send video frames to the process that will show the video
    :type show_video_conn: multiprocessing Pipe
    :param video_writer_conn: Pipe to send video frames to the process that will save the video frames
    :type video_writer_conn: multiprocessing Pipe
    :param run_pid: Flag to indicate whether the PID controllers should be run.
    :type run_pid: bool
    :param track_face: Flag to indicate whether face tracking should be used to move the drone
    :type track_face: bool
    :param fly: Flag used to indicate whether the drone should fly.  False is useful when you just want see the video stream.
    :type fly: bool
    :param max_speed_limit: Maximum speed that the drone will send as a command.
    :type max_speed_limit: int
    :return: None
    :rtype:
    """
    global tello
    from functools import partial
    local_signal_handler = partial(signal_handler, exit_event_obj=exit_event)
    signal.signal(signal.SIGINT, local_signal_handler)
    signal.signal(signal.SIGTERM, local_signal_handler)

    max_speed_threshold = max_speed_limit

    tello = Tello()

    tello.connect()

    try:
        battery_level = tello.get_battery()
        print(f"Current Battery Level: {battery_level}%")
        if battery_level < 20:
            print("Error: Battery low! Please charge the drone.")
            exit_event.set()
            return
    except Exception as e:
        print(f"Error checking battery: {e}")
        exit_event.set()
        return

    tello.streamon()
    frame_read = tello.get_frame_read()

    if fly:
        print("Attempting takeoff...")
    time.sleep(2)
    try:
        tello.takeoff()
        print("Takeoff successful.")
        time.sleep(1)
        print("Attempting move up...")
        tello.move_up(20)
        print("Move up successful.")

    except TelloException as e:
        print(f"ERROR during flight initialization: {e}")
        print("Aborting flight commands.")
        exit_event.set()

    except Exception as e:
        print(f"An unexpected error occurred during flight initialization: {e}")
        exit_event.set()

    face_center = ObjCenter("./haarcascade_frontalface_default.xml")
    pan_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    tilt_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    pan_pid.initialize()
    tilt_pid.initialize()

    while not exit_event.is_set():
        frame = frame_read.frame

        frame = imutils.resize(frame, width=400)
        H, W, _ = frame.shape

        centerX = W // 2
        centerY = H // 2

        cv2.circle(frame, center=(centerX, centerY), radius=5, color=(0, 0, 255), thickness=-1)

        frame_center = (centerX, centerY)
        objectLoc = face_center.update(frame, frameCenter=None)

        ((objX, objY), rect, d) = objectLoc
        if d > 25 or d == -1:
            if track_face and fly:
                tello.send_rc_control(0, 0, 0, 0)
            continue

        if rect is not None:
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

            cv2.circle(frame, center=(objX, objY), radius=5, color=(255, 0, 0), thickness=-1)

            cv2.arrowedLine(frame, frame_center, (objX, objY), color=(0, 255, 0), thickness=2)

            if run_pid:
                pan_error = centerX - objX
                pan_update = pan_pid.update(pan_error, sleep=0)

                tilt_error = centerY - objY
                tilt_update = tilt_pid.update(tilt_error, sleep=0)

                cv2.putText(frame, f"X Error: {pan_error} PID: {pan_update:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(frame, f"Y Error: {tilt_error} PID: {tilt_update:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 2, cv2.LINE_AA)

                if pan_update > max_speed_threshold:
                    pan_update = max_speed_threshold
                elif pan_update < -max_speed_threshold:
                    pan_update = -max_speed_threshold

                pan_update = pan_update * -1

                if tilt_update > max_speed_threshold:
                    tilt_update = max_speed_threshold
                elif tilt_update < -max_speed_threshold:
                    tilt_update = -max_speed_threshold

                print(int(pan_update), int(tilt_update))
                if track_face and fly:
                    tello.send_rc_control(int(pan_update // 3), 0, int(tilt_update // 2), 0)

        show_video_conn.send(frame)
        video_writer_conn.send(frame)
    print("Exiting track_face_in_video_feed loop.")
    local_signal_handler(None, None)

def show_video(exit_event, pipe_conn):
    from functools import partial
    local_signal_handler = partial(signal_handler, exit_event_obj=exit_event)
    signal.signal(signal.SIGINT, local_signal_handler)
    signal.signal(signal.SIGTERM, local_signal_handler)

    print("Show video process started.")
    while not exit_event.is_set():
        try:
            if pipe_conn.poll(0.01):
                frame = pipe_conn.recv()
                cv2.imshow("Drone Face Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed, setting exit event.")
                exit_event.set()
                break

        except (EOFError, BrokenPipeError):
            print("Show video pipe closed.")
            break
        except Exception as e:
            print(f"Error in show_video: {e}")
            break

    print("Exiting show_video loop.")
    cv2.destroyAllWindows()
    local_signal_handler(None, None)


def video_recorder(exit_event, pipe_conn, save_video, height=300, width=400):
    global video_writer
    video_writer = None

    from functools import partial
    local_signal_handler = partial(signal_handler, exit_event_obj=exit_event)
    signal.signal(signal.SIGINT, local_signal_handler)
    signal.signal(signal.SIGTERM, local_signal_handler)

    print("Video recorder process started.")
    if save_video:
        try:
            video_file = f"video_{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}.mp4"
            video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            if not video_writer.isOpened():
                print(f"Error: Could not open video writer for {video_file}")
                exit_event.set()
            else:
                print(f"Recording video to {video_file}")
        except Exception as e:
            print(f"Error initializing VideoWriter: {e}")
            exit_event.set()

    while not exit_event.is_set():
        try:
            if pipe_conn.poll(0.01):
                frame = pipe_conn.recv()
                if video_writer and video_writer.isOpened():
                    video_writer.write(frame)
        except (EOFError, BrokenPipeError):
            print("Video recorder pipe closed.")
            break
        except Exception as e:
            print(f"Error in video_recorder loop: {e}")
            break

    print("Exiting video_recorder loop.")
    if video_writer and video_writer.isOpened():
        print("Releasing video writer...")
        video_writer.release()
        print("Video writer released.")
    local_signal_handler(None, None)

if __name__ == '__main__':
    run_pid = True
    track_face = True
    save_video = True
    fly = True

    parent_conn, child_conn = Pipe()
    parent2_conn, child2_conn = Pipe()

    exit_event = Event()

    from functools import partial
    main_signal_handler = partial(signal_handler, exit_event_obj=exit_event)
    signal.signal(signal.SIGINT, main_signal_handler)
    signal.signal(signal.SIGTERM, main_signal_handler)

    with Manager() as manager:
        print("Starting processes...")
        p1 = Process(target=track_face_in_video_feed,
                     args=(exit_event, child_conn, child2_conn, run_pid, track_face, fly,))
        p2 = Process(target=show_video, args=(exit_event, parent_conn,))
        p3 = Process(target=video_recorder, args=(exit_event, parent2_conn, save_video,))

        p2.start()
        p3.start()
        p1.start()

        print("Waiting for main process (p1) to join...")
        p1.join()
        print("Main process (p1) joined.")

        if not exit_event.is_set():
            print("p1 exited unexpectedly, setting exit_event for others.")
            exit_event.set()

        print("Waiting for display process (p2) to join...")
        p2.join(timeout=5)
        print("Waiting for recorder process (p3) to join...")
        p3.join(timeout=5)

        if p2.is_alive():
            print("Warning: Display process (p2) did not exit gracefully, terminating.")
            p2.terminate()
        if p3.is_alive():
            print("Warning: Recorder process (p3) did not exit gracefully, terminating.")
            p3.terminate()

    print("Complete...")