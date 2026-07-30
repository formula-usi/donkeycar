#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt


video_source = "http://piracerpro-8.local:8890/video"


def main():
    camera = cv2.VideoCapture(video_source)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open video source: {video_source}"
        )

    plt.ion()

    figure, axis = plt.subplots()
    axis.set_title("Camera Feed")
    axis.axis("off")

    image_plot = None

    try:
        while plt.fignum_exists(figure.number):
            success, frame = camera.read()

            if not success:
                print("Could not read video frame.")
                break

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            if image_plot is None:
                image_plot = axis.imshow(frame_rgb)
                figure.tight_layout()
            else:
                image_plot.set_data(frame_rgb)

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nVideo visualization interrupted.")

    finally:
        camera.release()
        plt.ioff()
        plt.close("all")

        print("Video source closed.")


if __name__ == "__main__":
    main()