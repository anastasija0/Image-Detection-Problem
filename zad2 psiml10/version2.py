import imageio
import numpy as np
import time

def detect_circles(image):
    shapes = []
    gray = np.mean(image, axis=2)

    threshold = 100
    binary_image = np.where(gray > threshold, 1, 0)

    visited = []
    height, width = gray.shape
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1 and (i, j) not in visited:
                shape = dfs(binary_image, i, j) #vraca obojene delove oblika
                shapes.append(shape)
                visited.extend(shape)
    #print(shapes)
    #print(visited)
    #shape_centers = []
    circles = []
    rectangles = []
    for shape in shapes:
        #num_points = len(shape)
        x_min = 5000
        y_min = 5000
        x_max = 0
        y_max = 0
        for point in shape:
            x = point[0]
            y = point[1]
            if (x < x_min):
                x_min = x
            if (x > x_max):
                x_max = x
            if (y < y_min):
                y_min = y
            if (y > y_max):
                y_max = y
        center_x = (x_min + x_max)/2
        center_y = (y_min + y_max)/2
        if binary_image[int(center_x), int(center_y)] == 1:
            radius = (x_max - x_min)/2
            circles.append((center_x, center_y, radius))
        elif binary_image[int(center_x), int(center_y)] == 0:
            rectangles.append((center_x, center_y))

    leftmost_circle = min(circles, key=lambda c: c[1] - c[2])
    print(int(leftmost_circle[0]), int(leftmost_circle[1]))
    #print(circles)
    for r in rectangles:
        print(int(r[0]), int(r[1]))
    #return circles

def dfs(binary_image, i, j):
    stack = [(i, j)]
    colored = set()
    while stack:
        i, j = stack.pop()
        if (i, j) in colored or binary_image[i, j] == 0:
            continue
        colored.add((i, j))
        stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
    return colored

def find_bounces_video(video_path):
    frame_rate = 30
    vid = imageio.get_reader(video_path)
    frame_duration = 1.0 / frame_rate
    frame = vid.get_next_data()
    bounce_count = 0
    while frame:
        gray = np.mean(frame, axis=2)
        height,length = gray.shape
        shapes = detect_circles(frame, 1)
        x_r, y_r, height_r, length_r = 0,0,0,0
        for shape in shapes:
            if shape[3]!=0:
                x_r = shape[0]
                y_r = shape[1]
                height_r = shape[2]
                length_r = shape[3]
        for shape in shapes:
            if shape[3]==0:
                x = shape[0]
                y = shape[1]
                r = shape[2]
            if (x-r == 0) or (x+r == height) or (y-r == 0) or (y+r == length):
                bounce_count+=1
            elif (x-r <= x_r + height_r/2) or (x+r >= x_r - height_r/2) or (y-r <= y_r+length_r/2) or (y+r >= y_r-length_r):
                bounce_count+=1
        time.sleep(frame_duration)
        frame = vid.get_next_data()
    print(bounce_count)

def find_leftmost_circle(video_path):
    leftmost_circle = None
    frame_rate = 30
    vid = imageio.get_reader(video_path)
    frame_duration = 1.0 / frame_rate
    frame_idx = 0
    while not frame_idx:
        frame = vid.get_next_data()
        circles = detect_circles(frame)
        time.sleep(frame_duration)
        frame_idx += 1



if __name__ == "__main__":
  (video_path, audio_path) = input().split()
  find_leftmost_circle(video_path)
  find_bounces_video(video_path)

