


import cv2
import numpy as np
import os
import config
from component.ball_tracking import *

path = "testing/"
output_dir = "working dir/image/"
balls = []
image_shape = [1080, 1920]
xmin = int(image_shape[1] * config.xmin)
xmax = int(image_shape[1] * config.xmax)
ymin = int(image_shape[0] * config.ymin)
ymax = int(image_shape[0] * config.ymax)
previous_img = np.zeros((ymax - ymin, xmax - xmin))
imageCount = 0

def find_endpoints(contour):
    """Find the endpoints of a contour using the convex hull."""
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_points = hull[:, 0, :]
    return hull_points

def join_endpoints(image, endpoints):
    """Join the endpoints that are closest to each other."""
    if len(endpoints) < 2:
        return image

    # Find all pairs of endpoints
    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dist = np.linalg.norm(endpoints[i] - endpoints[j])
            pairs.append((dist, endpoints[i], endpoints[j]))

    # Sort pairs by distance
    pairs.sort(key=lambda x: x[0])

    # Join the closest pairs
    for _, pt1, pt2 in pairs:
        cv2.line(image, tuple(pt1), tuple(pt2), 255, 1)
    
    return image

def detect_triangles(endpoints):
    """Detect triangles from the given endpoints."""
    triangles = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            for k in range(j + 1, len(endpoints)):
                pt1, pt2, pt3 = endpoints[i], endpoints[j], endpoints[k]
                # Check if the points form a triangle
                if not np.isclose(np.linalg.norm(pt1 - pt2) + np.linalg.norm(pt2 - pt3), np.linalg.norm(pt1 - pt3) + np.linalg.norm(pt1 - pt2)):
                    # Calculate the area of the triangle
                    tri_area = cv2.contourArea(np.array([pt1, pt2, pt3], dtype=np.int32))
                    if tri_area > 0:
                        triangles.append((tri_area, [pt1, pt2, pt3]))
    # Sort triangles by area (largest first)
    triangles.sort(reverse=True, key=lambda x: x[0])
    return triangles

def perform_additional_operations(binary_image):
    """Apply additional operations to the binary image."""
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)

    # Find contours on the cleaned image
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours for visualization
    contour_image = np.zeros_like(cleaned_image)
    cv2.drawContours(contour_image, contours, -1, 255, 1)

    return cleaned_image, contour_image, contours


def classify_triangle(triangle, length_threshold):
    """Classify the points of a triangle."""
    if len(triangle) != 3:
        return None, None
    
    # Calculate the lengths of the sides
    dists = [
        np.linalg.norm(np.array(triangle[0]) - np.array(triangle[1])),
        np.linalg.norm(np.array(triangle[1]) - np.array(triangle[2])),
        np.linalg.norm(np.array(triangle[2]) - np.array(triangle[0]))
    ]
    
    # Find the point with the highest x-value
    ball = max(triangle, key=lambda p: p[0])
    
    # Find the base (shortest side)
    base = min(dists)
    
    # Classify as stick if the base length is below the threshold
    if base < length_threshold:
        return None, triangle  # Return None for ball, and all three points for stick
    else:
        stick_points = [p for p in triangle if not np.array_equal(p, ball)]
        if len(stick_points) == 2:
            return ball, stick_points
    return None, None

# Define the minimum area threshold for contours
min_contour_area = 200  # Adjust this value as needed

# Define the length threshold for distinguishing between ball and stick
length_threshold = 50  # Adjust this value as needed

for filename in os.listdir(path):
    image = cv2.imread(path + filename)
    image_with_triangle, IntestImage = regionOfIntrest(image, xmin, xmax, ymin, ymax)
    binary_image = convert_rgb_binery_image(image_with_triangle)
    
    if imageCount == 0:
        difference_mask = np.zeros_like(binary_image, dtype=np.uint8)
    else:
        difference_mask = np.maximum(np.abs(binary_image - previous_binary_mask), 0)
    
    difference_mask[difference_mask < 175] = 0
    difference_mask[difference_mask >= 175] = 255

    difference_mask = openingOperation(difference_mask)
    difference_mask = closingOperation(difference_mask)

    cv2.imwrite(output_dir + "closingOperation_" + filename, difference_mask)
    
    # Perform additional operations
    cleaned_image, contour_image, contours = perform_additional_operations(difference_mask)
    
    # Find endpoints and draw triangles
    endpoints = []
    for contour in contours:
        # Filter out contours with area below the threshold
        if cv2.contourArea(contour) >= min_contour_area:
            hull_points = find_endpoints(contour)
            endpoints.extend(hull_points)

    # Create an empty mask to draw the joined endpoints
    joined_mask = np.zeros_like(difference_mask)
    joined_mask = join_endpoints(joined_mask, endpoints)
    
    cv2.imwrite(output_dir + "joined_endpoints_" + filename, joined_mask)
    
    # Detect triangles from the endpoints
    triangles = detect_triangles(endpoints)
    
    if triangles:
        # Select the largest triangle
        largest_triangle = triangles[0][1]  # largest_triangle is the points of the largest triangle
        ball, stick_points = classify_triangle(largest_triangle, length_threshold)
        
        if ball is not None:
            # Draw the largest triangle
            cv2.polylines(image_with_triangle, [np.array(largest_triangle, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw the ball and stick points
            cv2.circle(image_with_triangle, tuple(ball), 5, (0, 0, 255), -1)  # Ball in red
            for pt in stick_points:
                cv2.circle(image_with_triangle, tuple(pt), 5, (255, 0, 0), -1)  # Stick in blue
                cv2.line(image_with_triangle, tuple(stick_points[0]), tuple(stick_points[1]), (0, 255, 0), 2)  # Stick line in green
        else:
            # Draw the stick if no ball is identified
            for pt1, pt2 in zip(stick_points[:-1], stick_points[1:]):
                cv2.line(image_with_triangle, tuple(pt1), tuple(pt2), (0, 255, 0), 2)  # Stick line in green
    
    cv2.imwrite(output_dir + "classified_" + filename, image_with_triangle)

    previous_binary_mask = binary_image
    imageCount += 1
