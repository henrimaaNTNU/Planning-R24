import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from time import time


#Forbedringspotensiale:
#Korter ned kjøretid gjennom bruk av dictionary?
#Korter ned kjøretid gjennom bruk av numpy arrays i stedet for lister


class TrackPlotter:
    def __init__(self):
        self.left_cones, self.right_cones = self.generate_parallel_curves()
        self.midpoints = [self.calculate_midpoint(self.left_cones[i], self.right_cones[i]) for i in range(len(self.left_cones))]
        self.T_n_points = []  # Initialize the T_n_points list
        time1 = time()
        self.optimized_points = self.calculate_optimized_points(self.midpoints)  # Ensure midpoints are passed here
        time3 = time()
        self.refined_points = self.refine_track_for_distance(self.optimized_points, max_distance=1.5)
        time2 = time()
        print("Time used: __init__", time2-time1)
        print("Time used: calculate_optimized_points", time3-time1)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-8, 45)
        self.ax.set_ylim(-8, 34)
        self.plot_track()

    def generate_parallel_curves(self):
        # Define the x-coordinates
        #x_values = np.linspace(-20, 20, 50)  # Adjust the range and number of points as needed

        # Define the sine wave for the left curve
        #amplitude_left = 1.25  # Adjust amplitude as needed
        #left_cones = [(x, amplitude_left * np.sin(x)+1) for x in x_values]

        # Define the sine wave for the right curve
        #amplitude_right = 1.25 # Adjust amplitude as needed
        #right_cones = [(x, amplitude_right * np.sin(x)+3.5) for x in x_values]
        points = [
            (5, 5), (4, 9), (7, 7), (5, 11), (10, 9), (8, 13), (13, 10), (12, 14),
    (16, 10), (17, 14), (19, 9), (21, 12), (20, 6), (24, 9), (20, 3), (25, 5),
    (22, 0), (26, 2), (25, -2), (29, 2), (29, -2), (32, 2), (34, -2), (35, 3),
    (38, 0), (37, 5), (41, 3), (38, 8), (42, 8), (37, 12), (42, 13), (34, 15),
    (39, 17), (31, 18), (35, 21), (27, 19), (31, 23), (24, 19), (26.5, 24),
    (22, 18), (22, 23), (20, 18), (20, 23), (18, 19), (18, 24), (15, 21), (16, 27),
    (13, 24), (12, 30), (10, 25), (7, 30), (8, 25), (2, 27), (6, 24), (0, 23),
    (5, 21), (0, 20), (4, 17), (-2, 17), (2, 14), (-6, 14), (-1, 12), (-6, 9),
    (-1, 10), (-4, 6), (0, 9), (1, 4), (2, 8), (5, 5), (4, 9), (7, 7), (5, 11), (10, 9),(8, 13),(13, 10), (12, 14),
    (16, 10), (17, 14), (19, 9)
        ]

        # Separating x and y values and converting to integers
        left_cones = [int(x) for x, y in points]
        right_cones = [int(y) for x, y in points]

        left_curve = [(left_cones[2*i], right_cones[2*i]) for i in range(len(left_cones)//2)]
        right_curve = [(left_cones[2*i+1], right_cones[2*i+1]) for i in range(len(left_cones)//2)]

        return left_curve, right_curve

    def calculate_midpoint(self, p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    

    def calculate_optimzied_point(self, p1, p2):
        time1 = time()
        if np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < 0.1:
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        else:
            # Check intersection with each line segment of the track border
            for i in range(len(self.left_cones) - 1):
                left_segment_p1 = self.left_cones[i]
                left_segment_p2 = self.left_cones[i + 1]
                right_segment_p1 = self.right_cones[i]
                right_segment_p2 = self.right_cones[i + 1]

                left_intersection = self.calculate_intersection(p1, p2, left_segment_p1, left_segment_p2)
                right_intersection = self.calculate_intersection(p1, p2, right_segment_p1, right_segment_p2)

                # Check if the intersection is valid (on the segment)
                if left_intersection and self.is_point_on_segment(left_intersection, left_segment_p1, left_segment_p2):
                    return left_intersection
                elif right_intersection and self.is_point_on_segment(right_intersection, right_segment_p1, right_segment_p2):
                    return right_intersection

        # If no valid intersection is found, return the midpoint
        time2 = time()
        print("Time used: calculate_o_point", time2-time1)
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def is_point_on_segment(self, point, segment_p1, segment_p2):
        """Check if the point lies on the line segment."""
        line_len = euclidean(segment_p1, segment_p2)
        len1 = euclidean(point, segment_p1)
        len2 = euclidean(point, segment_p2)
        return abs(line_len - (len1 + len2)) < 1e-6

    def calculate_optimized_points(self, midpoints):
        time1 = time()
        n = 0
        optimized_points = []
        p1, p2, p3 = 1/9, 2/3, 2/9  # Example adjustment
        for i in range(len(midpoints) - 2):
            angle_n = self.calculate_angle(midpoints[i], midpoints[i + 1])
            angle_n_plus_1 = self.calculate_angle(midpoints[i + 1], midpoints[i + 2])
            optimized_angle = p1 * angle_n + p2 * angle_n_plus_1 + p3 * angle_n

            dx = np.cos(optimized_angle)
            dy = np.sin(optimized_angle)
            line1_p2 = (midpoints[i][0] + dx, midpoints[i][1] + dy)

            normal_angle = self.calculate_angle(midpoints[i + 1], midpoints[i + 2]) + np.pi / 2
            normal_dx = np.cos(normal_angle)
            normal_dy = np.sin(normal_angle)
            line2_p2 = (midpoints[i + 1][0] + normal_dx, midpoints[i + 1][1] + normal_dy)

            T_n_plus_1 = self.calculate_intersection(midpoints[i], line1_p2, midpoints[i + 1], line2_p2)

            
            if T_n_plus_1 and self.is_point_within_track(T_n_plus_1):
                self.T_n_points.append(T_n_plus_1)
                parameter = self.parameter_for_optimized_point(midpoints[i + 1], T_n_plus_1)
                optimized_point = (
                    midpoints[i + 1][0] + parameter * (T_n_plus_1[0] - midpoints[i + 1][0]),
                    midpoints[i + 1][1] + parameter * (T_n_plus_1[1] - midpoints[i + 1][1])
                )
                optimized_points.append(optimized_point)
            else:
                # optimized_points.append(self.calculate_midpoint(midpoints[i + 1], midpoints[i + 2])) # idea 1: GOOD
                # optimized_points.append(midpoints[i + 1])                                            # idea 2: DECENT/BAD
                # optimized_points.append(self.calculate_midpoint(midpoints[i], midpoints[i + 1]))     # idea 3: GOOD 
                # optimized_points.append(self.calculate_midpoint(midpoints[i], midpoints[i + 2]))     # idea 4: GOOD 
                continue                                                                             # idea 5: GOOD/HIGH POTENTIAL 
        time2 = time()
        n += 1
        print("Calculate_o_points ", n, time2-time1)     
        return optimized_points

    def is_point_within_track(self, point):
        """
        Check if a point is within the track boundaries.
        You might need to adjust this method based on your track's specific geometry.
        """
        # For simplicity, let's consider a point to be within the track if it is not too far from any of the midpoints
        max_distance_from_midpoint = 3  # Adjust this threshold as needed
        for midpoint in self.midpoints:
            if euclidean(midpoint, point) <= max_distance_from_midpoint:
                return True
        return False

    def parameter_for_optimized_point(self, midpoint, T_n_point):
        """
        Calculate the parameter for the optimized point on the line segment between the midpoint and T_n_point. 
        The output is a ratio between 0 and 1.
        The ratio is calculated based on the distance between the T_n_point and the nearest border,
        relative to the distance between the midpoint and T_n_point.
        The closer the T_n_point is to the border, the closer the ratio is to 1.
        When the T_n_point is close to the midpoint, the ratio is close to 0.
        """
        # Find the nearest border segment to T_n_point
        nearest_distance_to_border = min(
            self.distance_to_nearest_segment(T_n_point, self.left_cones),
            self.distance_to_nearest_segment(T_n_point, self.right_cones)
        )

        # Calculate the distance from the midpoint to T_n_point
        distance_to_midpoint = euclidean(midpoint, T_n_point)+0.75

        # Avoid division by zero
        if distance_to_midpoint == 0:
            return 0

        # Calculate the ratio
        ratio = nearest_distance_to_border / distance_to_midpoint

        # Ensure the ratio is between 0 and 1
        return min(max(ratio, 0), 1)

    def distance_to_nearest_segment(self, point, segments):
        """
        Calculate the minimum distance from the point to the nearest segment in the segments list.
        """
        time1 = time()

        min_distance = float('inf')
        for i in range(len(segments) - 1):
            segment_distance = self.distance_to_segment(point, segments[i], segments[i + 1])
            min_distance = min(min_distance, segment_distance)
        time2 = time()
        print("Time used: distance_to_nearest_segment", time2-time1)
        return min_distance

    def distance_to_segment(self, point, segment_start, segment_end):
        """
        Calculate the perpendicular distance of 'point' to the line segment (segment_start, segment_end).
        """
        # Vector from start to point
        start_to_point = np.array(point) - np.array(segment_start)
        # Vector from start to end
        start_to_end = np.array(segment_end) - np.array(segment_start)

        # Project start_to_point onto start_to_end
        projection = np.dot(start_to_point, start_to_end) / np.linalg.norm(start_to_end) ** 2

        # Ensure the projection lies within the segment
        projection = max(0, min(1, projection))

        # Find the closest point on the segment
        closest_point = np.array(segment_start) + projection * start_to_end

        # Return the distance from point to closest point on the segment
        return np.linalg.norm(closest_point - point)



    def calculate_angle(self, p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def calculate_intersection(self, line1_p1, line1_p2, line2_p1, line2_p2):
        time1 = time()
        x1, y1 = line1_p1
        x2, y2 = line1_p2
        x3, y3 = line2_p1
        x4, y4 = line2_p2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Lines are parallel

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        time2 = time()
        print("Time used: calculate_intersection", time2-time1)
        return px, py

    #Upsample optimized points, makes it more strict
    def refine_track_for_distance(self, optimized_points, max_distance):
        # Refine the track based on the distance constraint
        time1 = time()
        refined_points = []
        for i in range(len(optimized_points) - 1):
            refined_points.append(optimized_points[i])
            next_point = optimized_points[i + 1]
            while euclidean(refined_points[-1], next_point) > max_distance:
                direction = np.array(next_point) - np.array(refined_points[-1])
                new_point = np.array(refined_points[-1]) + direction / np.linalg.norm(direction) * max_distance
                refined_points.append(new_point.tolist())
        refined_points.append(optimized_points[-1])
        time2 = time()
        print("Time used: refine_track_for_distance", time2-time1)
        return refined_points

    def plot_track(self):
        # Plot cones
        for cone in self.left_cones + self.right_cones:
            self.ax.plot(cone[0], cone[1], 'o', color='orange')

        # Plot midpoints
        for midpoint in self.midpoints:
            self.ax.plot(midpoint[0], midpoint[1], 'x', color='blue', label='Midpoints' if midpoint == self.midpoints[0] else "")

        # Plot T_n_points
        #for T_n_point in self.T_n_points:
        #    self.ax.plot(T_n_point[0], T_n_point[1], 'o', color='purple', label='T_n_points' if T_n_point == self.T_n_points[0] else "")

        # Plot refined points
        #refined_x, refined_y = zip(*self.refined_points)  # Unpack the refined points
        #self.ax.plot(refined_x, refined_y, 'x', color='red', label='Refined Points')


        # Plot optimized points
        optimized_x, optimized_y = zip(*self.optimized_points)  # Unpack the optimized points
        self.ax.plot(optimized_x, optimized_y, '*', color='green', label='Optimized Points')
        
        for i in range(len(self.left_cones) - 1):
            self.ax.plot([self.left_cones[i][0], self.left_cones[i+1][0]], 
                         [self.left_cones[i][1], self.left_cones[i+1][1]], 
                         color='orange', linestyle='-')
            self.ax.plot([self.right_cones[i][0], self.right_cones[i+1][0]], 
                         [self.right_cones[i][1], self.right_cones[i+1][1]], 
                         color='orange', linestyle='-')
        # Fit and plot a smoothing spline to the refined points, if needed

        time1 = time()
        if len(self.refined_points) > 3:
            tck, _ = splprep([np.array(self.refined_points)[:, 0], np.array(self.refined_points)[:, 1]], s=3)
            spline_points = splev(np.linspace(0, 1, len(self.refined_points) * 10), tck)
            self.ax.plot(spline_points[0], spline_points[1], '-', color='red', label='Spline')

            tck, _ = splprep([np.array(self.optimized_points)[:, 0], np.array(self.optimized_points)[:, 1]], s=3)
            spline_points = splev(np.linspace(0, 1, len(self.optimized_points) * 10), tck)
            self.ax.plot(spline_points[0], spline_points[1], '-', color='green', label='Spline')
            
            # Midline spline
            tck, _ = splprep([np.array(self.midpoints)[:, 0], np.array(self.midpoints)[:, 1]], s=3)
            spline_points = splev(np.linspace(0, 1, len(self.midpoints) * 10), tck)
            self.ax.plot(spline_points[0], spline_points[1], '-', color='blue', label='Spline')
        time2 = time()
        print("Spline calc " ,time2-time1)
        # Add legend
        self.ax.legend()
        
        plt.show() #(block=False)


if __name__ == '__main__':
    #t0 = time()
    plotter = TrackPlotter()
    #t1 = time()
    plotter.plot_track()

    #print(t1-t0)


