/*
This C++ module is designed to provide a comprehensive solution for generating and optimizing a racetrack layout based on a set of input coordinates. 
The primary focus of this module is to create an efficient and realistic track for racing simulations or planning purposes. 
The module contains a primary class, TrackPlotter, which encapsulates all the necessary functionalities for this purpose.

The TrackPlotter class is the core of the module, responsible for processing the input points, which represent the placement of midpoints on a racetrack, and generating an optimized path. 
It utilizes various algorithms and mathematical operations to achieve this.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

class TrackPlotter {

    /*
   This class, TrackPlotter, is designed to create and optimize a racetrack layout given a set of coordinates representing the midpoints. 
   The class uses various geometric and mathematical algorithms to compute an optimal path through the track. 
   The optimization process takes into account factors such as the angle of the curves and distances from the edges of the track.

    The implementation is divided into several methods within the TrackPlotter class. Each method is responsible for a specific part of the track creation and optimization process:
    - calculate_angle(): Calculates the angle between two points.
    - calculate_intersection(): Calculates the intersection between two lines.
    - is_point_on_segment(): Checks if a point is on a line segment.
    - calculate_optimized_point(): Calculates the optimized point between two points.
    - calculate_optimized_points(): Optimizes the track layout by adjusting the midpoints based on the track's geometry.
    - refine_track_for_distance(): Refines the track layout by ensuring the points are within a certain distance threshold.
    - plot_track(): Outputs the optimized track points for visualization or further use.

    */
private:
    std::vector<std::pair<double, double>> left_cones;
    std::vector<std::pair<double, double>> right_cones;
    std::vector<std::pair<double, double>> midpoints;
    std::vector<std::pair<double, double>> T_n_points;
    std::vector<std::pair<double, double>> optimized_points;
    std::vector<std::pair<double, double>> refined_points;
    
public:
    TrackPlotter() {
        auto start = std::chrono::high_resolution_clock::now();
        generate_parallel_curves();
        calculate_midpoints();
        calculate_optimized_points();
        refine_track_for_distance();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        plot_track();
    }
    
    void generate_parallel_curves() {
    std::vector<std::pair<double, double>> points = {
            {5, 5}, {4, 9}, {7, 7}, {5, 11}, {10, 9}, {8, 13}, {13, 10}, {12, 14},
            {16, 10}, {17, 14}, {19, 9}, {21, 12}, {20, 6}, {24, 9}, {20, 3}, {25, 5},
            {22, 0}, {26, 2}, {25, -2}, {29, 2}, {29, -2}, {32, 2}, {34, -2}, {35, 3},
            {38, 0}, {37, 5}, {41, 3}, {38, 8}, {42, 8}, {37, 12}, {42, 13}, {34, 15},
            {39, 17}, {31, 18}, {35, 21}, {27, 19}, {31, 23}, {24, 19}, {26.5, 24},
            {22, 18}, {22, 23}, {20, 18}, {20, 23}, {18, 19}, {18, 24}, {15, 21}, {16, 27},
            {13, 24}, {12, 30}, {10, 25}, {7, 30}, {8, 25}, {2, 27}, {6, 24}, {0, 23},
            {5, 21}, {0, 20}, {4, 17}, {-2, 17}, {2, 14}, {-6, 14}, {-1, 12}, {-6, 9},
            {-1, 10}, {-4, 6}, {0, 9}, {1, 4}, {2, 8}, {5, 5}, {4, 9}, {7, 7}, {5, 11}, {10, 9}, {8, 13}, {13, 10}, {12, 14},
            {16, 10}, {17, 14}, {19, 9}
                };

    // Assuming 'points' contains alternating left and right cone points
    for (int i = 0; i < points.size(); i += 2) {
        if (i + 1 < points.size()) {
            left_cones.push_back(points[i]);      // Even index: left cone
            right_cones.push_back(points[i + 1]); // Odd index: right cone
        } else {
            // If there's an odd number of points, add the last one to the left cones
            left_cones.push_back(points[i]);
        }
    }
}
    
    void calculate_midpoints() {
        for (int i = 0; i < left_cones.size(); i++) {
            double x = (left_cones[i].first + right_cones[i].first) / 2;
            double y = (left_cones[i].second + right_cones[i].second) / 2;
            midpoints.push_back(std::make_pair(x, y));
        }
    }
    
    double calculate_angle(std::pair<double, double> p1, std::pair<double, double> p2) {
        return std::atan2(p2.second - p1.second, p2.first - p1.first);
    }
    
    std::pair<std::pair<double, double>, bool> calculate_intersection(std::pair<double, double> line1_p1, std::pair<double, double> line1_p2, std::pair<double, double> line2_p1, std::pair<double, double> line2_p2) {
    double x1 = line1_p1.first;
    double y1 = line1_p1.second;
    double x2 = line1_p2.first;
    double y2 = line1_p2.second;
    double x3 = line2_p1.first;
    double y3 = line2_p1.second;
    double x4 = line2_p2.first;
    double y4 = line2_p2.second;

    double denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (denominator == 0) {
        // No intersection, return invalid flag
        return std::make_pair(std::make_pair(0, 0), false);
    }

    double px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator;
    double py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator;

    // Intersection found, return point and valid flag
    return std::make_pair(std::make_pair(px, py), true);

    }

    
    bool is_point_on_segment(std::pair<double, double> point, std::pair<double, double> segment_p1, std::pair<double, double> segment_p2) {
        double line_len = std::sqrt(std::pow(segment_p1.first - segment_p2.first, 2) + std::pow(segment_p1.second - segment_p2.second, 2));
        double len1 = std::sqrt(std::pow(point.first - segment_p1.first, 2) + std::pow(point.second - segment_p1.second, 2));
        double len2 = std::sqrt(std::pow(point.first - segment_p2.first, 2) + std::pow(point.second - segment_p2.second, 2));
        
        return std::abs(line_len - (len1 + len2)) < 1e-6;
    }
    
    std::pair<double, double> calculate_optimized_point(std::pair<double, double> p1, std::pair<double, double> p2) {
    if (std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2)) < 0.1) {
        return std::make_pair((p1.first + p2.first) / 2, (p1.second + p2.second) / 2);
    }
    else {
        for (int i = 0; i < left_cones.size() - 1; i++) {
            std::pair<double, double> left_segment_p1 = left_cones[i];
            std::pair<double, double> left_segment_p2 = left_cones[i + 1];
            std::pair<double, double> right_segment_p1 = right_cones[i];
            std::pair<double, double> right_segment_p2 = right_cones[i + 1];

            auto [left_intersection, left_valid] = calculate_intersection(p1, p2, left_segment_p1, left_segment_p2);
            auto [right_intersection, right_valid] = calculate_intersection(p1, p2, right_segment_p1, right_segment_p2);

            if (left_valid && is_point_on_segment(left_intersection, left_segment_p1, left_segment_p2)) {
                return left_intersection;
            }
            else if (right_valid && is_point_on_segment(right_intersection, right_segment_p1, right_segment_p2)) {
                return right_intersection;
            }
        }
    }

    return std::make_pair((p1.first + p2.first) / 2, (p1.second + p2.second) / 2);
}

    void calculate_optimized_points() {
    double p1 = 1.0 / 9;
    double p2 = 2.0 / 3;
    double p3 = 2.0 / 9;

    for (int i = 0; i < midpoints.size() - 2; i++) {
        double angle_n = calculate_angle(midpoints[i], midpoints[i + 1]);
        double angle_n_plus_1 = calculate_angle(midpoints[i + 1], midpoints[i + 2]);
        double optimized_angle = p1 * angle_n + p2 * angle_n_plus_1 + p3 * angle_n;
        double dx = std::cos(optimized_angle);
        double dy = std::sin(optimized_angle);
        std::pair<double, double> line1_p2 = std::make_pair(midpoints[i].first + dx, midpoints[i].second + dy);

        double normal_angle = calculate_angle(midpoints[i + 1], midpoints[i + 2]) + M_PI / 2;
        double normal_dx = std::cos(normal_angle);
        double normal_dy = std::sin(normal_angle);
        std::pair<double, double> line2_p2 = std::make_pair(midpoints[i + 1].first + normal_dx, midpoints[i + 1].second + normal_dy);

        auto [T_n_plus_1, T_n_valid] = calculate_intersection(midpoints[i], line1_p2, midpoints[i + 1], line2_p2);

        if (T_n_valid && is_point_within_track(T_n_plus_1)) {
            T_n_points.push_back(T_n_plus_1);
            double parameter = parameter_for_optimized_point(midpoints[i + 1], T_n_plus_1);
            std::pair<double, double> optimized_point = std::make_pair(
                midpoints[i + 1].first + parameter * (T_n_plus_1.first - midpoints[i + 1].first),
                midpoints[i + 1].second + parameter * (T_n_plus_1.second - midpoints[i + 1].second)
            );
            optimized_points.push_back(optimized_point);
        }
        else {
            // Append the midpoint between midpoints[i] and midpoints[i + 2]
            double mid_x = (midpoints[i].first + midpoints[i + 2].first) / 2;
            double mid_y = (midpoints[i].second + midpoints[i + 2].second) / 2;
            optimized_points.push_back(std::make_pair(mid_x, mid_y));
        }
        
    }
}

    
    bool is_point_within_track(std::pair<double, double> point) {
        double max_distance_from_midpoint = 3;
        for (auto midpoint : midpoints) {
            if (std::sqrt(std::pow(midpoint.first - point.first, 2) + std::pow(midpoint.second - point.second, 2)) <= max_distance_from_midpoint) {
                return true;
            }
        }
        return false;
    }
    
    double parameter_for_optimized_point(std::pair<double, double> midpoint, std::pair<double, double> T_n_point) {
        double nearest_distance_to_border = std::min(
            distance_to_nearest_segment(T_n_point, left_cones),
            distance_to_nearest_segment(T_n_point, right_cones)
        );
        
        double distance_to_midpoint = std::sqrt(std::pow(midpoint.first - T_n_point.first, 2) + std::pow(midpoint.second - T_n_point.second, 2)) + 0.75;
        
        if (distance_to_midpoint == 0) {
            return 0;
        }
        
        double ratio = nearest_distance_to_border / distance_to_midpoint;
        
        return std::min(std::max(ratio, 0.0), 1.0);
    }
    
    double distance_to_nearest_segment(std::pair<double, double> point, std::vector<std::pair<double, double>> segments) {
        double min_distance = std::numeric_limits<double>::infinity();
        for (int i = 0; i < segments.size() - 1; i++) {
            double segment_distance = distance_to_segment(point, segments[i], segments[i + 1]);
            min_distance = std::min(min_distance, segment_distance);
        }
        return min_distance;
    }
    
    double distance_to_segment(std::pair<double, double> point, std::pair<double, double> segment_start, std::pair<double, double> segment_end) {
        std::pair<double, double> start_to_point = std::make_pair(point.first - segment_start.first, point.second - segment_start.second);
        std::pair<double, double> start_to_end = std::make_pair(segment_end.first - segment_start.first, segment_end.second - segment_start.second);
        double projection = (start_to_point.first * start_to_end.first + start_to_point.second * start_to_end.second) / (std::pow(start_to_end.first, 2) + std::pow(start_to_end.second, 2));
        projection = std::max(0.0, std::min(1.0, projection));
        std::pair<double, double> closest_point = std::make_pair(segment_start.first + projection * start_to_end.first, segment_start.second + projection * start_to_end.second);
        return std::sqrt(std::pow(closest_point.first - point.first, 2) + std::pow(closest_point.second - point.second, 2));
    }
    
    void refine_track_for_distance() {
        double max_distance = 1.5;
        for (int i = 0; i < optimized_points.size() - 1; i++) {
            refined_points.push_back(optimized_points[i]);
            std::pair<double, double> next_point = optimized_points[i + 1];
            while (std::sqrt(std::pow(refined_points.back().first - next_point.first, 2) + std::pow(refined_points.back().second - next_point.second, 2)) > max_distance) {
                std::pair<double, double> direction = std::make_pair(next_point.first - refined_points.back().first, next_point.second - refined_points.back().second);
                std::pair<double, double> new_point = std::make_pair(refined_points.back().first + direction.first / std::sqrt(std::pow(direction.first, 2) + std::pow(direction.second, 2)) * max_distance, refined_points.back().second + direction.second / std::sqrt(std::pow(direction.first, 2) + std::pow(direction.second, 2)) * max_distance);
                refined_points.push_back(new_point);
            }
        }
        refined_points.push_back(optimized_points.back());
    }
    
    void plot_track() {
        // Print optimzied points
        for (const auto& point : optimized_points) {
            std::cout << point.first << ", " << point.second << std::endl;
        }

        // Print refined points
        for (const auto& point : refined_points) {
            std::cout << point.first << ", " << point.second << std::endl;
        }

    }

};

int main() {
    TrackPlotter plotter;
    return 0;
}


