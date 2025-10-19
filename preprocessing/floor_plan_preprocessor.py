import cv2
import numpy as np


class FloorPlanPreprocessor:
    def __init__(self, image_fpath):
        self.floor_plan_img = self.parse_image(image_fpath)
        self.door_clusters = []

    def parse_image(self, image_fpath):
        return cv2.imread(image_fpath)

    def fill_in_black_color(self, img):
        black_mask = np.all(img == [0, 0, 0], axis=2)
        result = img.copy()
        result[black_mask] = [0, 0, 0]
        return result

    def _is_there_garbage(self):
        target = np.array([64, 64, 64])
        distances = np.linalg.norm(self.floor_plan_img - target, axis=2)
        garbage_pixels = np.sum(distances <= 10)
        total_pixels = self.floor_plan_img.shape[0] * self.floor_plan_img.shape[1]
        return (garbage_pixels / total_pixels) > 0.01

    def cluster_by_colors(self, n_clusters=5):
        pixels = self.floor_plan_img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.01)
        _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return labels.reshape(self.floor_plan_img.shape[:2]), centers

    def get_garbage_cluster(self, clusters):
        labels, centers = clusters
        target = np.array([64, 64, 64])
        distances = np.linalg.norm(centers - target, axis=1)
        garbage_cluster_idx = np.argmin(distances)
        return labels, garbage_cluster_idx

    def _is_red_or_green(self, pixel, threshold=40):
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
        is_red = (r - b > threshold) and (r - g > threshold)
        is_green = (g - r > threshold) and (g - b > threshold)
        return is_red or is_green

    def fill_garbage_cluster(self, labels, garbage_idx):
        mask = (labels == garbage_idx).astype(np.uint8)
        filled = mask.copy()

        while np.any(filled):
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(filled, kernel, iterations=1)
            boundary = filled - eroded

            y_coords, x_coords = np.where(boundary == 1)
            if len(y_coords) == 0:
                break

            for y, x in zip(y_coords, x_coords):
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.floor_plan_img.shape[0] and 0 <= nx < self.floor_plan_img.shape[1]:
                            if filled[ny, nx] == 0 and not self._is_red_or_green(self.floor_plan_img[ny, nx], threshold=70):
                                neighbors.append(self.floor_plan_img[ny, nx])

                if neighbors:
                    self.floor_plan_img[y, x] = np.mean(neighbors, axis=0).astype(np.uint8)
                filled[y, x] = 0

    def select_white_only(self, epsilon=20):
        white = np.array([255, 255, 255])
        distances = np.linalg.norm(self.floor_plan_img - white, axis=2)
        binary = np.where(distances < epsilon, 255, 0).astype(np.uint8)
        return binary

    def fill_in_noise(self, binary_img, kernel_size=5, iterations=1, kernel_shape='rect', closing_iterations=1):
        if kernel_shape == 'rect':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        closing = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
        return closing

    def show_results(self):
        combined = np.hstack([self.orig_image, self.floor_plan_img])
        cv2.imshow('Original | Processed', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_door_contours(self, ang_min=30, ang_max=60, min_area_px=45, max_area_px=75, min_ar=1.2, max_ar=4.0):
        h, w = self.floor_plan_img.shape[:2]
        gray = cv2.cvtColor(self.floor_plan_img, cv2.COLOR_BGR2GRAY)

        k = max(9, int(round(min(h, w) * 0.02)) | 1)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

        blur = cv2.GaussianBlur(tophat, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        keep = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            rect = cv2.minAreaRect(cnt)
            (w_box, h_box) = rect[1]
            if w_box <= 0 or h_box <= 0:
                continue
            area = w_box * h_box
            if area < min_area_px or area > max_area_px:
                continue

            ar = max(w_box, h_box) / min(w_box, h_box)
            if ar < min_ar or ar > max_ar:
                continue

            ang = abs(rect[2])
            if ang > 90:
                ang = 180 - ang
            if not (ang_min <= ang <= ang_max):
                continue

            keep.append(rect)

        return keep

    def group_by_distance(self):
        door_contours = self.find_door_contours()
        num_contours = len(door_contours)

        if num_contours == 0:
            return []

        centers = np.array([rect[0] for rect in door_contours], dtype=np.float32)

        k = max(1, num_contours // 4)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(centers, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        clusters = [[] for _ in range(k)]
        for idx, label in enumerate(labels.flatten()):
            clusters[label].append(door_contours[idx])

        self.door_clusters = clusters
        return clusters

    def calculate_doors(self):
        if not self.door_clusters:
            self.group_by_distance()

        for cluster in self.door_clusters:
            if not cluster:
                continue

            all_points = []
            for rect in cluster:
                box = cv2.boxPoints(rect)
                all_points.extend(box)

            all_points = np.array(all_points)
            x_min, y_min = all_points.min(axis=0).astype(int)
            x_max, y_max = all_points.max(axis=0).astype(int)

            x_diff = x_max - x_min
            y_diff = y_max - y_min

            if x_diff > y_diff:
                x_min = max(0, x_min - 20)
                x_max = min(self.floor_plan_img.shape[1], x_max + 20)
            else:
                y_min = max(0, y_min - 20)
                y_max = min(self.floor_plan_img.shape[0], y_max + 20)

            self.floor_plan_img[y_min:y_max, x_min:x_max] = [255, 255, 255]

    def process_floor_plan(self):
        self.orig_image = self.floor_plan_img.copy()
        self.calculate_doors()
        binary_img = self.select_white_only(epsilon=20)
        filled_noise = self.fill_in_noise(binary_img, kernel_size=3, iterations=1, kernel_shape='rect')
        return filled_noise
