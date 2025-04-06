import random

import cv2
import numpy as np
from albumentations import DualTransform

from skimage.color import rgb2hsv, rgb2gray, rgb2yuv

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]

    if min(w, h) == size:
        return img

    if w < h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size

    interpolation = interpolation_up if scale > 1 else interpolation_down

    img = img.astype('uint8')

    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, min_side, always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.min_side = min_side
        self.interpolations = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]


    def apply(self, img, **params):
        return isotropically_resize_image(img, size=self.min_side, interpolation_down=random.choice(self.interpolations),
                                          interpolation_up=random.choice(self.interpolations))

    def apply_to_mask(self, img, **params):
        return self.apply(img, **params)

    def get_transform_init_args_names(self):
        return ("min_side", )


class FFT(DualTransform):
    def __init__(self, mode, p=0.5) -> None:
        super(FFT, self).__init__(p=p)
        self.prob = p
        self.mode = mode

    def apply(self, img, copy=True, **params):
        dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(img)))
        mask = np.log(abs(dark_image_grey_fourier)).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        if self.mode == 0:
            return np.asarray(cv2.bitwise_and(img, img, mask=mask))
        else:
            mask = np.asarray(mask)
            image =  cv2.merge((mask, mask, mask))
            return image


class DCT(DualTransform):
    def __init__(self, mode, p=0.5) -> None:
        super(DCT, self).__init__(p=p)
        self.prob = p
        self.mode = mode

    def rgb2gray(self, rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    def apply(self, img, copy=True, **params):
        gray_img = self.rgb2gray(img)
        dct_coefficients = cv2.dct(cv2.dct(np.float32(gray_img), flags=cv2.DCT_ROWS), flags=cv2.DCT_ROWS)
        epsilon = 1
        mask = np.log(np.abs(dct_coefficients) + epsilon).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))


        if self.mode == 0:
            return cv2.bitwise_and(img, img, mask=mask)
        else:
            dct_coefficients = np.asarray(dct_coefficients)
            image = cv2.merge((dct_coefficients, dct_coefficients, dct_coefficients))
            return image


class FrequencyPatterns(DualTransform):
    def __init__(self, p=0.5, required_pattern=0) -> None:
        super(FrequencyPatterns, self).__init__(p=p)
        self.prob = p
        self.geometric_patterns = [self.pattern_grid, self.pattern_symmetric_grid, self.pattern_checkerdboard,
                                   self.pattern_circular_checkerboard, self.pattern_circles, self.pattern_ovals,
                                   self.pattern_squares, self.pattern_random_lines, self.pattern_stripes]
        self.peaks_patterns = [self.pattern_random_auras, self.pattern_ring_aura, self.pattern_centered_aura,
                               self.pattern_variable_intensity, self.pattern_circular_peaks, self.pattern_random_points]
        self.fourier_patterns = [self.pattern_symmetric_blobs_fourier, self.pattern_symmetric_points_fourier]
        self.patterns = []

        if required_pattern == 0:  # All Patterns
            self.patterns.extend(self.geometric_patterns)
            self.patterns.extend(self.peaks_patterns)
            self.patterns.extend(self.fourier_patterns)
        if required_pattern == 1:  # Geometric Patterns
            self.patterns = self.geometric_patterns
        if required_pattern == 2:  # Peaks Patterns
            self.patterns = self.peaks_patterns
        if required_pattern == 3:
            self.patterns = self.fourier_patterns
        if required_pattern == 4:  # Glide pattern
            self.patterns = [self.pattern_glide]
            self.files = []

    '''
    def generate_pattern(self, pattern_function, cols, rows):
        pattern = pattern_function(cols//2, rows//2)

        pattern = np.vstack([np.hstack([pattern, np.fliplr(pattern)]),
                             np.hstack([np.flipud(pattern), np.flipud(np.fliplr(pattern))])])

        return pattern, True
    '''

    def apply(self, img, required_pattern_fn=None, return_pattern=False, copy=True, weight=0.8, mode=0, **params):
        result_channels = []

        if required_pattern_fn is None:
            pattern_function = random.choice(self.patterns)
        else:
            pattern_function = required_pattern_fn

        res = pattern_function(cols=img.shape[1], rows=img.shape[0])
        pattern = res[0]
        make_pattern_fft = res[1]

        if return_pattern:
            return pattern, make_pattern_fft

        if make_pattern_fft:
            f_pattern = np.fft.fft2(pattern, s=(img.shape[0], img.shape[1]))
        else:  # The pattern comes from a file and it is already the FFT
            f_pattern = pattern

        f_transform_channels = []

        for channel in range(img.shape[2]):
            if mode == 0:
                # Do the Fourier Transform of the channel of the image
                f_transform_channel = np.fft.fft2(img[..., channel])
                f_transform_channels.append(f_transform_channel)

                # Move to magnitude/phase representation
                magnitude_original = np.abs(f_transform_channel)
                phase_original = np.angle(f_transform_channel)

                magnitude_pattern = np.abs(f_pattern)
                # Make the weighted sum of the two magnitudes
                # magnitude_result = (1 - weight) * magnitude_original + weight * magnitude_pattern
                magnitude_result = magnitude_original + magnitude_pattern

                # Combine again the obtained magnitude with the phase of the original image
                f_result_channel = magnitude_result * np.exp(1j * phase_original)

                # Make the inverse fourier transform
                result_channel = np.fft.ifft2(f_result_channel).real

                # Append the resulting channel
                result_channels.append(result_channel)

            elif mode == 1:
                # Move to magnitude/phase representation
                magnitude_pattern = np.abs(f_pattern)

                # Make the inverse fourier transform of the pattern
                pattern = np.fft.ifft2(magnitude_pattern).real

                # Combine the resulting pattern with the image in the spatial dimension
                result_channel = (1 - weight) * img[..., channel] + weight * pattern

                # Append the resulting channel
                result_channels.append(result_channel)
            elif mode == 2:
                # Do the Fourier Transform of the channel of the image
                f_transform_channel = np.fft.fft2(img[..., channel])
                f_transform_channels.append(f_transform_channel)

                # Move to magnitude/phase representation
                magnitude_original = np.abs(f_transform_channel)
                phase_original = np.angle(f_transform_channel)

                magnitude_pattern = np.abs(f_pattern)
                phase_pattern = np.angle(f_pattern)

                # Make the weighted sum of the two magnitudes
                magnitude_result = magnitude_original + magnitude_pattern

                # Make the weighted sum of the two phases
                phase_result = (1 - weight) * phase_original + weight * phase_pattern

                # Combine again the obtained magnitude with the phase of the original image
                f_result_channel = magnitude_result * np.exp(1j * phase_result)

                # Make the inverse fourier transform
                result_channel = np.fft.ifft2(f_result_channel).real

                # Append the resulting channel
                result_channels.append(result_channel)
            elif mode == 3:
                # Do the Fourier Transform of the channel of the image
                f_transform_channel = np.fft.fft2(img[..., channel])
                f_transform_channels.append(f_transform_channel)

                # Move to magnitude/phase representation
                magnitude_original = np.abs(f_transform_channel)
                phase_original = np.angle(f_transform_channel)

                magnitude_pattern = np.abs(f_pattern)
                # Make the weighted sum of the two magnitudes
                # magnitude_result = (1 - weight) * magnitude_original + weight * magnitude_pattern
                magnitude_result = np.maximum(magnitude_original, magnitude_pattern)

                # Combine again the obtained magnitude with the phase of the original image
                f_result_channel = magnitude_result * np.exp(1j * phase_original)

                # Make the inverse fourier transform
                result_channel = np.fft.ifft2(f_result_channel).real

                # Append the resulting channel
                result_channels.append(result_channel)

        # Stack together the channels
        result = np.stack(result_channels, axis=-1)
        '''
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(f_pattern.clip(0, 1),
                   clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
        plt.xticks([])
        plt.yticks([])

        fig.savefig("pattern_fft.png",
                    bbox_inches='tight', pad_inches=0.0)

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.mean(f_transform_channels, -1).clip(0, 1),
                   clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
        plt.xticks([])
        plt.yticks([])

        fig.savefig("image_fft.png",
                    bbox_inches='tight', pad_inches=0.0)

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("pattern_applied_image.png", result)
        '''

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result

    def pattern_glide(self, cols, rows, directory_path='outputs/fouriers/output_glide'):
        pattern = self.pattern_from_file(cols, rows, directory_path)

        resized_pattern = cv2.resize(pattern, (cols, rows))

        return resized_pattern, False

    def pattern_from_file(self, cols, rows, directory_path):
        if len(self.files) == 0:
            self.files = [np.load(os.path.join(directory_path, f)) for f in os.listdir(directory_path) if
                          f.endswith('.npy') and f.startswith("fft_sample")]

        if not self.files:
            raise ValueError("No pattern file found.")
        pattern_index = np.random.choice(len(self.files))
        pattern = self.files[pattern_index]
        '''
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.mean(pattern, -1).clip(0, 1),
                   clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
        plt.xticks([])
        plt.yticks([])

        fig.savefig("pattern_fft_readed.png",
                    bbox_inches='tight', pad_inches=0.0)
        '''
        return np.mean(pattern, -1)

    def pattern_circular_checkerboard(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_sectors = random.randint(4, 16)

        center_x, center_y = cols // 2, rows // 2
        max_radius = min(center_x, center_y)
        if max_radius == 0:
            max_radius == 1

        sector_angle = 2 * np.pi / num_sectors

        for y in range(rows):
            for x in range(cols):
                dx = x - center_x
                dy = y - center_y
                angle = np.arctan2(dy, dx)

                if angle < 0:
                    angle += 2 * np.pi

                sector_index = int(angle / sector_angle)
                if sector_index % 2 == 0:
                    pattern[y, x] = 255

        return pattern, True

    def pattern_grid(self, cols, rows):
        pattern = np.zeros((rows, cols))

        cell_size = random.randint(10, 30)
        intensity_range = (50, 200)

        for y in range(0, rows, cell_size):
            for x in range(0, cols, cell_size):
                cell_intensity = random.randint(*intensity_range)
                pattern[y:y + cell_size, x:x + cell_size] = cell_intensity

        return pattern, True

    def pattern_checkerdboard(self, cols, rows):
        pattern = np.zeros((rows, cols))

        cell_size = random.randint(10, 30)
        num_cells = random.randint(5, 20)
        intensity = random.randint(150, 255)

        center_x, center_y = cols // 2, rows // 2
        max_offset = min(center_x, center_y) - cell_size
        for y in range(center_y - max_offset, center_y + max_offset, cell_size * 2):
            for x in range(center_x - max_offset, center_x + max_offset, cell_size * 2):
                pattern[y:y + cell_size, x:x + cell_size] = intensity

        return pattern, True

    def pattern_squares(self, cols, rows):
        pattern = np.zeros((rows, cols))

        num_cells = random.randint(5, 20)
        cell_size = random.randint(10, 30)
        intensity = random.randint(150, 255)

        center_or_edge = random.choice(['center', 'edge'])

        for _ in range(num_cells):
            if center_or_edge == 'center':
                cell_x, cell_y = self.random_position_in_center(cols, rows, cell_size)
            else:
                cell_x, cell_y = self.random_position_in_edge(cols, rows, cell_size)

            pattern[cell_y:cell_y + cell_size, cell_x:cell_x + cell_size] = intensity

        return pattern, True

    def random_position_in_center(self, cols, rows, cell_size):
        center_x, center_y = cols // 2, rows // 2
        max_offset = min(center_x, center_y) - cell_size

        cell_x = random.randint(center_x - max_offset, center_x + max_offset - cell_size)
        cell_y = random.randint(center_y - max_offset, center_y + max_offset - cell_size)

        return cell_x, cell_y

    def random_position_in_edge(self, cols, rows, cell_size):
        edge_x = random.choice([0, cols - 1])
        edge_y = random.choice([0, rows - 1])

        cell_x = random.randint(edge_x, cols - cell_size) if edge_x == 0 else random.randint(0, cols - cell_size)
        cell_y = random.randint(edge_y, rows - cell_size) if edge_y == 0 else random.randint(0, rows - cell_size)

        return cell_x, cell_y

    def pattern_random_lines(self, cols, rows):
        pattern = np.zeros((rows, cols))

        num_lines = random.randint(5, 20)

        for _ in range(num_lines):
            line_intensity = random.randint(50, 200)
            line_length = random.randint(10, min(cols, rows) // 2)
            line_angle = random.uniform(0, 2 * np.pi)
            start_x = cols // 2
            start_y = rows // 2

            for i in range(line_length):
                x = int(start_x + i * np.cos(line_angle))
                y = int(start_y + i * np.sin(line_angle))

                if 0 <= x < cols and 0 <= y < rows:
                    pattern[y, x] = line_intensity

        return pattern, True

    def pattern_random_points(self, cols, rows):
        pattern = np.zeros((rows, cols))

        num_points = random.randint(2, 30)

        for _ in range(num_points):
            point_intensity = random.randint(10, 255)
            point_x = random.randint(0, cols - 1)
            point_y = random.randint(0, rows - 1)
            pattern[point_y, point_x] = point_intensity

        return pattern, True

    def pattern_circular_peaks(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_peaks = random.randint(2, 30)

        for _ in range(num_peaks):
            peak_intensity = random.randint(100, 255)
            peak_x = random.randint(0, cols - 1)
            peak_y = random.randint(0, rows - 1)

            inner_radius = random.randint(5, 10)
            outer_radius = random.randint(15, 30)
            angle = random.uniform(0, 2 * np.pi)

            for i in range(-outer_radius, outer_radius + 1):
                for j in range(-outer_radius, outer_radius + 1):
                    distance = np.sqrt(i ** 2 + j ** 2)
                    if inner_radius <= distance <= outer_radius:
                        x = peak_x + i
                        y = peak_y + j

                        x = np.clip(x, 0, cols - 1)
                        y = np.clip(y, 0, rows - 1)

                        pattern[y, x] = peak_intensity * np.exp(
                            -0.5 * ((distance - inner_radius) / (outer_radius - inner_radius)) ** 2)
        return pattern, True

    def pattern_symmetric_points_fourier(self, cols, rows):
        pattern = np.zeros((rows, cols))

        num_peaks = random.randint(5, 20)

        for _ in range(num_peaks):
            peak_intensity = random.uniform(100, 200)
            peak_x = random.randint(0, cols // 2 - 1)
            peak_y = random.randint(0, rows // 2 - 1)

            pattern[peak_y, peak_x] = peak_intensity
            pattern[rows - 1 - peak_y, cols - 1 - peak_x] = peak_intensity
            pattern[peak_y, cols - 1 - peak_x] = peak_intensity
            pattern[rows - 1 - peak_y, peak_x] = peak_intensity

        return np.fft.ifft2(pattern ** 2).real, True

    def pattern_symmetric_blobs_fourier(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_blobs = random.randint(2, 12)

        for _ in range(num_blobs):
            blob_intensity = random.uniform(100, 255)
            blob_size = random.randint(5, 30)
            blob_x = random.randint(0, cols // 2 - blob_size)
            blob_y = random.randint(0, rows // 2 - blob_size)

            pattern[blob_y:blob_y + blob_size, blob_x:blob_x + blob_size] = self.create_gaussian_blob(blob_size,
                                                                                                      blob_intensity)
            pattern[rows - 1 - blob_y - blob_size:rows - 1 - blob_y,
            cols - 1 - blob_x - blob_size:cols - 1 - blob_x] = self.create_gaussian_blob(blob_size, blob_intensity)
            pattern[blob_y:blob_y + blob_size,
            cols - 1 - blob_x - blob_size:cols - 1 - blob_x] = self.create_gaussian_blob(blob_size, blob_intensity)
            pattern[rows - 1 - blob_y - blob_size:rows - 1 - blob_y,
            blob_x:blob_x + blob_size] = self.create_gaussian_blob(blob_size, blob_intensity)

        return np.fft.ifft2(pattern ** 2).real, True

    def create_gaussian_blob(self, size, intensity):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x_center, y_center = size // 2, size // 2
        distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        normalized_distance = distance / (size / 2)
        blob = intensity * np.exp(-5 * normalized_distance ** 2)
        return blob

    def pattern_centered_aura(self, cols, rows):
        pattern = np.zeros((rows, cols))

        center_x, center_y = cols // 2, rows // 2
        max_radius = min(center_x, center_y)
        if max_radius == 0:
            max_radius = 1

        for y in range(rows):
            for x in range(cols):
                distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                normalized_distance = distance_to_center / max_radius

                pattern[y, x] = np.exp(-5 * normalized_distance ** 2)

        return pattern, True

    def pattern_ring_aura(self, cols, rows):
        pattern = np.zeros((rows, cols))

        center_x, center_y = cols // 2, rows // 2
        inner_radius = random.randint(20, min(center_x, center_y) - 20)
        outer_radius = random.randint(inner_radius + 10, min(center_x, center_y) - 10)

        if inner_radius == outer_radius:
            return pattern, False

        for y in range(rows):
            for x in range(cols):
                distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                if inner_radius <= distance_to_center <= outer_radius:
                    normalized_distance = (distance_to_center - inner_radius) / (outer_radius - inner_radius)
                    pattern[y, x] = np.exp(-5 * normalized_distance ** 2)

        return pattern, True

    def pattern_random_auras(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_auras = random.randint(2, 12)

        for _ in range(num_auras):
            center_x, center_y = np.random.randint(cols), np.random.randint(rows)
            max_radius = min(center_x, center_y) // 12

            if max_radius == 0:
                max_radius = 1

            intensity = np.random.uniform(0.5, 1)

            for y in range(rows):
                for x in range(cols):
                    distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    normalized_distance = distance_to_center / max_radius
                    pattern[y, x] += intensity * np.exp(-5 * normalized_distance ** 2)

        pattern = pattern / np.max(pattern)

        return pattern, True

    def pattern_symmetric_grid(self, cols, rows):
        pattern = np.zeros((rows, cols))

        value_x = np.random.uniform(1, 50)
        value_y = value_x

        for y in range(rows):
            for x in range(cols):
                value = 128 + 127 * np.sin(2 * np.pi * value_x * x / cols) * np.sin(2 * np.pi * value_y * y / rows)
                pattern[y, x] = value

        return pattern, True

    def pattern_ovals(self, cols, rows):
        pattern = np.zeros((rows, cols))

        num_ovals = random.randint(2, 20)

        for _ in range(num_ovals):
            oval_intensity = random.randint(50, 200)
            oval_major_axis = random.randint(10, min(cols, rows) // 4)
            oval_minor_axis = random.randint(5, oval_major_axis - 1)
            center_x = random.randint(oval_major_axis, cols - oval_major_axis)
            center_y = random.randint(oval_major_axis, rows - oval_major_axis)

            y, x = np.ogrid[:rows, :cols]
            mask = ((x - center_x) / oval_major_axis) ** 2 + ((y - center_y) / oval_minor_axis) ** 2 <= 1
            pattern[mask] = oval_intensity

        return pattern, True

    def pattern_circles(self, cols, rows):
        pattern = np.zeros((rows, cols))

        num_circles = random.randint(2, 20)

        for _ in range(num_circles):
            circle_intensity = random.randint(50, 200)
            circle_radius = random.randint(10, min(cols, rows) // 4)
            center_x = random.randint(circle_radius, cols - circle_radius)
            center_y = random.randint(circle_radius, rows - circle_radius)

            y, x = np.ogrid[:rows, :cols]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= circle_radius ** 2
            pattern[mask] = circle_intensity

        return pattern, True

    def pattern_variable_intensity(self, cols, rows, towards_center=True):
        pattern = np.zeros((rows, cols))

        max_intensity = random.randint(100, 200)
        aura_width = random.uniform(0.1, 0.5)

        for y in range(rows):
            for x in range(cols):
                distance_to_center = np.sqrt((x - cols // 2) ** 2 + (y - rows // 2) ** 2)
                normalized_distance = distance_to_center / max(cols // 2, rows // 2)

                intensity = max_intensity * np.exp(-0.5 * ((normalized_distance / aura_width) ** 2))

                if towards_center:
                    intensity = max_intensity - intensity

                pattern[y, x] = intensity

        return pattern, True

    def pattern_stripes(self, cols, rows):
        pattern = np.zeros((rows, cols))

        stripe_direction = random.choice(['horizontal', 'vertical', 'both'])

        if stripe_direction == 'both':
            vertical_stripes = self.pattern_stripes_single_direction(cols, rows, 'vertical')
            horizontal_stripes = self.pattern_stripes_single_direction(cols, rows, 'horizontal')
            pattern = np.maximum(vertical_stripes, horizontal_stripes)
        else:
            pattern = self.pattern_stripes_single_direction(cols, rows, stripe_direction)

        return pattern, True

    def pattern_stripes_single_direction(self, cols, rows, direction):
        pattern = np.zeros((rows, cols))

        num_stripes = random.randint(5, 20)
        stripe_width = random.randint(10, 30)
        stripe_distance = random.randint(30, 60)
        intensity = random.randint(100, 255)
        if direction == 'horizontal':
            for i in range(num_stripes):
                start_y = random.randint(0, rows - 1)
                pattern[start_y:start_y + stripe_width, :] = intensity
                start_y += stripe_distance
        elif direction == 'vertical':
            for i in range(num_stripes):
                start_x = random.randint(0, cols - 1)
                pattern[:, start_x:start_x + stripe_width] = intensity
                start_x += stripe_distance

        return pattern



