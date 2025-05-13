import cv2
import numpy as np


class PatternMaker:
    def __init__(self, cols, width, height, conner_size=(5, 5)):
        self.cols = cols
        self.rows = -1
        self.width = width
        self.height = height
        self.conner_size = conner_size
        self.g = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.g.fill(255)

    def make_circle_pattern(self, radius=20):
        xspacing = self.conner_size[0] // 2 + radius
        yspacing = self.conner_size[1] // 2 + radius

        cv2.circle(self.g, (xspacing, yspacing),
                   radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.g, (self.width - xspacing, yspacing),
                   radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.g, (xspacing, self.height - yspacing),
                   radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.g, (self.width - xspacing, self.height -
                   yspacing), radius, (0, 0, 0), -1, cv2.LINE_AA)

    def make_checkerboard_pattern(self, square_size=None, margin_percent=15):
        """
        Create a checkerboard pattern optimized for detection by OpenCV.
        
        Args:
            square_size: Optional size of each square (in pixels). If None, calculated based on cols.
            margin_percent: Percentage of the image to use as margin (default 15%)
        """
        # Calculate margins
        margin_x = int(self.width * margin_percent / 100)
        margin_y = int(self.height * margin_percent / 100)
        
        # Calculate available space for the checkerboard
        available_width = self.width - 2 * margin_x
        available_height = self.height - 2 * margin_y
        
        # Calculate square size if not provided
        if square_size is None:
            # Make sure we have reasonable sized squares that fit within the available space
            square_size = min(available_width // self.cols, available_height // self.cols)
        
        # Calculate rows based on available height and square size
        self.rows = min(self.cols, available_height // square_size)
        
        # Calculate total board size
        board_width = self.cols * square_size
        board_height = self.rows * square_size
        
        # Calculate centering offsets
        offset_x = margin_x + (available_width - board_width) // 2
        offset_y = margin_y + (available_height - board_height) // 2
        
        # Clear the canvas
        self.g.fill(0)
        
        # Draw the checkerboard
        for i in range(self.rows):
            for j in range(self.cols):
                if (i + j) % 2 == 0:
                    continue  # Skip white squares
                
                x1 = offset_x + j * square_size
                y1 = offset_y + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                cv2.rectangle(self.g, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # Draw a border around the checkerboard for better visibility
        cv2.rectangle(self.g, 
                     (offset_x - 2, offset_y - 2), 
                     (offset_x + board_width + 2, offset_y + board_height + 2), 
                     (128, 128, 128), 2)

    def get(self):
        return self.g

    def get_size_chess(self):
        return [self.cols - 1, self.rows - 1]

    def generate_aruco_board(self, dictionary_id=cv2.aruco.DICT_4X4_50, marker_size=100, margin_percent=10):
        """
        Generate an ArUco marker board as an alternative to checkerboard for easier detection.
        
        Args:
            dictionary_id: ArUco dictionary ID
            marker_size: Size of each marker in pixels
            margin_percent: Percentage of margin around the board
        """
        # Clear the canvas
        self.g.fill(255)
        
        # Calculate margins
        margin_x = int(self.width * margin_percent / 100)
        margin_y = int(self.height * margin_percent / 100)
        
        # Calculate available space
        available_width = self.width - 2 * margin_x
        available_height = self.height - 2 * margin_y
        
        # Determine grid size for markers
        grid_cols = min(5, self.cols)  # Limit to 5 columns for better detection
        grid_rows = min(5, grid_cols)  # Square grid works better
        
        # Calculate marker size and spacing
        marker_size = min(available_width // grid_cols, available_height // grid_rows)
        spacing = marker_size // 10  # Small spacing between markers
        
        # Calculate total board dimensions
        board_width = grid_cols * (marker_size + spacing) - spacing
        board_height = grid_rows * (marker_size + spacing) - spacing
        
        # Calculate centering offsets
        offset_x = margin_x + (available_width - board_width) // 2
        offset_y = margin_y + (available_height - board_height) // 2
        
        # Create ArUco dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        
        # Draw markers
        marker_id = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                if marker_id >= dictionary.bytesList.shape[0]:
                    break
                    
                x = offset_x + j * (marker_size + spacing)
                y = offset_y + i * (marker_size + spacing)
                
                # Generate marker image
                marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
                marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, marker_img, 1)
                
                # Convert to 3 channels
                marker_img_color = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
                
                # Place on canvas
                self.g[y:y+marker_size, x:x+marker_size] = marker_img_color
                
                marker_id += 1
        
        # Update rows and cols to match the grid
        self.rows = grid_rows
        self.cols = grid_cols
