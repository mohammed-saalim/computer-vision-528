"""
2D Transform Editor - Exercise 2.2 (from Szeliski's )

Usage:
1. Run this script to open an empty canvas window.
2. **Shift + Drag** with the left mouse button to create a new rectangle (rubber-band style).
3. Select a transformation mode from the provided radio buttons:
   - Translation – move the rectangle without rotation or scaling.
   - Rigid – rotate and translate (no scaling, shape rigid).
   - Similarity – uniform scale + rotation + translation (preserves aspect ratio).
   - Affine – general affine (independent scaling/shearing, turns rectangle into parallelogram).
   - Perspective – full projective transform (turns rectangle into arbitrary quadrilateral).
4. Click and **drag any corner** of a rectangle to apply the selected transform in real time.
   The rectangle’s shape updates as you drag that corner.
5. Repeat to create multiple rectangles or change modes for different transformations.
6. Use the **Save** button to save all rectangles (their base size and transform matrix) to a JSON file.
7. Use the **Load** button to reload rectangles from a JSON file (restoring their last saved positions and shapes).

This program separates the UI, geometry, and transformation logic for clarity. Each rectangle maintains its own 3x3 transform matrix. Corner dragging computes a new transform that maps the rectangle’s base corners to new positions according to the chosen motion model.
"""
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import json

class Rectangle:
    """Represents a rectangle with a base size and a transformation matrix."""
    def __init__(self, x, y, width, height):
        """
        Initialize a rectangle at canvas position (x, y) with given width and height.
        The base (untransformed) rectangle spans from (0,0) to (width, height).
        The initial transform places the rectangle's top-left at (x, y) (translation).
        """
        self.base_width = width
        self.base_height = height
        # Homogeneous transform matrix (3x3) mapping base to canvas. Start with translation.
        self.transform = np.array([[1, 0, x],
                                   [0, 1, y],
                                   [0, 0, 1]], dtype=float)
        self.canvas_item = None  # Canvas item ID for the rectangle's polygon

    def get_corners(self):
        """Return the four corner coordinates of the rectangle on the canvas."""
        # Base rectangle corners as homogeneous coordinates (column vectors)
        base_corners = np.array([
            [0, 0, 1],                               # top-left corner in base coords
            [self.base_width, 0, 1],                 # top-right
            [self.base_width, self.base_height, 1],  # bottom-right
            [0, self.base_height, 1]                 # bottom-left
        ], dtype=float).T  # shape = (3,4)
        # Apply the 3x3 transform to all base corner points
        transformed = self.transform.dot(base_corners)  # shape = (3,4)
        # Convert from homogeneous coordinates to 2D (divide by last row)
        transformed /= transformed[2]  # broadcast division for each column
        pts_2d = transformed[:2].T  # take x,y rows, transpose to list of points
        return [(float(x), float(y)) for x, y in pts_2d]

class TransformEditorApp:
    """Main application class for the 2D Transform Editor GUI."""
    def __init__(self, master):
        self.root = master
        self.root.title("2D Transform Editor")
        # Selected transformation mode (one of "translation", "rigid", "similarity", "affine", "perspective")
        self.mode = tk.StringVar(value="translation")
        # Canvas for drawing rectangles
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Control panel for mode selection and save/load
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=2)
        modes = ["translation", "rigid", "similarity", "affine", "perspective"]
        for m in modes:
            rb = ttk.Radiobutton(ctrl_frame, text=m.capitalize(), variable=self.mode, value=m)
            rb.pack(side=tk.LEFT, padx=5)
        save_btn = ttk.Button(ctrl_frame, text="Save", command=self.save_to_file)
        load_btn = ttk.Button(ctrl_frame, text="Load", command=self.load_from_file)
        load_btn.pack(side=tk.RIGHT, padx=5)
        save_btn.pack(side=tk.RIGHT, padx=5)
        # Internal state for rectangles and interactions
        self.rectangles = []        # list of Rectangle objects on canvas
        self.current_rect = None    # Rectangle currently being drawn (during Shift+Drag)
        self.dragging_rect = None   # Rectangle currently being transformed (on corner drag)
        self.drag_corner_index = None  # Index (0-3) of which corner is being dragged
        # (For rigid/similarity modes) store initial center and corner positions for calculations
        self._initial_center = None
        self._initial_corner = None
        # Bind mouse events on canvas
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_down(self, event):
        """Mouse button press handler. Starts drawing a new rectangle (Shift+Click) or begins a corner drag."""
        if event.state & 0x0001:  # Shift key is held during mouse down
            # Start a new rectangle at the click point
            self.current_rect = Rectangle(event.x, event.y, 0, 0)
            # Draw a temporary rectangle outline (rubber-band) on the canvas
            self.current_rect.canvas_item = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="black", dash=(4, 2)
            )
        else:
            # Attempt to select a corner of an existing rectangle for transformation
            selected_rect = None
            selected_corner = None
            tol = 5  # selection tolerance in pixels
            for rect in self.rectangles:
                for i, (cx, cy) in enumerate(rect.get_corners()):
                    if abs(event.x - cx) <= tol and abs(event.y - cy) <= tol:
                        selected_rect = rect
                        selected_corner = i
                        break
                if selected_rect:
                    break
            if selected_rect:
                # Begin dragging this rectangle's corner
                self.dragging_rect = selected_rect
                self.drag_corner_index = selected_corner
                # Bring the selected rectangle to front (so it isn't obscured)
                self.canvas.tag_raise(selected_rect.canvas_item)
                # For rigid and similarity, record initial center and corner positions for computing transform
                if self.mode.get() in ("rigid", "similarity"):
                    corners = selected_rect.get_corners()
                    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
                    self._initial_center = (sum(xs) / 4.0, sum(ys) / 4.0)
                    self._initial_corner = corners[self.drag_corner_index]
                else:
                    self._initial_center = None
                    self._initial_corner = None
            else:
                # Click not on a corner or not in Shift mode – do nothing
                self.dragging_rect = None
                self.drag_corner_index = None

    def on_mouse_move(self, event):
        """Mouse drag handler. Updates rubber-band rectangle or applies transform to a rectangle during drag."""
        if self.current_rect:
            # Update the rubber-band (in-progress) rectangle's outline as mouse moves
            x0 = self.current_rect.transform[0, 2]
            y0 = self.current_rect.transform[1, 2]
            # Adjust the temporary rectangle to go from initial point (x0,y0) to current mouse (event.x,event.y)
            self.canvas.coords(self.current_rect.canvas_item, x0, y0, event.x, event.y)
        elif self.dragging_rect:
            # A rectangle's corner is being dragged – compute and apply new transform
            rect = self.dragging_rect
            mode = self.mode.get()
            new_T = self.compute_transform_for_drag(rect, self.drag_corner_index, event.x, event.y, mode)
            if new_T is not None:
                rect.transform = new_T
                # Update the polygon coordinates on the canvas to reflect the new corners
                new_coords = [coord for point in rect.get_corners() for coord in point]
                self.canvas.coords(rect.canvas_item, *new_coords)

    def on_mouse_up(self, event):
        """Mouse button release handler. Finalizes rectangle creation or corner dragging."""
        if self.current_rect:
            # Finish creating a new rectangle on mouse release
            x0 = self.current_rect.transform[0, 2]
            y0 = self.current_rect.transform[1, 2]
            x1, y1 = event.x, event.y
            # Normalize coordinates so that (x_min,y_min) is top-left and (x_max,y_max) is bottom-right
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = min(y0, y1), max(y0, y1)
            width = x_max - x_min
            height = y_max - y_min
            # Update the Rectangle object with final size and position
            self.current_rect.base_width = width
            self.current_rect.base_height = height
            self.current_rect.transform = np.array([[1, 0, x_min],
                                                   [0, 1, y_min],
                                                   [0, 0, 1]], dtype=float)
            # Remove the temporary outline rectangle
            self.canvas.delete(self.current_rect.canvas_item)
            # Draw a permanent polygon for the new rectangle
            coords = [coord for point in self.current_rect.get_corners() for coord in point]
            poly_id = self.canvas.create_polygon(coords, outline="black", fill="", width=2)
            self.current_rect.canvas_item = poly_id
            # Add this rectangle to the list of rectangles
            self.rectangles.append(self.current_rect)
            self.current_rect = None
        elif self.dragging_rect:
            # Finished transforming a rectangle; clear the drag state
            self.dragging_rect = None
            self.drag_corner_index = None
            self._initial_center = None
            self._initial_corner = None

    def compute_transform_for_drag(self, rect, corner_idx, new_x, new_y, mode):
        """
        Compute a new 3x3 transform matrix for `rect` when its corner `corner_idx` is dragged to (new_x, new_y).
        This respects the constraints of the given transformation mode.
        """
        # Current corner coordinates (before moving)
        old_corners = rect.get_corners()
        cx, cy = old_corners[corner_idx]
        # Base corner coordinates (homogeneous) for reference
        w, h = rect.base_width, rect.base_height
        base_pts = {
            0: np.array([0, 0, 1], dtype=float),        # base top-left
            1: np.array([w, 0, 1], dtype=float),        # base top-right
            2: np.array([w, h, 1], dtype=float),        # base bottom-right
            3: np.array([0, h, 1], dtype=float)         # base bottom-left
        }
        # Current target positions for each base corner (before moving any)
        tgt_positions = [np.array([x, y], dtype=float) for (x, y) in old_corners]
        # Prepare the new target for the dragged corner
        tgt_positions[corner_idx] = np.array([new_x, new_y], dtype=float)

        if mode == "translation":
            # Only translate: compute shift and apply to transform
            dx = new_x - cx
            dy = new_y - cy
            new_transform = rect.transform.copy()
            new_transform[0, 2] += dx
            new_transform[1, 2] += dy
            return new_transform

        elif mode == "rigid":
            # Rigid: rotation + translation, no scaling. We rotate around the rectangle's center.
            center = self._initial_center or ((old_corners[0][0] + old_corners[2][0]) / 2.0,
                                              (old_corners[0][1] + old_corners[2][1]) / 2.0)
            # Vectors from center to the dragged corner: before and after
            v_old = np.array([cx - center[0], cy - center[1]], dtype=float)
            v_new = np.array([new_x - center[0], new_y - center[1]], dtype=float)
            # Compute rotation angle to align v_old -> v_new
            angle = 0.0
            if np.linalg.norm(v_old) > 1e-6 and np.linalg.norm(v_new) > 1e-6:
                angle = np.arctan2(v_new[1], v_new[0]) - np.arctan2(v_old[1], v_old[0])
            cosA, sinA = np.cos(angle), np.sin(angle)
            R = np.array([[cosA, -sinA],
                          [sinA,  cosA]], dtype=float)
            # Rotate all corners around center by this angle
            rotated_pts = []
            for (ox, oy) in old_corners:
                vec = np.array([ox - center[0], oy - center[1]], dtype=float)
                rv = R.dot(vec)
                rotated_pts.append(np.array([center[0] + rv[0], center[1] + rv[1]], dtype=float))
            # Now translate so that the dragged corner matches the new position exactly
            dragged_rotated = rotated_pts[corner_idx]
            t_dx = new_x - dragged_rotated[0]
            t_dy = new_y - dragged_rotated[1]
            new_pts = [pt + np.array([t_dx, t_dy]) for pt in rotated_pts]
            # Solve for affine transform (a11,a12,t1, a21,a22,t2) mapping base -> new_pts for base corners (0,1,3)
            base_triplet = [base_pts[0], base_pts[1], base_pts[3]]  # use base TL, TR, BL
            target_triplet = [new_pts[0], new_pts[1], new_pts[3]]
            A = []; b = []
            for base, target in zip(base_triplet, target_triplet):
                x, y, _ = base
                X, Y = target
                # Each point gives two equations for affine: [a11*x + a12*y + t1 = X] and [a21*x + a22*y + t2 = Y]
                A.extend([[x, y, 1, 0, 0, 0],
                          [0, 0, 0, x, y, 1]])
                b.extend([X, Y])
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
            # Solve least-squares (in case points are slightly off affine due to numeric rotation)
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            a11, a12, t1, a21, a22, t2 = sol
            new_transform = np.array([[a11, a12, t1],
                                      [a21, a22, t2],
                                      [0,   0,   1]], dtype=float)
            return new_transform

        elif mode == "similarity":
            # Similarity: uniform scale + rotation + translation around the rectangle's center.
            center = self._initial_center or ((old_corners[0][0] + old_corners[2][0]) / 2.0,
                                              (old_corners[0][1] + old_corners[2][1]) / 2.0)
            v_old = np.array([cx - center[0], cy - center[1]], dtype=float)
            v_new = np.array([new_x - center[0], new_y - center[1]], dtype=float)
            angle = 0.0
            scale = 1.0
            if np.linalg.norm(v_old) > 1e-6 and np.linalg.norm(v_new) > 1e-6:
                angle = np.arctan2(v_new[1], v_new[0]) - np.arctan2(v_old[1], v_old[0])
                scale = np.linalg.norm(v_new) / np.linalg.norm(v_old)
            cosA, sinA = np.cos(angle), np.sin(angle)
            # Rotate and scale all corners around the center
            transformed_pts = []
            for (ox, oy) in old_corners:
                vec = np.array([ox - center[0], oy - center[1]], dtype=float)
                # apply rotation and uniform scale
                rv = scale * np.array([cosA * vec[0] - sinA * vec[1],
                                       sinA * vec[0] + cosA * vec[1]], dtype=float)
                transformed_pts.append(np.array([center[0] + rv[0], center[1] + rv[1]], dtype=float))
            # Translate so the dragged corner exactly hits (new_x, new_y)
            dragged_transformed = transformed_pts[corner_idx]
            t_dx = new_x - dragged_transformed[0]
            t_dy = new_y - dragged_transformed[1]
            new_pts = [pt + np.array([t_dx, t_dy]) for pt in transformed_pts]
            # Solve affine mapping base -> new_pts (using three base corners as before)
            base_triplet = [base_pts[0], base_pts[1], base_pts[3]]  # TL, TR, BL
            target_triplet = [new_pts[0], new_pts[1], new_pts[3]]
            A = []; b = []
            for base, target in zip(base_triplet, target_triplet):
                x, y, _ = base
                X, Y = target
                A.extend([[x, y, 1, 0, 0, 0],
                          [0, 0, 0, x, y, 1]])
                b.extend([X, Y])
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            a11, a12, t1, a21, a22, t2 = sol
            new_transform = np.array([[a11, a12, t1],
                                      [a21, a22, t2],
                                      [0,   0,   1]], dtype=float)
            return new_transform

        elif mode == "affine":
            # Affine: 6-DOF linear transform (keeps parallel sides). Use 3 points to solve exactly.
            # Choose three base corners (including the dragged one) and their desired targets:
            if corner_idx == 0:  # top-left dragged
                base_triplet = [base_pts[1], base_pts[3], base_pts[0]]      # base TR, BL, TL
                target_triplet = [tgt_positions[1], tgt_positions[3], tgt_positions[0]]
            elif corner_idx == 1:  # top-right dragged
                base_triplet = [base_pts[0], base_pts[2], base_pts[1]]      # base TL, BR, TR
                target_triplet = [tgt_positions[0], tgt_positions[2], tgt_positions[1]]
            elif corner_idx == 2:  # bottom-right dragged
                base_triplet = [base_pts[1], base_pts[3], base_pts[2]]      # base TR, BL, BR
                target_triplet = [tgt_positions[1], tgt_positions[3], tgt_positions[2]]
            elif corner_idx == 3:  # bottom-left dragged
                base_triplet = [base_pts[0], base_pts[2], base_pts[3]]      # base TL, BR, BL
                target_triplet = [tgt_positions[0], tgt_positions[2], tgt_positions[3]]
            # Set up linear system A * params = b for affine (params = [a11,a12,t1,a21,a22,t2])
            A = []
            b = []
            for bp, tp in zip(base_triplet, target_triplet):
                x, y, _ = bp
                X, Y = tp
                A.extend([[x, y, 1, 0, 0, 0],
                          [0, 0, 0, x, y, 1]])
                b.extend([float(X), float(Y)])
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
            try:
                params = np.linalg.solve(A, b)  # solve 6x6 exactly
            except np.linalg.LinAlgError:
                try:
                    params, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None
            a11, a12, t1, a21, a22, t2 = params
            new_transform = np.array([[a11, a12, t1],
                                      [a21, a22, t2],
                                      [0,   0,   1]], dtype=float)
            return new_transform

        elif mode == "perspective":
            # Perspective: 8-DOF homography. Use all four corner correspondences (3 old + 1 new).
            base_points = [base_pts[0], base_pts[1], base_pts[2], base_pts[3]]
            target_points = tgt_positions  # list of 4 (including updated one)
            # Build linear system for homography: 8 unknowns h0..h7 (with h8=1 fixed)
            A = []
            for bp, tp in zip(base_points, target_points):
                x, y, _ = bp
                X, Y = tp
                A.extend([
                    [x, y, 1, 0, 0, 0, -X * x, -X * y, -X],
                    [0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y]
                ])
            A = np.array(A, dtype=float)
            # Solve A * h = 0 with h[8] = 1. We can take the 9th column as RHS and solve 8x8 for h0..h7.
            A_mat = A[:, :8]  # coefficients for h0..h7
            b_vec = -A[:, 8]  # bring -X or -Y to other side
            try:
                h0_7, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                return None
            h = np.append(h0_7, 1.0)  # append h8 = 1
            H = h.reshape(3, 3)
            return H

        return None  # just in case

    def save_to_file(self):
        """Save all rectangles (base size and transform matrix) to a JSON file."""
        file_path = filedialog.asksaveasfilename(title="Save Rectangles",
                                                 defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
        if not file_path:
            return  # user canceled
        data = []
        for rect in self.rectangles:
            data.append({
                "base_width": rect.base_width,
                "base_height": rect.base_height,
                "transform": rect.transform.flatten().tolist()
            })
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(self.rectangles)} rectangles to {file_path}")

    def load_from_file(self):
        """Load rectangles from a JSON file and redraw them on the canvas."""
        file_path = filedialog.askopenfilename(title="Load Rectangles",
                                               defaultextension=".json",
                                               filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Clear current rectangles from canvas
        for rect in self.rectangles:
            self.canvas.delete(rect.canvas_item)
        self.rectangles.clear()
        # Create rectangles from loaded data
        for entry in data:
            w = entry.get("base_width", 0)
            h = entry.get("base_height", 0)
            rect = Rectangle(0, 0, w, h)
            # Restore transform matrix
            if "transform" in entry:
                t_list = entry["transform"]
                try:
                    rect.transform = np.array(t_list, dtype=float).reshape(3, 3)
                except Exception:
                    rect.transform = np.array([[1, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 1]], dtype=float)
            # Draw the rectangle on canvas
            coords = [coord for point in rect.get_corners() for coord in point]
            rect.canvas_item = self.canvas.create_polygon(coords, outline="black", fill="", width=2)
            self.rectangles.append(rect)
        print(f"Loaded {len(self.rectangles)} rectangles from {file_path}")

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()

# If run as a script, start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TransformEditorApp(root)
    app.run()
