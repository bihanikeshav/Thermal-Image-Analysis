"""GUI application for thermal image analysis."""
import os
import pickle
import sys
from tkinter.messagebox import showwarning
from fractions import Fraction
import numpy as np
import pygame
import pygame_gui as pygui
from scipy.ndimage.interpolation import zoom
from thermal_base import ThermalImage
from thermal_base import utils as ThermalImageHelpers

from utils import WindowHandler, openImage, saveImage

pygame.init()
WINDOW_SIZE = (1210, 910)
NEW_FILE = False


class Manager(pygui.UIManager):
    """Class for manager.

    A manager is a set of menu buttons and descriptions.
    A manager is assigned to each page of the application.
    """

    def __init__(self, buttons, textbox=None, fields=None):
        """Initilizer for manager."""
        super().__init__(WINDOW_SIZE)
        self.buttons = [
            (
                pygui.elements.UIButton(
                    relative_rect=pygame.Rect(pos, size), text=text, manager=self
                ),
                func,
            )
            for pos, size, text, func in buttons
        ]
        if textbox:
            self.textbox = pygui.elements.ui_text_box.UITextBox(
                html_text=textbox[2],
                relative_rect=pygame.Rect(textbox[:2]),
                manager=self,
            )
        if fields:
            self.fields = [
                (
                    pygui.elements.ui_text_entry_line.UITextEntryLine(
                        relative_rect=pygame.Rect((pos[0], pos[1] + 40), size),
                        manager=self,
                    ),
                    pygui.elements.ui_text_box.UITextBox(
                        html_text=text,
                        relative_rect=pygame.Rect(pos, (-1, -1)),
                        manager=self,
                    ),
                )
                for pos, size, text in fields
            ]

    def process_events(self, event):
        """Process button presses."""
        if event.type == pygame.USEREVENT:
            if event.user_type == pygui.UI_BUTTON_PRESSED:
                for button, func in self.buttons:
                    if event.ui_element == button:
                        func()

        super().process_events(event)


class Window:
    """Class that handles the main window."""

    fonts = [
        pygame.font.SysFont("monospace", 20),
        pygame.font.SysFont("monospace", 24),
        pygame.font.SysFont("arial", 18),
    ]

    cursors = [
        pygame.image.load("./assets/cursors/pointer.png"),
        pygame.image.load("./assets/cursors/crosshair.png"),
    ]
    logo = pygame.transform.scale(pygame.image.load("./assets/DTlogo.png"), (100, 100))
    clip = lambda x, a, b: a if x < a else b if x > b else x

    @staticmethod
    def renderText(surface, text, location):
        """Render text at a given location."""
        whitetext = Window.fonts[2].render(text, 1, (255, 255, 255))
        Window.fonts[0].set_bold(True)
        blacktext = Window.fonts[2].render(text, 1, (0, 0, 0))
        Window.fonts[0].set_bold(False)

        textrect = whitetext.get_rect()
        for i in range(-3, 4):
            for j in range(-3, 4):
                textrect.center = [a + b for a, b in zip(location, (i, j))]
                surface.blit(blacktext, textrect)
        textrect.center = location
        surface.blit(whitetext, textrect)

    def __init__(self, thermal_image=None, filename=None):
        """Initializer for the main window."""
        self.exthandler = WindowHandler()
        if thermal_image is not None:
            mat = thermal_image.thermal_np.astype(np.float32)

            if mat.shape != (512, 640):
                y0, x0 = mat.shape
                mat = zoom(mat, [512 / y0, 640 / x0])

            self.mat = mat
            self.mat_orig = mat.copy()
            self.mat_emm = mat.copy()
            self.raw = thermal_image.raw_sensor_np
            self.meta = thermal_image.meta
            self.overlays = pygame.Surface((640, 512), pygame.SRCALPHA)
            self.thermalData = thermal_image.thermal_np
            # np.savetxt(filename, self.thermalData, delimiter=",")

        else:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            print(filename)
            self.mat = data.mat
            self.mat_orig = data.mat_orig
            self.mat_emm = data.mat_emm
            self.raw = data.raw
            self.meta = data.meta
            self.thermalData=data.thermal_np
            self.overlays = pygame.image.fromstring(data.overlays, (640, 512), "RGBA")

            for entry in data.tableEntries:
                self.exthandler.addToTable(entry)
            self.exthandler.loadGraph(data.plots)
            self.exthandler.addRects(data.rects)

        self.colorMap = "jet"
        self.lineNum = 0
        self.boxNum = 0
        self.spotNum = 0
        self.areaMode = "poly"
        self.selectionComplete = False
        self.work("colorMap", self.colorMap)

        self.mode = "main"
        # Dictionary of pages. Each page is a manager.
        self.managers = {}
        self.managers["main"] = Manager(
            buttons=[
                ((15, 15), (215, 45), "Spot marking", lambda: self.changeMode("spot")),
                (
                    (15, 75),
                    (215, 45),
                    "Line measurement",
                    lambda: self.changeMode("line"),
                ),
                ((15, 135), (215, 45), "Area marking", lambda: self.changeMode("area")),
                ((15, 195), (215, 45), "ROI scaling", lambda: self.changeMode("scale")),
                (
                    (15, 255),
                    (215, 45),
                    "Change colorMap",
                    lambda: self.changeMode("colorMap"),
                ),
                (
                    (15, 315),
                    (215, 45),
                    "Emissivity scaling",
                    lambda: self.changeMode("emissivity"),
                ),
                (
                    (15, 470),
                    (215, 45),
                    "Reset modifications",
                    lambda: self.work("reset"),
                ),
                ((15, 530), (100, 45), "Open", lambda: self.work("open")),
                ((130, 530), (100, 45), "Save", lambda: np.savetxt(filename, self.thermalData, delimiter=",")
),
            ]
        )
        self.managers["spot"] = Manager(
            buttons=[((15, 530), (215, 45), "Back", lambda: self.changeMode("main"))],
            textbox=((15, 15), (215, -1), "Click to mark spots"),
        )
        self.managers["line"] = Manager(
            buttons=[
                (
                    (15, 410),
                    (215, 45),
                    "Continue",
                    lambda: self.work("line") if len(self.linePoints) == 2 else None,
                ),
                ((15, 470), (215, 45), "Reset", lambda: self.changeMode("line")),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Click to mark the end points of the line. Click continue to get plot and reset to remove the line",
            ),
        )
        self.managers["area"] = Manager(
            buttons=[
                ((15, 470), (215, 45), "Continue", lambda: self.work("area")),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Click and drag to draw selection. Select continue to mark",
            ),
        )
        self.managers["scale"] = Manager(
            buttons=[
                (
                    (15, 270),
                    (215, 45),
                    "Switch to rect mode",
                    lambda: self.work("scale", "switchMode"),
                ),
                (
                    (15, 350),
                    (215, 45),
                    "Continue",
                    lambda: self.work("scale", "scale")
                    if self.selectionComplete
                    else None,
                ),
                (
                    (15, 410),
                    (215, 45),
                    "Reset scaling",
                    lambda: self.work("scale", "reset"),
                ),
                (
                    (15, 470),
                    (215, 45),
                    "Reset selection",
                    lambda: self.changeMode("scale"),
                ),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Click to mark vertices. Press Ctrl and click to close the selection",
            ),
        )
        self.managers["colorMap"] = Manager(
            buttons=[
                ((15, 15), (215, 45), "Jet", lambda: self.work("colorMap", "jet")),
                ((15, 75), (215, 45), "Hot", lambda: self.work("colorMap", "hot")),
                ((15, 135), (215, 45), "Cool", lambda: self.work("colorMap", "cool")),
                ((15, 195), (215, 45), "Gray", lambda: self.work("colorMap", "gray")),
                (
                    (15, 255),
                    (215, 45),
                    "Inferno",
                    lambda: self.work("colorMap", "inferno"),
                ),
                (
                    (15, 315),
                    (215, 45),
                    "Copper",
                    lambda: self.work("colorMap", "copper"),
                ),
                (
                    (15, 375),
                    (215, 45),
                    "Winter",
                    lambda: self.work("colorMap", "winter"),
                ),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ]
        )
        self.managers["emissivity"] = Manager(
            buttons=[
                (
                    (15, 410),
                    (215, 45),
                    "Continue",
                    lambda: self.work("emissivity", "update")
                    if self.selectionComplete
                    else None,
                ),
                (
                    (15, 470),
                    (215, 45),
                    "Reset",
                    lambda: self.work("emissivity", "reset"),
                ),
                ((15, 530), (215, 45), "Back", lambda: self.changeMode("main")),
            ],
            textbox=(
                (15, 15),
                (215, -1),
                "Select region, enter values and press continue. Click to mark vertices."
                "Press Ctrl and click to close the selection",
            ),
            fields=[
                ((15, 165), (215, 45), "Emissivity:"),
                ((15, 240), (215, 45), "Reflected Temp.:"),
                ((15, 315), (215, 45), "Atmospheric Temp.:"),
            ],
        )

        self.linePoints = []

        self.cursor_rect = self.cursors[0].get_rect()
        self.background = pygame.Surface(WINDOW_SIZE)
        self.background.fill((0, 0, 0))

    def changeMode(self, mode):
        """Change mode."""
        # Mode change - reset handler
        if self.mode == "line":
            if mode in ("main", "line"):
                self.linePoints = []

        if self.mode in ("scale", "area", "emissivity"):
            if mode in ("main", "scale", "area"):
                self.selectionComplete = False
                self.linePoints = []

        self.mode = mode

    def work(self, mode, *args):
        """Work based on mode."""
        if mode == "reset":
            # Resetting overlays and plots
            self.overlays = pygame.Surface((WINDOW_SIZE[0] - 245, 512), pygame.SRCALPHA)
            self.lineNum = 0
            self.boxNum = 0
            self.spotNum = 0
            self.exthandler.killThreads()
            self.exthandler = WindowHandler()

            # Resetting ROI scaling
            self.work("scale", "reset")
            # Resetting Emissivity changes
            self.work("emissivity", "reset")

        if mode == "open":
            global NEW_FILE
            NEW_FILE = True

        if mode == "line":
            self.lineNum += 1
            linePoints = [
                [a - b for a, b in zip(points, (245, 15))] for points in self.linePoints
            ]
            pygame.draw.line(
                self.overlays, (255, 255, 255), linePoints[0], linePoints[1], 3
            )

            center = (
                (linePoints[0][0] + linePoints[1][0]) / 2,
                (linePoints[0][1] + linePoints[1][1]) / 2,
            )

            self.renderText(self.overlays, f"l{self.lineNum}", center)

            self.exthandler.linePlot(
                self.mat_emm,
                f"l{self.lineNum}",
                np.array(linePoints[0][::-1]),
                np.array(linePoints[1][::-1]),
            )
            self.linePoints = []

        if mode == "spot":
            self.spotNum += 1
            self.renderText(
                self.overlays,
                f"s{self.spotNum}",
                (self.mx - 245 + 15, self.my - 15 - 13),
            )
            pygame.draw.line(
                self.overlays,
                (255, 255, 255),
                (self.mx - 245, self.my - 15 - 5),
                (self.mx - 245, self.my - 15 + 5),
                3,
            )
            pygame.draw.line(
                self.overlays,
                (255, 255, 255),
                (self.mx - 245 - 5, self.my - 15),
                (self.mx - 245 + 5, self.my - 15),
                3,
            )
            val = self.mat_emm[self.cy - 15, self.cx - 245]
            self.exthandler.addToTable([f"s{self.spotNum}", val, val, val])

        if mode == "area":
            if self.selectionComplete:
                points = [(a - 245, b - 15) for a, b in self.linePoints]
                x_coords, y_coords = zip(*points)
                xmin = min(x_coords)
                xmax = max(x_coords)
                ymin = min(y_coords)
                ymax = max(y_coords)
                if xmin == xmax or ymin == ymax:
                    return
                self.boxNum += 1
                chunk = self.mat_emm[ymin:ymax, xmin:xmax]
                self.exthandler.addToTable(
                    [f"a{self.boxNum}", np.min(chunk), np.max(chunk), np.mean(chunk)]
                )
                self.exthandler.addRects([[xmin, xmax, ymin, ymax]])
                pygame.draw.lines(self.overlays, (255, 255, 255), True, points, 3)
                self.renderText(
                    self.overlays, f"a{self.boxNum}", (xmin + 12, ymin + 10)
                )

        if mode == "colorMap":
            self.colorMap = args[0]

            minVal = np.min(self.mat)
            delVal = np.max(self.mat) - minVal
            self.cbarVals = [minVal + i * delVal / 4 for i in range(5)][::-1]

            cbar = np.row_stack(20 * (np.arange(256),))[:, ::-1].astype(np.float32)

            self.image = ThermalImageHelpers.cmap_matplotlib(self.mat, args[0])
            cbar = ThermalImageHelpers.cmap_matplotlib(cbar, args[0])

            self.imsurf = pygame.Surface((WINDOW_SIZE[0] - 245, 512))
            self.imsurf.blit(
                pygame.surfarray.make_surface(
                    np.transpose(self.image[..., ::-1], (1, 0, 2))
                ),
                (0, 0),
            )
            self.imsurf.blit(pygame.surfarray.make_surface(cbar[..., ::-1]), (663, 85))
            for i, val in enumerate(self.cbarVals):
                self.imsurf.blit(
                    self.fonts[0].render(f"{val:.1f}", 1, (255, 255, 255)),
                    (690, 75 + i * 65),
                )
            self.imsurf.blit(
                self.fonts[0].render("\N{DEGREE SIGN}" + "C", 1, (255, 255, 255)),
                (660, 60),
            )
            self.imsurf.blit(self.logo, (658, 405))

        if mode == "scale":
            if args[0] == "reset":
                self.mat = self.mat_emm.copy()
                self.work("colorMap", self.colorMap)

            if args[0] == "switchMode":
                self.managers["scale"].buttons[0][0].set_text(
                    f"Switch to {self.areaMode} mode"
                )
                self.areaMode = "poly" if self.areaMode == "rect" else "rect"
                self.changeMode("scale")

            if args[0] == "scale":

                chunk = self.mat_emm[
                    ThermalImageHelpers.coordinates_in_poly(
                        [(x - 245, y - 15) for x, y in self.linePoints], self.raw.shape
                    )
                ]

                if len(chunk) > 0:
                    self.mat = np.clip(self.mat_emm, np.min(chunk), np.max(chunk))
                    self.work("colorMap", self.colorMap)

        if mode == "emissivity":
            if args[0] == "update":
                emmissivity = self.managers["emissivity"].fields[0][0].get_text()
                ref_temp = self.managers["emissivity"].fields[1][0].get_text()
                atm_temp = self.managers["emissivity"].fields[2][0].get_text()

                np_indices = ThermalImageHelpers.coordinates_in_poly(
                    [(x - 245, y - 15) for x, y in self.linePoints], self.raw.shape
                )
                self.mat_emm[
                    np_indices
                ] = ThermalImageHelpers.change_emissivity_for_roi(
                    thermal_np=None,
                    meta=self.meta,
                    roi_contours=None,
                    raw_roi_values=self.raw[np_indices],
                    indices=None,
                    new_emissivity=emmissivity,
                    ref_temperature=ref_temp,
                    atm_temperature=atm_temp,
                    np_indices=True,
                )

            if args[0] == "reset":
                self.mat_emm = self.mat_orig.copy()

            self.work("scale", "reset")

    def process(self, event):
        """Process input event."""
        self.mx, self.my = self.cursor_rect.center = pygame.mouse.get_pos()
        self.cx = Window.clip(self.mx, 245, 884)
        self.cy = Window.clip(self.my, 15, 526)
        self.cursor_in = (245 < self.mx < 885) and (15 < self.my < 527)
        self.managers[self.mode].process_events(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.cursor_in:
                if self.mode == "line":
                    if len(self.linePoints) < 2:
                        self.linePoints.append((self.mx, self.my))
                if (
                    self.mode == "scale" and self.areaMode == "poly"
                ) or self.mode == "emissivity":
                    if self.selectionComplete:
                        self.linePoints = []
                        self.selectionComplete = False
                    self.linePoints.append((self.mx, self.my))
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        if len(self.linePoints) > 2:
                            self.selectionComplete = True
                if (
                    self.mode == "scale" and self.areaMode == "rect"
                ) or self.mode == "area":
                    self.changeMode(self.mode)
                    self.linePoints.append((self.mx, self.my))

                if self.mode == "spot":
                    self.work("spot")

        if event.type == pygame.MOUSEBUTTONUP:
            if (
                self.mode == "scale" and self.areaMode == "rect"
            ) or self.mode == "area":
                if len(self.linePoints) == 1:
                    self.linePoints.append((self.cx, self.linePoints[0][1]))
                    self.linePoints.append((self.cx, self.cy))
                    self.linePoints.append((self.linePoints[0][0], self.cy))
                    self.selectionComplete = True

    def update(self, time_del):
        """Update events."""
        self.managers[self.mode].update(time_del)

    def draw(self, surface):
        """Draw contents on screen."""
        surface.blit(self.background, (0, 0))
        surface.blit(self.imsurf, (245, 15))
        surface.blit(self.overlays, (245, 15))

        pygame.draw.rect(surface, (255, 255, 255), (245, 540, 760, 35), 1)
        self.managers[self.mode].draw_ui(surface)
        surface.blit(
            self.fonts[1].render(
                f"x:{self.cx - 245:03}   y:{self.cy - 15:03}   temp:{self.mat_emm[self.cy - 15, self.cx - 245]:.4f}",
                1,
                (255, 255, 255),
            ),
            (253, 544),
        )

        if self.mode == "line":
            if len(self.linePoints) == 1:
                pygame.draw.line(
                    surface, (255, 255, 255), self.linePoints[0], (self.cx, self.cy), 3
                )
            if len(self.linePoints) == 2:
                pygame.draw.line(
                    surface, (255, 255, 255), self.linePoints[0], self.linePoints[1], 3
                )

        if (
            self.mode == "scale" and self.areaMode == "poly"
        ) or self.mode == "emissivity":
            if len(self.linePoints) > 0:
                pygame.draw.lines(
                    surface,
                    (255, 255, 255),
                    self.selectionComplete,
                    self.linePoints
                    + ([] if self.selectionComplete else [(self.cx, self.cy)]),
                    3,
                )
        if (self.mode == "scale" and self.areaMode == "rect") or self.mode == "area":
            if not self.selectionComplete:
                if len(self.linePoints) > 0:
                    pygame.draw.lines(
                        surface,
                        (255, 255, 255),
                        True,
                        self.linePoints
                        + [
                            (self.cx, self.linePoints[0][1]),
                            (self.cx, self.cy),
                            (self.linePoints[0][0], self.cy),
                        ],
                        3,
                    )
            else:
                pygame.draw.lines(surface, (255, 255, 255), True, self.linePoints, 3)

        surface.blit(self.cursors[self.cursor_in], self.cursor_rect)

import webbrowser
import exifread

def plot_graph(surface, data):
    """Plot a graph on the Pygame surface."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(data, cmap='jet', extent=[0, data.shape[1], 0, data.shape[0]], origin='upper')

    # Add a colorbar for each image
    cbar = plt.colorbar(ax.get_images()[0], ax=ax)
    cbar.set_label('Temperature')

    # Show the plot for each image
    plt.title(f'Image with Hotspots: Max temp: {round(np.max(data), 1)}°C')

    # Render the plot to the Pygame surface
    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    # Create a Pygame surface from the plot data
    img = pygame.image.fromstring(raw_data, size, "RGB")
    surface.blit(img, (0, 0))

def extract_gps_coords(image_path):
    with open(image_path, 'rb') as file:
        tags = exifread.process_file(file)

    latitude = tags.get('GPS GPSLatitude')
    longitude = tags.get('GPS GPSLongitude')
    # return latitude, longitude
    if latitude and longitude:
        if isinstance(latitude, exifread.classes.IfdTag):
            lat_deg, lat_min, lat_sec_frac = map(str, latitude.values)
        else:
            lat_deg, lat_min, lat_sec_frac = map(str, latitude[0].values)

        if isinstance(longitude, exifread.classes.IfdTag):
            lon_deg, lon_min, lon_sec_frac = map(str, longitude.values)
        else:
            lon_deg, lon_min, lon_sec_frac = map(str, longitude[0].values)

        # Convert seconds fraction to decimal
        lat_sec = float(Fraction(lat_sec_frac))
        lon_sec = float(Fraction(lon_sec_frac))

        # Calculate the decimal degrees from degrees, minutes, and seconds
        lat_decimal = float(lat_deg) + float(lat_min) / 60 + lat_sec / 3600
        lon_decimal = float(lon_deg) + float(lon_min) / 60 + lon_sec / 3600

        # Check for North or South hemisphere and East or West longitude
        lat_direction = str(tags.get('GPS GPSLatitudeRef'))
        lon_direction = str(tags.get('GPS GPSLongitudeRef'))

        if lat_direction == 'S':
            lat_decimal *= -1
        if lon_direction == 'W':
            lon_decimal *= -1

        return lat_decimal, lon_decimal
    else:
        return None, None

def open_google_maps(latitude, longitude):
    if latitude is not None and longitude is not None:
        # Generate the Google Maps URL
        maps_url = f"https://www.google.com/maps/place/{latitude},{longitude}"
        
        # Open the URL in the default web browser
        webbrowser.open(maps_url)
    else:
        print("GPS coordinates not found.")

# def get_mouse_position_on_plot(mouse_pos, plot_rect, data_shape):
#     left=80
#     width=400
#     top=90
#     height=320
#     print(mouse_pos)
    # Calculate the relative position of the mouse on the Matplotlib plot
        
def get_mouse_position_on_plot(mouse_pos, plot_rect, data_shape):
    from_range = (80, 85, 476, 425)
    # Target range
    to_range = (0, 0, 640, 512)

    # Calculate the relative position of the mouse on the Matplotlib plot
    print(plot_rect.left, plot_rect.bottom, plot_rect.width, plot_rect.height)
    x_rel = (mouse_pos[0] - plot_rect.left) / plot_rect.width * data_shape[1]
    y_rel = (plot_rect.bottom - mouse_pos[1]) / plot_rect.height * data_shape[0]

    # Ensure the coordinates are within the valid range
    x_rel = np.clip(x_rel, 0, data_shape[1] - 1)
    y_rel = np.clip(y_rel, 0, data_shape[0] - 1)


    x_scaled,y_scaled= scale_point((x_rel, y_rel), from_range, to_range)

    # Convert to integer indices
    x_index = int(round(x_scaled))
    y_index = int(round(y_scaled))

    # Original range

    # Scale the point
    return x_index,y_index

def scale_point(point, from_range, to_range):
    """
    Scale a point from one range to another.

    Parameters:
    - point: Tuple (x, y) representing the point to scale.
    - from_range: Tuple (x_min, y_min, x_max, y_max) representing the original range.
    - to_range: Tuple (x_min, y_min, x_max, y_max) representing the target range.

    Returns:
    - Tuple (scaled_x, scaled_y) representing the scaled point.
    """
    x, y = point
    x_min, y_min, x_max, y_max = from_range
    to_x_min, to_y_min, to_x_max, to_y_max = to_range

    scaled_x = (x - x_min) / (x_max - x_min) * (to_x_max - to_x_min) + to_x_min
    scaled_y = (y - y_min) / (y_max - y_min) * (to_y_max - to_y_min) + to_y_min

    return scaled_x, scaled_y

def render_text(surface, text, position, font, color):
    """
    Render and display text on the Pygame surface.

    Parameters:
    - surface: Pygame surface to render the text on.
    - text: The text to display.
    - position: Tuple (x, y) representing the position of the text.
    - font: Pygame font object.
    - color: Tuple (R, G, B) representing the color of the text.
    """
    text_render = font.render(text, True, color)
    surface.blit(text_render, position)

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    NEXT_BUTTON_RECT = pygame.Rect(20, 20, 100, 40)
    MAP_BUTTON_RECT = pygame.Rect(20, 80, 150, 40)
    TEMP_BUTTON_RECT = pygame.Rect(150, 20, 100, 40)
    global dynamic
    dynamic="30"
    current_file_index = 0  
    surface = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Detect Thermal Image Analysis Tool")

    clock = pygame.time.Clock()
    done = False
    NEW_FILE = True
    filenames, hotspot_threshold = openImage()
    font = pygame.font.Font(None, 36)
    x_index = 0
    y_index = 0


    for file in filenames:
        image = ThermalImage(file, camera_manufacturer="flir")
        max=np.max(image.thermal_np)
        if max >= 33:
            np.savetxt(file.split(".")[0] + ".csv", image.thermal_np, delimiter=",")
        
        print("-")
    print("done")
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()

                if NEXT_BUTTON_RECT.collidepoint(mouse_pos):
                    current_file_index = (current_file_index + 1) % len(filenames)
                # Check if "Google Maps" button is clicked
                elif MAP_BUTTON_RECT.collidepoint(mouse_pos):
                    open_google_maps(latitude, longitude)
                elif plot_rect.collidepoint(mouse_pos):
                    # Get the mouse position on the Matplotlib plot
                    # x_index, y_index = get_mouse_position_on_plot(mouse_pos, plot_rect, data.shape)
                    x_index=int((mouse_pos[0]-150)/1.6)
                    y_index=int((mouse_pos[1]-175)/1.6)
                    # y_index=abs(y_index-348)
                    if x_index>=len(data[0])-1:
                        x_index=len(data[0])-1
                    if y_index>=len(data)-1:
                        y_index=len(data)-1
                    print(f"Clicked at position ({x_index}, {y_index})")
                    # surface.blit(Window.fonts[2].render("Next", 1, (0, 0, 0)), (90, 90))
                    dynamic=str(round(data[y_index, x_index], 2))
                    # render_text(surface, f"Temperature: {data[y_index, x_index]}", (500, 10), font, (255, 255, 255))


        if NEW_FILE:
            filename = filenames[current_file_index]  # Get the current filename
            if filename:
                surface.fill((0, 0, 0))
# /                surface.blit(Window.fonts[2].render("Loading...", 1, (255, 255, 255)), (460, 275))
#                 pygame.display.update()
                newwindow = None

                try:
                    image = ThermalImage(filename, camera_manufacturer="DJI")
                    try:
                        latitude, longitude = extract_gps_coords(filename)
                    except:
                        latitude, longitude = 0,0

                    # if latitude is not None and longitude is not None:
                    #     print(f"GPS Coordinates: {latitude}, {longitude}")
                except Exception as err:
                    image = ThermalImage(filename, camera_manufacturer="DJI")
                    # latitude, longitude = extract_gps_coords(filename)
                    try:
                        latitude, longitude = extract_gps_coords(filename)
                    except:
                        latitude, longitude = 0,0

                    # if latitude is not None and longitude is not None:
                    #     print(f"GPS Coordinates: {latitude}, {longitude}")

                np.savetxt(filename.split(".")[0] + ".csv", image.thermal_np, delimiter=",")

                # csv_path = os.path.join(folder_path, filename)
                df = pd.read_csv(filename.split(".")[0]+".csv", header=None)
                # Reshape the DataFrame into a 2D NumPy array
                data = df.values

                # Check if there are hotspots in the image
                max=np.max(data)
                if max >= hotspot_threshold:
                    # Create a figure and axes
                    fig, ax = plt.subplots(figsize=(12, 9))

                    # Plot the image using the "jet" colormap
                    im = ax.imshow(data, cmap='jet', extent=[0, data.shape[1], 0, data.shape[0]], origin='upper')

                    # Add a colorbar for each image
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Temperature')
                    max_index = np.unravel_index(np.argmax(data), data.shape)
                    # Draw a circle around the maximum value
                    circle_radius = 5  # Set your desired circle radius
                    circle = patches.Circle((max_index[1], abs(348-max_index[0])), circle_radius, edgecolor='k', linewidth=2, facecolor='none')
                    ax.add_patch(circle)

                    ax.plot(x_index, abs(348-y_index), color='white', marker='x', markersize=10, markeredgewidth=2)

                    # Show the plot for each image
                    flast=filename.split('/')[-1]
                    plt.title(f'Image with Hotspots: {flast}, Max temp: {round(max, 1)}°')
                    plt.savefig('Res_'+filename.split('/')[-1])
                    plt.close()
                    # plt.show()
                # Draw "Next" button
                    image = pygame.image.load('Res_'+filename.split('/')[-1])
                    plot_rect = surface.blit(image, (0, 0))

                    pygame.draw.rect(surface, (0, 255, 0), NEXT_BUTTON_RECT)
                    surface.blit(Window.fonts[2].render("Next", 1, (0, 0, 0)), (30, 30))

                    # Draw "Google Maps" button
                    pygame.draw.rect(surface, (0, 255, 0), MAP_BUTTON_RECT)
                    surface.blit(Window.fonts[2].render("Google Maps", 1, (0, 0, 0)), (30, 90))

                    pygame.draw.rect(surface, (0, 255, 255), TEMP_BUTTON_RECT)
                    surface.blit(Window.fonts[2].render("Temp: "+dynamic, 1, (0, 0, 0)), (160, 30))
                    pygame.display.flip()
                    clock.tick(30)

    pygame.quit()


