import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import cv2
import pyautogui
import pygetwindow as gw
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import threading
import time
import os
import win32clipboard
from io import BytesIO
import glob
import logging
import requests

class ScreenTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Translator")
        self.running = False

        # Prompt user to select download folder
        self.select_download_folder()

        self.create_widgets()

        self.live_feed_window = tk.Toplevel(self.root)
        self.live_feed_window.title("Live Feed")
        self.live_feed_label = tk.Label(self.live_feed_window)
        self.live_feed_label.pack()

        self.translated_feed_window = tk.Toplevel(self.root)
        self.translated_feed_window.title("Translated Feed")
        self.translated_feed_label = tk.Label(self.translated_feed_window)
        self.translated_feed_label.pack()

    # --- New method using the API to download a chapter and process translation ---
    def scrape_mangadex(self):
        # Prompt for the MangaDex chapter URL
        url = simpledialog.askstring("MangaDex URL", "Enter mangadex.org chapter URL:")
        if not url:
            return

        try:
            # Assume URL format: https://mangadex.org/chapter/<chapter_id>/<page>
            parts = url.strip().split('/')
            chapter_id = parts[4]
            print(f"Extracted chapter id: {chapter_id}")
        except Exception as e:
            print("Error parsing URL:", e)
            return

        # Call the Mangadex API for the chapter
        api_url = f"https://api.mangadex.org/at-home/server/{chapter_id}?forcePort443=false"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data.get("result") != "ok":
                print("API response not OK")
                return
        except Exception as e:
            print("Error calling Mangadex API:", e)
            return

        base_url = data["baseUrl"]
        chapter = data["chapter"]
        hash_val = chapter["hash"]
        pages = chapter["data"]

        downloaded_files = []
        for i, filename in enumerate(pages):
            image_url = f"{base_url}/data/{hash_val}/{filename}"
            try:
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                file_path = os.path.join(self.download_folder or ".", f"chapter_{chapter_id}_{i+1:03d}.png")
                with open(file_path, "wb") as f:
                    f.write(img_response.content)
                downloaded_files.append(file_path)
                print(f"Downloaded {file_path}")
            except Exception as e:
                print(f"Error downloading {image_url}: {e}")

        # Process each downloaded image through the translation flow
        translated_images = []
        for file_path in downloaded_files:
            translated_img = self.process_translation_for_image(file_path)
            if translated_img:
                translated_images.append(translated_img)

        if translated_images:
            stitched_image = self.stitch_images_vertically(translated_images)
            self.show_stitched_image(stitched_image)
            final_path = os.path.join(self.download_folder or ".", f"translated_manga_{chapter_id}_{int(time.time())}.png")
            stitched_image.save(final_path)
            print(f"Final manga saved as: {final_path}")
        else:
            print("No translated images obtained.")

    def process_translation_for_image(self, image_path):
        """
        Mimics your translation flow:
         - Loads the image,
         - Updates the live feed,
         - Copies the image to the clipboard,
         - Initiates the OCR/translation process,
         - Waits for the translated image and returns it.
        """
        try:
            img = Image.open(image_path)
            # Display image on the live feed (simulate a capture)
            imgtk = ImageTk.PhotoImage(img)
            self.live_feed_label.imgtk = imgtk
            self.live_feed_label.configure(image=imgtk)
            self.root.update()

            # Copy image to clipboard and invoke translation
            self.copy_image_to_clipboard(img)
            self.upload_image_to_ocr()
            # Wait and retrieve the translated image
            translated_img = self.wait_for_translated_image()
            return translated_img
        except Exception as e:
            print(f"Error translating {image_path}: {e}")
            return None

    def wait_for_translated_image(self, max_tries=5, retry_interval=1000):
        import glob
        tries = 0
        while tries < max_tries:
            pattern = os.path.join(self.download_folder, "translated_image_en*.png") if self.download_folder else "translated_image_en*.png"
            image_files = glob.glob(pattern)
            if image_files:
                image_path = image_files[0]
                print(f"Translated image found: {image_path}")
                # Use a context manager to open and copy the image, then close it.
                with Image.open(image_path) as img:
                    img_copy = img.copy()
                os.remove(image_path)  # Now that the file is closed, it can be removed.
                return img_copy
            else:
                tries += 1
                print(f"Waiting for translated image... {tries}/{max_tries}")
                time.sleep(retry_interval / 1000.0)
        print("Translated image not found after maximum retries.")
        return None

    def stitch_images_vertically(self, images):
        """
        Stitches a list of PIL Image objects vertically.
        """
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        max_width = max(widths)
        total_height = sum(heights)
        stitched = Image.new("RGB", (max_width, total_height), "white")
        y_offset = 0
        for img in images:
            stitched.paste(img, (0, y_offset))
            y_offset += img.height
        return stitched

    def show_stitched_image(self, image):
        """
        Displays the stitched manga image in a new window with a vertical scrollbar.
        """
        win = tk.Toplevel(self.root)
        win.title("Translated Manga")
        canvas = tk.Canvas(win)
        scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        imgtk = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=imgtk)
        canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
        canvas.config(scrollregion=canvas.bbox("all"))

    def copy_image_to_clipboard(self, image):
        """
        Copy an image to the Windows clipboard.
        """
        from io import BytesIO
        import win32clipboard
        output = BytesIO()
        image.convert('RGB').save(output, 'BMP')
        data = output.getvalue()[14:]
        output.close()

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()

    def create_widgets(self):
        self.window_list = ttk.Combobox(self.root, postcommand=self.preview_window)
        self.window_list.grid(row=0, column=0, padx=10, pady=10)

        # Speed control slider
        self.speed_var = tk.DoubleVar()
        self.speed_var.set(5.0)  # Default speed is set to 5 second
        self.speed_slider = tk.Scale(self.root, from_=0.1, to=15.0, resolution=0.1,
                                     orient='horizontal', label='Update Speed (s)',
                                     variable=self.speed_var)
        self.speed_slider.grid(row=2, column=0, columnspan=3, sticky='ew', padx=10, pady=10)

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_translation)
        self.start_button.grid(row=1, column=0, padx=10, pady=10)
        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_translation)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)

        self.translate_button = ttk.Button(self.root, text="Translate", command=self.manual_translate)
        self.translate_button.grid(row=1, column=2, padx=10, pady=10)  # Adjust column as needed

         # New button to scrape MangaDex chapter via API
        self.scrape_button = ttk.Button(self.root, text="Scrape MangaDex URL", command=self.scrape_mangadex)
        self.scrape_button.grid(row=1, column=3, padx=10, pady=10)

        self.enableRotate = tk.IntVar()
        self.rotate_toggle = ttk.Checkbutton(self.root, text="Rotate Image", variable=self.enableRotate, 
                                             onvalue=1, offvalue=0, command=self.on_rotate_toggle)
        self.rotate_toggle.grid(row=1, column=4, padx=10, pady=10)

        self.refresh_window_list()

    def on_rotate_toggle(self):
        if self.enableRotate.get() == 1:
            print("Enabled rotate image")
        else:
            print("Disabled rotate image")

    def select_download_folder(self):
        self.download_folder = filedialog.askdirectory(title="Select Folder for Downloaded Translations")
        if not self.download_folder:  # In case the user cancels the selection
            self.download_folder = None
            print("No folder selected, defaulting to current directory for downloads.")
        else:
            print(f"Download folder set to: {self.download_folder}")

    def display_and_delete_translated_image(self, max_tries=5, retry_interval=1000):
        # Use a list to hold the state across retries
        retry_state = {'tries': 0}

        def try_display_image():
            nonlocal max_tries  # Ensure we can access and modify max_tries
            pattern = os.path.join(self.download_folder, "translated_image_en*.png") if self.download_folder else "translated_image_en*.png"
            image_files = glob.glob(pattern)

            if image_files:
                image_path = image_files[0]
                print(f"Displaying image: {image_path}")

                img = Image.open(image_path)
                # Calculate the new size to fit the entire rotated image
                original_width, original_height = img.size
                diagonal_length = int((original_width ** 2 + original_height ** 2) ** 0.5)
                new_size = (diagonal_length, diagonal_length)

                # Create a new blank image with the new size and white background
                new_img = Image.new("RGB", new_size, "white")
                # Paste the original image into the center of the new blank image
                new_img.paste(img, ((new_size[0] - original_width) // 2, (new_size[1] - original_height) // 2))

                # Rotate the image by 45 degrees without cropping
                rotated_img = new_img.rotate(45, expand=True) if self.enableRotate.get() else new_img

                imgtk = ImageTk.PhotoImage(image=rotated_img)
                self.translated_feed_label.imgtk = imgtk  # Keep a reference!
                self.translated_feed_label.configure(image=imgtk)

                # Delete the image file after displaying
                os.remove(image_path)
                print(f"Deleted image: {image_path}")
            else:
                retry_state['tries'] += 1
                print(f"No translated image found to display. Retrying... {retry_state['tries']}/{max_tries}")
                if retry_state['tries'] < max_tries:
                    # Schedule another try
                    self.root.after(retry_interval, try_display_image)
                else:
                    print("Failed to find translated image after maximum retries.")

        # Start the first attempt
        self.root.after(0, try_display_image)


    def refresh_window_list(self):
        windows = gw.getAllTitles()
        self.window_list['values'] = windows

    def manual_translate(self):
        """Manually capture the screen and initiate the OCR process."""
        self.capture_and_update_feed()
        self.upload_image_to_ocr()
        self.display_and_delete_translated_image()

    def preview_window(self):
        selected_title = self.window_list.get()
        if selected_title:
            try:
                window = gw.getWindowsWithTitle(selected_title)[0]
                if not window.isActive:
                    window.activate()
                preview = pyautogui.screenshot(region=(
                    window.left, window.top, window.width, window.height))
                preview = cv2.cvtColor(np.array(preview), cv2.COLOR_RGB2BGR)
                preview = cv2.resize(preview, (320, 240))
                img = Image.fromarray(preview)
                imgtk = ImageTk.PhotoImage(image=img)
                self.live_feed_label.imgtk = imgtk
                self.live_feed_label.configure(image=imgtk)
            except Exception as e:
                print(f"Window activation failed: {e}")

    def start_translation(self):
        self.running = True
        threading.Thread(target=self.process_screen, daemon=True).start()

    def stop_translation(self):
        self.running = False

    def process_screen(self):
        while self.running:
            self.capture_and_update_feed()
            self.upload_image_to_ocr()
            self.display_and_delete_translated_image()
            time.sleep(self.speed_var.get())

    def activate_window(self, title_contains):
        """Maximize the window if it's not already maximized."""
        windows = gw.getWindowsWithTitle(title_contains)
        for window in windows:
            if title_contains in window.title:
                window.activate()
                time.sleep(1)  # Give it a moment to maximize
                break  # Assuming you only need to maximize the first matching window


    def capture_and_update_feed(self):
        selected_title = self.window_list.get()
        if selected_title:
            try:
                window = gw.getWindowsWithTitle(selected_title)[0]
                if not window.isActive:
                    window.activate()
                screenshot = pyautogui.screenshot(region=(
                    window.left, window.top, window.width, window.height))
                self.update_live_feed(screenshot)
                self.copy_image_to_clipboard(screenshot)
            except Exception as e:
                print(f"Error during screen capture: {e}")

    def update_live_feed(self, image):
        img = Image.fromarray(np.array(image))
        imgtk = ImageTk.PhotoImage(image=img)
        self.live_feed_label.imgtk = imgtk
        self.live_feed_label.configure(image=imgtk)

    def copy_image_to_clipboard(self, image):
        """Copy an image to the Windows clipboard."""
        output = BytesIO()
        image.convert('RGB').save(output, 'BMP')
        data = output.getvalue()[14:]  # The file header offest for BMP files is 14 bytes
        output.close()

        win32clipboard.OpenClipboard()  # Open the clipboard
        win32clipboard.EmptyClipboard()  # Clear the clipboard contents
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)  # Set the clipboard data as DIB which is a Device Independent Bitmap
        win32clipboard.CloseClipboard()  # Close the clipboard

    def upload_image_to_ocr(self):
        # original location of user cursor
        self.activate_window("Google Translate")
        original_x, original_y = pyautogui.position()
        click_image("translateimages0.png")
        # wait 3 seconds for the image to be uploaded
        time.sleep(3)
        click_image("translateimages1.png", max_tries=3, retry_interval=1)
        click_image("translateimages2.png")
        # move the cursor back to the original location
        pyautogui.moveTo(original_x, original_y)
        print(f"Moved cursor back to original location: ({original_x}, {original_y})")


def click_image(image_name, confidence=0.9, max_tries=3, retry_interval=1):

    current_confidence = confidence
    tries = 0
    while tries < max_tries:
        image_path = os.path.join("images", image_name)
        worked = False
        current_confidence = confidence
        while not worked and current_confidence >= 0.6:  # Lower threshold to 0.6 or adjust based on your testing
            try:
                x, y = pyautogui.locateCenterOnScreen(image_path, confidence=current_confidence)
                # Perform an instant click at the located position
                pyautogui.click(x, y)
                worked = True
            except pyautogui.ImageNotFoundException:
                current_confidence -= 0.1  # Decrease confidence level and try again
                print(f"Trying with reduced confidence: {current_confidence}")
            except Exception as e:
                print(f"Unexpected error: {e}")
                break  # Exit loop on unexpected errors

        if worked:
            print(f"Clicked on image: {image_name}")
            return worked
        else:
            tries += 1
            print(f"Failed to locate image {image_path}. Retrying... {tries}/{max_tries}")
            time.sleep(retry_interval)

    return worked



if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenTranslatorApp(root)
    root.mainloop()
