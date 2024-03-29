import os
import cv2
import re
import time
from threading import RLock
import numpy as np
import pdb
from codependent_thread import CodependentThread
from image_misc import cv2_imshow_rgb, read_cam_frame, crop_to_square
from misc import tsplit, get_files_list

import caffe
std=[0.229,0.224,0.225]
mean=[0.458,0.456,0.406]
class InputImageFetcher(CodependentThread):
    '''Fetches images from a webcam or loads from a directory.'''
    
    def __init__(self, settings):
        CodependentThread.__init__(self, settings.input_updater_heartbeat_required)
        self.daemon = True
        self.lock = RLock()
        self.quit = False
        self.latest_frame_idx = -1
        self.latest_frame_data = None
        self.latest_frame_is_from_cam = False

        # True for loading from file, False for loading from camera
        self.static_file_mode = True
        self.settings = settings

        # True for streching the image, False for cropping largest square
        self.static_file_stretch_mode = self.settings.static_file_stretch_mode
        
        # Cam input
        self.capture_device = settings.input_updater_capture_device
        self.no_cam_present = (self.capture_device is None)     # Disable all cam functionality
        self.bound_cap_device = None
        self.sleep_after_read_frame = settings.input_updater_sleep_after_read_frame
        self.latest_cam_frame = None
        self.freeze_cam = False

        # Static file input

        # latest image filename selected, used to avoid reloading
        self.latest_static_filename = None

        # latest loaded image frame, holds the pixels and used to force reloading
        self.latest_static_frame = None

        # latest label for loaded image
        self.latest_label = None

        # keeps current index of loaded file, doesn't seem important
        self.static_file_idx = None

        # contains the requested number of increaments for file index
        self.static_file_idx_increment = 0

        self.available_files, self.labels = get_files_list(self.settings)

    def bind_camera(self):
        # Due to OpenCV limitations, this should be called from the main thread
        print 'InputImageFetcher: bind_camera starting'
        if self.no_cam_present:
            print 'InputImageFetcher: skipping camera bind (device: None)'
        else:
            self.bound_cap_device = cv2.VideoCapture(self.capture_device)
            if self.bound_cap_device.isOpened():
                print 'InputImageFetcher: capture device %s is open' % self.capture_device
            else:
                print '\n\nWARNING: InputImageFetcher: capture device %s failed to open! Camera will not be available!\n\n' % self.capture_device
                self.bound_cap_device = None
                self.no_cam_present = True
        print 'InputImageFetcher: bind_camera finished'

    def free_camera(self):
        # Due to OpenCV limitations, this should be called from the main thread
        if self.no_cam_present:
            print 'InputImageFetcher: skipping camera free (device: None)'
        else:
            print 'InputImageFetcher: freeing camera'
            del self.bound_cap_device  # free the camera
            self.bound_cap_device = None
            print 'InputImageFetcher: camera freed'

    def set_mode_static(self):
        with self.lock:
            self.static_file_mode = True
        
    def set_mode_cam(self):
        with self.lock:
            if self.no_cam_present:
                print 'WARNING: ignoring set_mode_cam, no cam present'
            else:
                self.static_file_mode = False
                assert self.bound_cap_device != None, 'Call bind_camera first'
        
    def toggle_input_mode(self):
        with self.lock:
            if self.static_file_mode:
                self.set_mode_cam()
            else:
                self.set_mode_static()
        
    def set_mode_stretch_on(self):
        with self.lock:
            if not self.static_file_stretch_mode:
                self.static_file_stretch_mode = True
                self.latest_static_frame = None   # Force reload
                self.latest_label = None
                #self.latest_frame_is_from_cam = True  # Force reload
        
    def set_mode_stretch_off(self):
        with self.lock:
            if self.static_file_stretch_mode:
                self.static_file_stretch_mode = False
                self.latest_static_frame = None   # Force reload
                self.latest_label = None
                #self.latest_frame_is_from_cam = True  # Force reload
        
    def toggle_stretch_mode(self):
        with self.lock:
            if self.static_file_stretch_mode:
                self.set_mode_stretch_off()
            else:
                self.set_mode_stretch_on()
        
    def run(self):
        while not self.quit and not self.is_timed_out():
            #start_time = time.time()
            if self.static_file_mode:
                self.check_increment_and_load_image()
            else:
                if self.freeze_cam and self.latest_cam_frame is not None:
                    # If static file mode was switched to cam mode but cam is still frozen, we need to push the cam frame again
                    if not self.latest_frame_is_from_cam:

                        # future feature: implement more interesting combination of using a camera in sieamese mode
                        if self.settings.is_siamese:
                            im = (self.latest_cam_frame, self.latest_cam_frame)
                        else:
                            im = self.latest_cam_frame

                        self._increment_and_set_frame(im, True)
                else:
                    frame_full = read_cam_frame(self.bound_cap_device, color=not self.settings._calculated_is_gray_model)
                    #print '====> just read frame', frame_full.shape
                    frame = crop_to_square(frame_full)
                    with self.lock:
                        self.latest_cam_frame = frame

                        if self.settings.is_siamese:
                            im = (self.latest_cam_frame, self.latest_cam_frame)
                        else:
                            im = self.latest_cam_frame
                        self._increment_and_set_frame(im, True)
            
            time.sleep(self.sleep_after_read_frame)
            #print 'Reading one frame took', time.time() - start_time

        print 'InputImageFetcher: exiting run method'
        #print 'InputImageFetcher: read', self.read_frames, 'frames'

    def get_frame(self):
        '''Fetch the latest frame_idx and frame. The idx increments
        any time the frame data changes. If the idx is < 0, the frame
        is not valid.
        '''
        with self.lock:
            return (self.latest_frame_idx, self.latest_frame_data, self.latest_label, self.latest_static_filename)

    def increment_static_file_idx(self, amount = 1):
        with self.lock:
            self.static_file_idx_increment += amount

    def next_image(self):
        if self.static_file_mode:
            self.increment_static_file_idx(1)
        else:
            self.static_file_mode = True

    def prev_image(self):
        if self.static_file_mode:
            self.increment_static_file_idx(-1)
        else:
            self.static_file_mode = True

    def _increment_and_set_frame(self, frame, from_cam):
        assert frame is not None
        with self.lock:
            self.latest_frame_idx += 1
            self.latest_frame_data = frame
            self.latest_frame_is_from_cam = from_cam

    def check_increment_and_load_image(self):
        with self.lock:
            if (self.static_file_idx_increment == 0 and
                self.static_file_idx is not None and
                not self.latest_frame_is_from_cam and
                self.latest_static_frame is not None):
                # Skip if a static frame is already loaded and there is no increment
                return

            assert len(self.available_files) != 0, ('Error: No files found in %s matching %s (current working directory is %s)' %
                                               (self.settings.static_files_dir, self.settings.static_files_regexp, os.getcwd()))
            if self.static_file_idx is None:
                self.static_file_idx = 0
            self.static_file_idx = (self.static_file_idx + self.static_file_idx_increment) % len(self.available_files)
            self.static_file_idx_increment = 0
            if self.latest_static_filename != self.available_files[self.static_file_idx] or self.latest_static_frame is None:
                self.latest_static_filename = self.available_files[self.static_file_idx]

                failed = False
                try:
                    if self.settings.is_siamese:
                        # loading two images for siamese network
                        im1 = caffe.io.load_image(os.path.join(self.settings.static_files_dir, self.latest_static_filename[0]), color=not self.settings._calculated_is_gray_model)
                        im2 = caffe.io.load_image(os.path.join(self.settings.static_files_dir, self.latest_static_filename[1]), color=not self.settings._calculated_is_gray_model)
                        if not self.static_file_stretch_mode:
                            im1 = crop_to_square(im1)
                            im2 = crop_to_square(im2)

                        im = (im1,im2)

                    else:
                        im = caffe.io.load_image(os.path.join(self.settings.static_files_dir, self.latest_static_filename), color=not self.settings._calculated_is_gray_model)
			#im = (im-mean)/std
			#pdb.set_trace()
                        if not self.static_file_stretch_mode:
                            im = crop_to_square(im)
                except Exception as e:
                    failed = True
                    print 'Failed loading data'

                if not failed:
                    self.latest_static_frame = im

                    # if we have labels, keep it
                    if self.labels:
                        self.latest_label = self.labels[self.static_file_idx]

            self._increment_and_set_frame(self.latest_static_frame, False)
