import os
import time
from threading import Lock
from siamese_helper import SiameseViewMode, SiameseHelper
from caffe_misc import layer_name_to_top_name
from image_misc import get_tiles_height_width_ratio, gray_to_colormap
import pdb

class PatternMode:
    OFF = 0
    MAXIMAL_OPTIMIZED_IMAGE = 1
    MAXIMAL_INPUT_IMAGE = 2
    WEIGHTS_HISTOGRAM = 3
    MAX_ACTIVATIONS_HISTOGRAM = 4
    WEIGHTS_CORRELATION = 5
    ACTIVATIONS_CORRELATION = 6
    NUMBER_OF_MODES = 7

class BackpropMode:
    OFF = 0
    GRAD = 1
    DECONV_ZF = 2
    DECONV_GB = 3
    NUMBER_OF_MODES = 4

    @staticmethod
    def to_string(back_mode):
        if back_mode == BackpropMode.OFF:
            return 'off'
        elif back_mode == BackpropMode.GRAD:
            return 'grad'
        elif back_mode == BackpropMode.DECONV_ZF:
            return 'deconv zf'
        elif back_mode == BackpropMode.DECONV_GB:
            return 'deconv gb'

        return 'n/a'

class BackpropViewOption:
    RAW = 0
    GRAY = 1
    NORM = 2
    NORM_BLUR = 3
    POS_SUM = 4
    HISTOGRAM = 5
    NUMBER_OF_OPTIONS = 6

    @staticmethod
    def to_string(back_view_option):
        if back_view_option == BackpropViewOption.RAW:
            return 'raw'
        elif back_view_option == BackpropViewOption.GRAY:
            return 'gray'
        elif back_view_option == BackpropViewOption.NORM:
            return 'norm'
        elif back_view_option == BackpropViewOption.NORM_BLUR:
            return 'normblur'
        elif back_view_option == BackpropViewOption.POS_SUM:
            return 'sum>0'
        elif back_view_option == BackpropViewOption.HISTOGRAM:
            return 'histogram'

        return 'n/a'

class ColorMapOption:
    GRAY = 0
    JET = 1
    PLASMA = 2
    NUMBER_OF_OPTIONS = 3

    @staticmethod
    def to_string(color_map_option):
        if color_map_option == ColorMapOption.GRAY:
            return 'gray'
        elif color_map_option == ColorMapOption.JET:
            return 'jet'
        elif color_map_option == ColorMapOption.PLASMA:
            return 'plasma'

        return 'n/a'

class InputOverlayOption:
    OFF = 0
    OVER_ACTIVE = 1
    OVER_INACTIVE = 2
    NUMBER_OF_OPTIONS = 3

    @staticmethod
    def to_string(input_overlay_option):
        if input_overlay_option == InputOverlayOption.OFF:
            return 'off'
        elif input_overlay_option == InputOverlayOption.OVER_ACTIVE:
            return 'over active'
        elif input_overlay_option == InputOverlayOption.OVER_INACTIVE:
            return 'over inactive'

        return 'n/a'

class CaffeVisAppState(object):
    '''State of CaffeVis app.'''

    def __init__(self, net, settings, bindings, live_vis):
        self.lock = Lock()  # State is accessed in multiple threads
        self.settings = settings
        self.bindings = bindings
        self.net = net
        self.live_vis = live_vis

        self.fill_layers_list(net)

        self.siamese_helper = SiameseHelper(settings.layers_list)

        self._populate_net_blob_info(net)

        self.layer_boost_indiv_choices = self.settings.caffevis_boost_indiv_choices   # 0-1, 0 is noop
        self.layer_boost_gamma_choices = self.settings.caffevis_boost_gamma_choices   # 0-inf, 1 is noop
        self.caffe_net_state = 'free'     # 'free', 'proc', or 'draw'
        self.extra_msg = ''
        self.back_stale = True       # back becomes stale whenever the last back diffs were not computed using the current backprop unit and method (bprop or deconv)
        self.next_frame = None
        self.next_label = None
        self.next_filename = None
        self.last_frame = None
        self.jpgvis_to_load_key = None
        self.last_key_at = 0
        self.quit = False

        self._reset_user_state()

    def _populate_net_blob_info(self, net):
        '''For each blob, save the number of filters and precompute
        tile arrangement (needed by CaffeVisAppState to handle keyboard navigation).
        '''
        self.net_blob_info = {}
        for key in net.blobs.keys():
            self.net_blob_info[key] = {}
            # Conv example: (1, 96, 55, 55)
            # FC example: (1, 1000)
            blob_shape = net.blobs[key].data.shape

            # handle case when output is a single number per image in the batch
            if (len(blob_shape) == 1):
                blob_shape = (blob_shape[0], 1)

            self.net_blob_info[key]['isconv'] = (len(blob_shape) == 4)
            self.net_blob_info[key]['data_shape'] = blob_shape[1:]  # Chop off batch size
            self.net_blob_info[key]['n_tiles'] = blob_shape[1]
            self.net_blob_info[key]['tiles_rc'] = get_tiles_height_width_ratio(blob_shape[1], self.settings.caffevis_layers_aspect_ratio)
            self.net_blob_info[key]['tile_rows'] = self.net_blob_info[key]['tiles_rc'][0]
            self.net_blob_info[key]['tile_cols'] = self.net_blob_info[key]['tiles_rc'][1]

    def get_headers(self):

        headers = list()
        for layer_def in self.settings.layers_list:
            headers.append(SiameseHelper.get_header_from_layer_def(layer_def))

        return headers

    def _reset_user_state(self):
        self.show_maximal_score = True
        self.input_overlay_option = InputOverlayOption.OFF
        self.layer_idx = 0
        self.layer_boost_indiv_idx = self.settings.caffevis_boost_indiv_default_idx
        self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
        self.layer_boost_gamma_idx = self.settings.caffevis_boost_gamma_default_idx
        self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
        self.cursor_area = 'top'   # 'top' or 'bottom'
        self.selected_unit = 0
        self.siamese_view_mode = SiameseViewMode.BOTH_IMAGES

        # Which layer and unit (or channel) to use for backprop
        self.backprop_layer_idx = self.layer_idx
        self.backprop_unit = self.selected_unit
        self.backprop_selection_frozen = False    # If false, backprop unit tracks selected unit
        self.backprop_siamese_view_mode = SiameseViewMode.BOTH_IMAGES
        self.back_enabled = False
        self.back_mode = BackpropMode.OFF
        self.back_view_option = BackpropViewOption.RAW
        self.color_map_option = ColorMapOption.JET
        self.pattern_mode = PatternMode.OFF    # type of patterns to show instead of activations in layers pane: maximal optimized image, maximal input image, maximal histogram, off
        self.pattern_first_only = True         # should we load only the first pattern image for each neuron, or all the relevant images per neuron
        self.layers_pane_zoom_mode = 0       # 0: off, 1: zoom selected (and show pref in small pane), 2: zoom backprop
        self.layers_show_back = False   # False: show forward activations. True: show backward diffs
        self.show_label_predictions = self.settings.caffevis_init_show_label_predictions
        self.show_unit_jpgs = self.settings.caffevis_init_show_unit_jpgs
        self.drawing_stale = True
        kh,_ = self.bindings.get_key_help('help_mode')
        self.extra_msg = '%s for help' % kh[0]

    def handle_key(self, key):
        #print 'Ignoring key:', key
        if key == -1:
            return key

        with self.lock:
            key_handled = True
            self.last_key_at = time.time()
            tag = self.bindings.get_tag(key)
            if tag == 'reset_state':
                self._reset_user_state()
            elif tag == 'sel_layer_left':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = max(0, self.layer_idx - 1)

            elif tag == 'sel_layer_right':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = min(len(self.settings.layers_list) - 1, self.layer_idx + 1)

            elif tag == 'sel_left':
                self.move_selection('left')
            elif tag == 'sel_right':
                self.move_selection('right')
            elif tag == 'sel_down':
                self.move_selection('down')
            elif tag == 'sel_up':
                self.move_selection('up')

            elif tag == 'sel_left_fast':
                self.move_selection('left', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_right_fast':
                self.move_selection('right', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_down_fast':
                self.move_selection('down', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_up_fast':
                self.move_selection('up', self.settings.caffevis_fast_move_dist)

            elif tag == 'boost_individual':
                self.layer_boost_indiv_idx = (self.layer_boost_indiv_idx + 1) % len(self.layer_boost_indiv_choices)
                self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
            elif tag == 'boost_gamma':
                self.layer_boost_gamma_idx = (self.layer_boost_gamma_idx + 1) % len(self.layer_boost_gamma_choices)
                self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
            elif tag == 'next_pattern_mode':
                self.set_pattern_mode((self.pattern_mode + 1) % PatternMode.NUMBER_OF_MODES)

            elif tag == 'prev_pattern_mode':
                self.set_pattern_mode((self.pattern_mode - 1 + PatternMode.NUMBER_OF_MODES) % PatternMode.NUMBER_OF_MODES)

            elif tag == 'pattern_first_only':
                self.pattern_first_only = not self.pattern_first_only

            elif tag == 'show_back':
                self.set_show_back(not self.layers_show_back)

            elif tag == 'next_ez_back_mode_loop':
                self.set_back_mode((self.back_mode + 1) % BackpropMode.NUMBER_OF_MODES)

            elif tag == 'prev_ez_back_mode_loop':
                self.set_back_mode((self.back_mode - 1 + BackpropMode.NUMBER_OF_MODES) % BackpropMode.NUMBER_OF_MODES)

            elif tag == 'next_back_view_option':
                self.set_back_view_option((self.back_view_option + 1) % BackpropViewOption.NUMBER_OF_OPTIONS)

            elif tag == 'prev_back_view_option':
                self.set_back_view_option((self.back_view_option - 1 + BackpropViewOption.NUMBER_OF_OPTIONS) % BackpropViewOption.NUMBER_OF_OPTIONS)

            elif tag == 'next_color_map':
                self.color_map_option = (self.color_map_option  + 1) % ColorMapOption.NUMBER_OF_OPTIONS

            elif tag == 'prev_color_map':
                self.color_map_option  = (self.color_map_option  - 1 + ColorMapOption.NUMBER_OF_OPTIONS) % ColorMapOption.NUMBER_OF_OPTIONS

            elif tag == 'freeze_back_unit':
                self.toggle_freeze_back_unit()

            elif tag == 'zoom_mode':
                self.layers_pane_zoom_mode = (self.layers_pane_zoom_mode + 1) % 3
                if self.layers_pane_zoom_mode == 2 and not self.back_enabled:
                    # Skip zoom into backprop pane when backprop is off
                    self.layers_pane_zoom_mode = 0

            elif tag == 'toggle_label_predictions':
                self.show_label_predictions = not self.show_label_predictions

            elif tag == 'toggle_unit_jpgs':
                self.show_unit_jpgs = not self.show_unit_jpgs

            elif tag == 'siamese_view_mode':
                self.siamese_view_mode = (self.siamese_view_mode + 1) % SiameseViewMode.NUMBER_OF_MODES

            elif tag == 'toggle_maximal_score':
                self.show_maximal_score = not self.show_maximal_score

            elif tag == 'next_input_overlay':
                self.set_input_overlay((self.input_overlay_option + 1) % InputOverlayOption.NUMBER_OF_OPTIONS)

            elif tag == 'prev_input_overlay':
                self.set_input_overlay((self.input_overlay_option  - 1 + InputOverlayOption.NUMBER_OF_OPTIONS) % InputOverlayOption.NUMBER_OF_OPTIONS)

            else:
                key_handled = False

            self._ensure_valid_selected()

            self.drawing_stale = key_handled   # Request redraw any time we handled the key

        return (None if key_handled else key)

    def handle_mouse_left_click(self, x, y, flags, param, panes, header_boxes, buttons_boxes):

        for pane_name, pane in panes.items():
            if pane.j_begin <= x < pane.j_end and pane.i_begin <= y < pane.i_end:

                if pane_name == 'caffevis_control': # layers list

                    # search for layer clicked on
                    for box_idx, box in enumerate(header_boxes):
                        start_x, end_x, start_y, end_y, text = box
                        if start_x <= x - pane.j_begin < end_x and start_y <= y - pane.i_begin <= end_y:
                            # print 'layers list clicked on layer %d (%s,%s)' % (box_idx, x, y)
                            self.layer_idx = box_idx
                            self.cursor_area = 'top'
                            self._ensure_valid_selected()
                            self.drawing_stale = True  # Request redraw any time we handled the mouse
                            return
                    # print 'layers list clicked on (%s,%s)' % (x, y)

                elif pane_name == 'caffevis_layers': # channels list
                    # print 'channels list clicked on (%s,%s)' % (x, y)

                    default_layer_name = self.get_default_layer_name()
                    default_top_name = layer_name_to_top_name(self.net, default_layer_name)

                    tile_rows, tile_cols = self.net_blob_info[default_top_name]['tiles_rc']

                    dy_per_channel = (pane.data.shape[0] + 1) / float(tile_rows)
                    dx_per_channel = (pane.data.shape[1] + 1) / float(tile_cols)

                    tile_x = int(((x - pane.j_begin) / dx_per_channel) + 1)
                    tile_y = int(((y - pane.i_begin) / dy_per_channel) + 1)

                    channel_id = (tile_y-1) * tile_cols + (tile_x - 1)

                    self.selected_unit = channel_id
                    self.cursor_area = 'bottom'

                    self.validate_state_for_summary_only_patterns()
                    self._ensure_valid_selected()
                    self.drawing_stale = True  # Request redraw any time we handled the mouse
                    return

                elif pane_name == 'caffevis_buttons':
                    # print 'buttons!'

                    # search for layer clicked on
                    for box_idx, box in enumerate(buttons_boxes):
                        start_x, end_x, start_y, end_y, text = box
                        if start_x <= x - pane.j_begin < end_x and start_y <= y - pane.i_begin <= end_y:
                            # print 'DEBUG: pressed %s' % text

                            if text == 'File':
                                self.live_vis.input_updater.set_mode_static()
                                pass
                            elif text == 'Prev':
                                self.live_vis.input_updater.prev_image()
                                pass
                            elif text == 'Next':
                                self.live_vis.input_updater.next_image()
                                pass
                            elif text == 'Camera':
                                self.live_vis.input_updater.set_mode_cam()
                                pass

                            elif text == 'Modes':
                                pass
                            elif text == 'Activations':
                                self.set_show_back(False)
                                pass
                            elif text == 'Gradients':
                                self.set_show_back(True)
                                pass
                            elif text == 'Maximal Optimized':
                                with self.lock:
                                    self.set_pattern_mode(PatternMode.MAXIMAL_OPTIMIZED_IMAGE)
                                pass
                            elif text == 'Maximal Input':
                                with self.lock:
                                    self.set_pattern_mode(PatternMode.MAXIMAL_INPUT_IMAGE)
                                pass
                            elif text == 'Weights Histogram':
                                with self.lock:
                                    self.set_pattern_mode(PatternMode.WEIGHTS_HISTOGRAM)
                                pass
                            elif text == 'Activations Histogram':
                                with self.lock:
                                    self.set_pattern_mode(PatternMode.MAX_ACTIVATIONS_HISTOGRAM)
                                pass
                            elif text == 'Weights Correlation':
                                with self.lock:
                                    self.set_pattern_mode(PatternMode.WEIGHTS_CORRELATION)
                                pass
                            elif text == 'Activations Correlation':
                                with self.lock:
                                    self.set_pattern_mode(PatternMode.ACTIVATIONS_CORRELATION)
                                pass

                            elif text == 'Input Overlay':
                                pass
                            elif text == 'No Overlay':
                                self.set_input_overlay(InputOverlayOption.OFF)
                                pass
                            elif text == 'Over Active':
                                self.set_input_overlay(InputOverlayOption.OVER_ACTIVE)
                                pass
                            elif text == 'Over Inactive':
                                self.set_input_overlay(InputOverlayOption.OVER_INACTIVE)
                                pass

                            elif text == 'Backprop Modes':
                                pass
                            elif text == 'No Backprop':
                                self.set_back_mode(BackpropMode.OFF)
                                pass
                            elif text == 'Gradient':
                                self.set_back_mode(BackpropMode.GRAD)
                                pass
                            elif text == 'ZF Deconv':
                                self.set_back_mode(BackpropMode.DECONV_ZF)
                                pass
                            elif text == 'Guided Backprop':
                                self.set_back_mode(BackpropMode.DECONV_GB)
                                pass
                            elif text == 'Freeze Origin':
                                self.toggle_freeze_back_unit()
                                pass

                            elif text == 'Backprop Views':
                                pass
                            elif text == 'Raw':
                                self.set_back_view_option(BackpropViewOption.RAW)
                                pass
                            elif text == 'Gray':
                                self.set_back_view_option(BackpropViewOption.GRAY)
                                pass
                            elif text == 'Norm':
                                self.set_back_view_option(BackpropViewOption.NORM)
                                pass
                            elif text == 'Blurred Norm':
                                self.set_back_view_option(BackpropViewOption.NORM_BLUR)
                                pass
                            elif text == 'Sum > 0':
                                self.set_back_view_option(BackpropViewOption.POS_SUM)
                                pass
                            elif text == 'Gradient Histogram':
                                self.set_back_view_option(BackpropViewOption.HISTOGRAM)
                                pass

                            elif text == 'Help':
                                self.live_vis.toggle_help_mode()
                                pass

                            elif text == 'Quit':
                                self.live_vis.set_quit_flag()
                                pass

                            self._ensure_valid_selected()
                            self.drawing_stale = True
                            return

                else:
                    # print "Clicked: %s - %s" % (x, y)
                    pass
                break

        pass

    def redraw_needed(self):
        with self.lock:
            return self.drawing_stale

    def get_current_layer_definition(self):
        return self.settings.layers_list[self.layer_idx]

    def get_current_backprop_layer_definition(self):
        return self.settings.layers_list[self.backprop_layer_idx]

    def get_single_selected_data_blob(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_single_selected_data_blob(net, layer_def, self.siamese_view_mode)

    def get_single_selected_diff_blob(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_single_selected_diff_blob(net, layer_def, self.siamese_view_mode)

    def get_siamese_selected_data_blobs(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_siamese_selected_data_blobs(net, layer_def, self.siamese_view_mode)

    def get_siamese_selected_diff_blobs(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_siamese_selected_diff_blobs(net, layer_def, self.siamese_view_mode)


    def backward_from_layer(self, net, backprop_layer_def, backprop_unit):

        try:
            return SiameseHelper.backward_from_layer(net, backprop_layer_def, backprop_unit, self.siamese_view_mode)
        except AttributeError:
            print 'ERROR: required bindings (backward_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
            raise
        except ValueError:
            print "ERROR: probably impossible to backprop layer %s, ignoring to avoid crash" % (str(backprop_layer_def['name/s']))
            with self.lock:
                self.back_enabled = False

    def deconv_from_layer(self, net, backprop_layer_def, backprop_unit, deconv_type):

        try:
            return SiameseHelper.deconv_from_layer(net, backprop_layer_def, backprop_unit, self.siamese_view_mode, deconv_type)
        except AttributeError:
            print 'ERROR: required bindings (deconv_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
            raise
        except ValueError:
            print "ERROR: probably impossible to deconv layer %s, ignoring to avoid crash" % (str(backprop_layer_def['name/s']))
            with self.lock:
                self.back_enabled = False

    def get_default_layer_name(self, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_default_layer_name(layer_def)

    def siamese_view_mode_has_two_images(self, layer_def = None):
        '''
        helper function which checks whether the input mode is two images, and the provided layer contains two layer names
        :param layer: can be a single string layer name, or a pair of layer names
        :return: True if both the input mode is BOTH_IMAGES and layer contains two layer names, False oherwise
        '''
        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return SiameseHelper.siamese_view_mode_has_two_images(layer_def, self.siamese_view_mode)

    def move_selection(self, direction, dist = 1):

        default_layer_name = self.get_default_layer_name()
        default_top_name = layer_name_to_top_name(self.net, default_layer_name)

        if direction == 'left':
            if self.cursor_area == 'top':
                self.layer_idx = max(0, self.layer_idx - dist)
            else:
                self.selected_unit -= dist
        elif direction == 'right':
            if self.cursor_area == 'top':
                self.layer_idx = min(len(self.settings.layers_list) - 1, self.layer_idx + dist)
            else:
                self.selected_unit += dist
        elif direction == 'down':
            if self.cursor_area == 'top':
                self.cursor_area = 'bottom'
            else:
                self.selected_unit += self.net_blob_info[default_top_name]['tile_cols'] * dist
        elif direction == 'up':
            if self.cursor_area == 'top':
                pass
            else:
                self.selected_unit -= self.net_blob_info[default_top_name]['tile_cols'] * dist
                if self.selected_unit < 0:
                    self.selected_unit += self.net_blob_info[default_top_name]['tile_cols']
                    self.cursor_area = 'top'

        self.validate_state_for_summary_only_patterns()

    def _ensure_valid_selected(self):

        default_layer_name = self.get_default_layer_name()
        default_top_name = layer_name_to_top_name(self.net, default_layer_name)

        n_tiles = self.net_blob_info[default_top_name]['n_tiles']

        # Forward selection
        self.selected_unit = max(0, self.selected_unit)
        self.selected_unit = min(n_tiles-1, self.selected_unit)

        # Backward selection
        if not self.backprop_selection_frozen:
            # If backprop_selection is not frozen, backprop layer/unit follows selected unit
            if not (self.backprop_layer_idx == self.layer_idx and self.backprop_unit == self.selected_unit and self.backprop_siamese_view_mode == self.siamese_view_mode):
                self.backprop_layer_idx = self.layer_idx
                self.backprop_unit = self.selected_unit
                self.backprop_siamese_view_mode = self.siamese_view_mode
                self.back_stale = True    # If there is any change, back diffs are now stale

    def fill_layers_list(self, net):

        # if layers list is empty, fill it with layer names
        if not self.settings.layers_list:

            # go over layers
            self.settings.layers_list = []
            for layer_name in list(net._layer_names):

                # skip inplace layers
                if len(net.top_names[layer_name]) == 1 and len(net.bottom_names[layer_name]) == 1 and net.top_names[layer_name][0] == net.bottom_names[layer_name][0]:
                    continue

                self.settings.layers_list.append( {'format': 'normal', 'name/s': layer_name} )

        # filter layers if needed
        if hasattr(self.settings, 'caffevis_filter_layers'):
            for layer_def in self.settings.layers_list:
                if self.settings.caffevis_filter_layers(layer_def['name/s']):
                    print '  Layer filtered out by caffevis_filter_layers: %s' % str(layer_def['name/s'])
            self.settings.layers_list = filter(lambda layer_def: not self.settings.caffevis_filter_layers(layer_def['name/s']), self.settings.layers_list)


        pass

    def gray_to_colormap(self, gray_image):

        if self.color_map_option == ColorMapOption.GRAY:
            return gray_image

        elif self.color_map_option == ColorMapOption.JET:
            return gray_to_colormap('jet', gray_image)

        elif self.color_map_option == ColorMapOption.PLASMA:
            return gray_to_colormap('plasma', gray_image)

    def convert_image_pair_to_network_input_format(self, settings, frame_pair, resize_shape):
        return SiameseHelper.convert_image_pair_to_network_input_format(frame_pair, resize_shape, settings.siamese_input_mode)

    def get_layer_output_size(self, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return SiameseHelper.get_layer_output_size(self.net, self.settings.is_siamese, layer_def, self.siamese_view_mode)

    def get_layer_output_size_string(self, layer_def=None):

        layer_output_size = self.get_layer_output_size(layer_def)

        if len(layer_output_size) == 1:
            return '(%d)' % (layer_output_size[0])
        elif len(layer_output_size) == 2:
            return '(%d,%d)' % (layer_output_size[0],layer_output_size[1])
        elif len(layer_output_size) == 3:
            return '(%d,%d,%d)' % (layer_output_size[0], layer_output_size[1], layer_output_size[2])
        else:
            return str(layer_output_size)

    def validate_state_for_summary_only_patterns(self):
        if self.pattern_mode in [PatternMode.ACTIVATIONS_CORRELATION, PatternMode.WEIGHTS_CORRELATION]:
            self.cursor_area = 'top'

    def set_pattern_mode(self, pattern_mode):
        self.pattern_mode = pattern_mode
        self.validate_state_for_summary_only_patterns()

    def set_show_back(self, show_back):
        # If in pattern mode: switch to fwd/back. Else toggle fwd/back mode
        if self.pattern_mode != PatternMode.OFF:
            self.set_pattern_mode(PatternMode.OFF)

        self.layers_show_back = show_back
        if self.layers_show_back:
            if not self.back_enabled:
                if self.back_mode == BackpropMode.OFF:
                    self.back_mode = BackpropMode.GRAD
                self.back_enabled = True
                self.back_stale = True

    def set_input_overlay(self, input_overlay):
        self.input_overlay_option = input_overlay

    def set_back_mode(self, back_mode):
        self.back_mode = back_mode
        self.back_enabled = (self.back_mode != BackpropMode.OFF)
        self.back_stale = True

    def toggle_freeze_back_unit(self):
        # Freeze selected layer/unit as backprop unit
        self.backprop_selection_frozen = not self.backprop_selection_frozen
        if self.backprop_selection_frozen:
            # Grap layer/selected_unit upon transition from non-frozen -> frozen
            self.backprop_layer_idx = self.layer_idx
            self.backprop_unit = self.selected_unit
            self.backprop_siamese_view_mode = self.siamese_view_mode

    def set_back_view_option(self, back_view_option):
        self.back_view_option = back_view_option
