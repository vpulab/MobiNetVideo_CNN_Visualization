#! /usr/bin/env python

# this import must comes first to make sure we use the non-display backend
import matplotlib
matplotlib.use('Agg')

# add parent folder to search path, to enable import of core modules like settings
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
#import ipdb as pdb
import pdb
import cPickle as pickle

import settings
from caffevis.caffevis_helper import set_mean
from siamese_helper import SiameseHelper

from jby_misc import WithTimer
from max_tracker import output_max_patches
from find_max_acts import load_max_tracker_from_file
from settings_misc import load_network


def main():
    #pdb.set_trace()
    parser = argparse.ArgumentParser(description='Loads a pickled NetMaxTracker and outputs one or more of {the patches of the image, a deconv patch, a backprop patch} associated with the maxes.')
    parser.add_argument('--N',            type = int, default = 9, help = 'Note and save top N activations.')
    parser.add_argument('--gpu',          action = 'store_true', default=settings.caffevis_mode_gpu, help = 'Use gpu.')
    parser.add_argument('--do-maxes',     action = 'store_true', default=settings.max_tracker_do_maxes, help = 'Output max patches.')
    parser.add_argument('--do-deconv',    action = 'store_true', default=settings.max_tracker_do_deconv, help = 'Output deconv patches.')
    parser.add_argument('--do-deconv-norm', action = 'store_true', default=settings.max_tracker_do_deconv_norm, help = 'Output deconv-norm patches.')
    parser.add_argument('--do-backprop',  action = 'store_true', default=settings.max_tracker_do_backprop, help = 'Output backprop patches.')
    parser.add_argument('--do-backprop-norm', action = 'store_true', default=settings.max_tracker_do_backprop_norm, help = 'Output backprop-norm patches.')
    parser.add_argument('--do-info',      action = 'store_true', default=settings.max_tracker_do_info, help = 'Output info file containing max filenames and labels.')
    parser.add_argument('--idx-begin',    type = int, default = None, help = 'Start at this unit (default: all units).')
    parser.add_argument('--idx-end',      type = int, default = None, help = 'End at this unit (default: all units).')
    
    parser.add_argument('--nmt_pkl',      type = str, default = os.path.join(settings.caffevis_outputs_dir, 'find_max_acts_output.pickled'), help = 'Which pickled NetMaxTracker to load.')
    parser.add_argument('--net_prototxt', type = str, default = settings.caffevis_deploy_prototxt, help = 'network prototxt to load')
    parser.add_argument('--net_weights',  type = str, default = settings.caffevis_network_weights, help = 'network weights to load')
    parser.add_argument('--datadir',      type = str, default = settings.static_files_dir, help = 'directory to look for files in')
    parser.add_argument('--filelist',     type = str, default = settings.static_files_input_file, help = 'List of image files to consider, one per line. Must be the same filelist used to produce the NetMaxTracker!')
    parser.add_argument('--outdir',       type = str, default = settings.caffevis_outputs_dir, help = 'Which output directory to use. Files are output into outdir/layer/unit_%%04d/{maxes,deconv,backprop}_%%03d.png')
    parser.add_argument('--search-min',    action='store_true', default=False, help='Should we also search for minimal activations?')
    args = parser.parse_args()

    settings.caffevis_deploy_prototxt = args.net_prototxt
    settings.caffevis_network_weights = args.net_weights

    net, data_mean = load_network(settings)

    # validate batch size
    if settings.is_siamese and settings._calculated_siamese_network_format == 'siamese_batch_pair':
        # currently, no batch support for siamese_batch_pair networks
        # it can be added by simply handle the batch indexes properly, but it should be thoroughly tested
        assert (settings.max_tracker_batch_size == 1)

    # set network batch size
    current_input_shape = net.blobs[net.inputs[0]].shape
    current_input_shape[0] = settings.max_tracker_batch_size
    net.blobs[net.inputs[0]].reshape(*current_input_shape)
    net.reshape()

    assert args.do_maxes or args.do_deconv or args.do_deconv_norm or args.do_backprop or args.do_backprop_norm or args.do_info, 'Specify at least one do_* option to output.'

    siamese_helper = SiameseHelper(settings.layers_list)

    nmt = load_max_tracker_from_file(args.nmt_pkl)

    for layer_name in settings.layers_to_output_in_offline_scripts:

        print 'Started work on layer %s' % (layer_name)

        normalized_layer_name = siamese_helper.normalize_layer_name_for_max_tracker(layer_name)

        mt = nmt.max_trackers[normalized_layer_name]

        if args.idx_begin is None:
            idx_begin = 0
        if args.idx_end is None:
            idx_end = mt.max_vals.shape[0]

        with WithTimer('Saved %d images per unit for %s units %d:%d.' % (args.N, normalized_layer_name, idx_begin, idx_end)):

            output_max_patches(settings, mt, net, normalized_layer_name, idx_begin, idx_end,
                               args.N, args.datadir, args.filelist, args.outdir, False,
                               (args.do_maxes, args.do_deconv, args.do_deconv_norm, args.do_backprop, args.do_backprop_norm, args.do_info))

            if args.search_min:
                output_max_patches(settings, mt, net, normalized_layer_name, idx_begin, idx_end,
                                   args.N, args.datadir, args.filelist, args.outdir, True,
                                   (args.do_maxes, args.do_deconv, args.do_deconv_norm, args.do_backprop, args.do_backprop_norm, args.do_info))

if __name__ == '__main__':
    main()
