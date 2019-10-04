
# basic network configuration
base_folder = '%DVT_ROOT%/'
caffevis_deploy_prototxt = base_folder + './models/indoor/alexnet_indoor.prototxt'
caffevis_network_weights = base_folder + './models/indoor/alexnet_indoor.caffemodel'

# input images
static_files_dir = base_folder + './input_images'


# UI customization
caffevis_label_layers    = ['fc3', 'prob']
caffevis_labels          = base_folder + './models/alexnet_indoor/categories_indoor.txt'
caffevis_prob_layer      = 'prob'

def caffevis_layer_pretty_name_fn(name):
    return name.replace('pool','p').replace('norm','n')

# offline scripts configuration
caffevis_outputs_dir = base_folder + './models/indoor/outputs'
layers_to_output_in_offline_scripts = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3', 'prob']
