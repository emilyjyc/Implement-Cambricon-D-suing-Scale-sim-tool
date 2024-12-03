import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from scalesim.scale_config import scale_config 
from scalesim.topology_utils import topologies
from scalesim.simulator import simulator as sim

class CambriconDConvReLU(nn.Module):
    def __init__(self,
                 save_disk_space=False,
                 verbose=True,
                 config='',
                 topology='',
                 input_type_gemm=False):
        
        # super(CambriconDConvReLU, self).__init__()

        # Data structures
        self.config = scale_config()
        self.topo = topologies()

        # File paths
        self.config_file = ''
        self.topology_file = ''
        
        # Define only convolution and ReLU layers from the original architecture
        # Assume some example convolution layers (the actual configuration may vary)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # ReLU activation
        self.relu = nn.ReLU()

        # Sign bit tensor storage for each convolution layer
        self.sign_bits = {}

        # PE array parameters for handling deltas and outliers
        self.pe_array = {
            'conv1': {'bitwidth': 8, 'outlier_count': 0},
            'conv2': {'bitwidth': 8, 'outlier_count': 0},
            'conv3': {'bitwidth': 8, 'outlier_count': 0},
        }

        # Member objects
        #self.runner = r.run_nets()
        self.runner = sim()

        # Flags
        self.read_gemm_inputs = input_type_gemm
        self.save_space = save_disk_space
        self.verbose_flag = verbose
        self.run_done_flag = False
        self.logs_generated_flag = False

        self.set_params(config_filename=config, topology_filename=topology)

    def forward(self, x):
        # Forward pass with only Conv and ReLU layers, maintaining sign bits and handling PE array
        x, self.sign_bits['conv1'], self.pe_array['conv1'] = self.process_layer(x, self.conv1, 'conv1')
        x, self.sign_bits['conv2'], self.pe_array['conv2'] = self.process_layer(x, self.conv2, 'conv2')
        x, self.sign_bits['conv3'], self.pe_array['conv3'] = self.process_layer(x, self.conv3, 'conv3')
        return x
    
    def process_layer(self, x, conv_layer, layer_name):
        # Perform convolution
        x = conv_layer(x)
        
        # Maintain sign bit tensor
        sign_bits = torch.sign(x).clamp(min=0).byte()
        
        # Handle PE array deltas and outliers
        delta = x - torch.round(x)  # Example delta calculation
        outlier_mask = delta.abs() > 0.1  # Identify outliers (threshold is an example)
        outlier_count = outlier_mask.sum().item()
        
        # Update PE array info
        pe_info = {
            'bitwidth': 8,  # Assume an 8-bit representation
            'outlier_count': outlier_count
        }
        
        # Apply ReLU
        x = self.relu(x)
        return x, sign_bits, pe_info

def set_params(self,
                   config_filename='',
                   topology_filename='' ):
        # First check if the user provided a valid topology file
        if not topology_filename == '':
            if not os.path.exists(topology_filename):
                print("ERROR: scalesim.scale.py: Topology file not found")
                print("Input file:" + topology_filename)
                print('Exiting')
                exit()
            else:
                self.topology_file = topology_filename

        if not os.path.exists(config_filename):
            print("ERROR: scalesim.scale.py: Config file not found") 
            print("Input file:" + config_filename)
            print('Exiting')
            exit()
        else: 
            self.config_file = config_filename

        # Parse config first
        self.config.read_conf_file(self.config_file)

        # Take the CLI topology over the one in config
        # If topology is not passed from CLI take the one from config
        if self.topology_file == '':
            self.topology_file = self.config.get_topology_path()
        else:
            self.config.set_topology_file(self.topology_file)

        # Parse the topology
        self.topo.load_arrays(topofile=self.topology_file, mnk_inputs=self.read_gemm_inputs)

        #num_layers = self.topo.get_num_layers()
        #self.config.scale_memory_maps(num_layers=num_layers)


# Example usage
def run_cambricon_d_conv_relu():
    # Create the model
    model = CambriconDConvReLU()
    
    # Generate some random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input size
    
    # Run the model
    output = model(input_tensor)
    
    # Print the output shape
    print("Output shape:", output.shape)
    
    # Print sign bit information and PE array details
    for layer, sign_bits in model.sign_bits.items():
        print(f"Sign bits for {layer}: {sign_bits.shape}")
    for layer, pe_info in model.pe_array.items():
        print(f"PE array info for {layer}: {pe_info}")


def run_scale(self, top_path='.'):

        self.top_path = top_path
        save_trace = not self.save_space
        self.runner.set_params(
            config_obj=self.config,
            topo_obj=self.topo,
            top_path=self.top_path,
            verbosity=self.verbose_flag,
            save_trace=save_trace
        )
        self.run_once()

def run_once(self):

        if self.verbose_flag:
            self.print_run_configs()

        #save_trace = not self.save_space

        # TODO: Anand
        # TODO: This release
        # TODO: Call the class member functions
        #self.runner.run_net(
        #    config=self.config,
        #    topo=self.topo,
        #    top_path=self.top_path,
        #    save_trace=save_trace,
        #    verbosity=self.verbose_flag
        #)
        self.runner.run()
        self.run_done_flag = True

        #self.runner.generate_all_logs()
        self.logs_generated_flag = True

        if self.verbose_flag:
            print("************ SCALE SIM Run Complete ****************")

    #
def print_run_configs(self):
    df_string = "Output Stationary"
    df = self.config.get_dataflow()

    if df == 'ws':
        df_string = "Weight Stationary"
    elif df == 'is':
        df_string = "Input Stationary"

    print("====================================================")
    print("******************* SCALE SIM **********************")
    print("====================================================")

    arr_h, arr_w = self.config.get_array_dims()
    print("Array Size: \t" + str(arr_h) + "x" + str(arr_w))

    ifmap_kb, filter_kb, ofmap_kb = self.config.get_mem_sizes()
    print("SRAM IFMAP (kB): \t" + str(ifmap_kb))
    print("SRAM Filter (kB): \t" + str(filter_kb))
    print("SRAM OFMAP (kB): \t" + str(ofmap_kb))
    print("Dataflow: \t" + df_string)
    print("CSV file path: \t" + self.config.get_topology_path())

    if self.config.use_user_dram_bandwidth():
        print("Bandwidth: \t" + self.config.get_bandwidths_as_string())
        print('Working in USE USER BANDWIDTH mode.')
    else:
        print('Working in ESTIMATE BANDWIDTH mode.')

    print("====================================================")


def get_total_cycles(self):
    me = 'scale.' + 'get_total_cycles()'
    if not self.run_done_flag:
        message = 'ERROR: ' + me
        message += ' : Cannot determine cycles. Run the simulation first'
        print(message)
        return

    return self.runner.get_total_cycles()



