import torch
import pandas as pd

# models provided by guided-diffusion
model_128_path = "128x128_diffusion.pt"
model_512_path = "512x512_diffusion.pt"

# load model
model_128 = torch.load(model_128_path, map_location=torch.device('cpu'))
model_512 = torch.load(model_512_path, map_location=torch.device('cpu'))

# The parameters used for information storage
def extract_model_layers_info(model):
    layer_info = []
    input_height, input_width = 128, 128  # Assume the input size is 128 x 128

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
            filter_height, filter_width = layer.kernel_size
            channels = layer.in_channels
            num_filter = layer.out_channels
            stride_height, stride_width = layer.stride

            # Calculate IFMAP Height and Width
            if isinstance(layer, torch.nn.Conv2d):
                ifmap_height = (input_height - filter_height + 2 * layer.padding[0]) // stride_height + 1
                ifmap_width = (input_width - filter_width + 2 * layer.padding[1]) // stride_width + 1
            elif isinstance(layer, torch.nn.ConvTranspose2d):
                ifmap_height = (input_height - 1) * stride_height - 2 * layer.padding[0] + filter_height
                ifmap_width = (input_width - 1) * stride_width - 2 * layer.padding[1] + filter_width

            layer_info.append([name, input_height, input_width, filter_height, filter_width, channels, num_filter, stride_height])

            # Update next layer size
            input_height, input_width = ifmap_height, ifmap_width

    return layer_info

layer_info_128 = extract_model_layers_info(model_128)
layer_info_512 = extract_model_layers_info(model_512)

# Create dataFrame to display structure information
columns = ["Layer name", "IFMAP Height", "IFMAP Width", "Filter Height", "Filter Width", "Channels", "Num Filter", "Strides"]
df_128 = pd.DataFrame(layer_info_128, columns=columns)
df_512 = pd.DataFrame(layer_info_512, columns=columns)

# Print DataFrame or store to CSV file
print("128x128 Model Structure:")
print(df_128)
df_128.to_excel("GUID128.csv", index=False)

print("\n512x512 Model Structure:")
print(df_512)
df_512.to_excel("GUID512.csv", index=False)
