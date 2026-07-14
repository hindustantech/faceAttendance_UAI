# Vendored (verbatim, trimmed to inference-only functions) from:
#   https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
#   src/utility.py  (Apache-2.0, Minivision)
#
# Only parse_model_name() and get_kernel() are needed for inference.
# The training-only helpers (get_time, make_if_not_exist, get_width_height)
# were dropped since this project only runs inference.


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def parse_model_name(model_name):
    """
    Decode a model filename like '2.7_80x80_MiniFASNetV2.pth' into
    (h_input, w_input, model_type, scale).

    The naming convention is part of the training pipeline's contract:
    <crop_scale>_<h>x<w>_<ModelClass>.pth
    A scale of 'org' means "use the raw detector bbox, no rescale crop".
    """
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale
