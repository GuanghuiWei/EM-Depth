from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

# import networks
from layers import disp_to_depth
import cv2
import heapq
from networks.depth_hrnet import DepthEncoder
from networks.depth_decoder_My2S import DepthDecoder_My2S
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images')

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        required=True)
    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )
    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="EM-Depth",
                        choices=["EM-Depth"])
    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=192)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args, encoder, depth_decoder, folder_image_path=None, save_disp_path=None):
    """Function to predict for a single image or folder of images
    """
    assert args.load_weights_folder is not None, \
        "You must specify the --load_weights_folder parameter"

    # FINDING INPUT IMAGES
    if folder_image_path is not None:
        paths = [folder_image_path]
        output_directory = save_disp_path

    elif folder_image_path is None and args.image_path is not None:
        paths = [args.image_path]   #glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = os.path.dirname(args.image_path)
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((args.width, args.height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))


    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # LOADING PRETRAINED MODEL
    encoder = DepthEncoder(18, False)
    depth_decoder = DepthDecoder_My2S(encoder.num_ch_enc, scales=range(1))

    print("   Loading pretrained encoder")
    model_dict = encoder.state_dict()
    depth_model_dict = depth_decoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    print("   Loading pretrained decoder")
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    encoder.to(device)
    depth_decoder.to(device)
    encoder.eval()
    depth_decoder.eval()
    test_simple(args, encoder=encoder, depth_decoder=depth_decoder)

    # test more images
    #img_path = "/path/to/your/images/"
    #img_save_path = "/path/to/save/"

    #for filename in os.listdir(img_path):
        #test_simple(args, encoder, depth_decoder, img_path+filename, img_save_path)