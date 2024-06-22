import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
import base64
import io
import numpy as np
from PIL import Image

from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    "restoration": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
}


# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img


class InferenceArgs:
    def __init__(
        self,
        input_path: str = "input/images",
        output_path: str = None,
        fidelity_weight: float = 0.7,
        upscale: int = 1,
        has_aligned: bool = False,
        only_center_face: bool = False,
        draw_box: bool = False,
        detection_model: str = "retinaface_resnet50",
        bg_upsampler: str = True,
        face_upsample: bool = True,
        bg_tile: int = 400,
        suffix: str = None,
    ):
        # Input image, video or folder. Default: inputs/whole_imgs
        self.input_path: str = input_path
        # Output folder. Default: results/<input_name>_<w>
        self.output_path: str = output_path
        # Balance the quality and fidelity. Default: 0.7
        self.fidelity_weight: float = fidelity_weight
        # The final upsampling scale of the image. Default: 1
        self.upscale: int = upscale
        # Input are cropped and aligned faces. Default: False
        self.has_aligned: bool = has_aligned
        # Only restore the center face. Default: False
        self.only_center_face: bool = only_center_face
        # Draw the bounding box for the detected faces. Default: False
        self.draw_box: bool = draw_box

        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        # Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib.
        # Default: retinaface_resnet50",
        self.detection_model: str = detection_model
        # "Background upsampler. Optional: realesrgan"
        self.bg_upsampler: str = bg_upsampler
        # Face upsampler after enhancement. Default: False
        self.face_upsample: bool = face_upsample
        # Tile size for background sampler. Default: 400
        self.bg_tile: int = bg_tile
        # Suffix of the restored faces. Default: None
        self.suffix: str = suffix


def set_realesrgan(args: InferenceArgs):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available():  # set False in CPU/MPS mode
        no_half_gpu_list = ["1650", "1660"]  # set False for GPUs that don't support f16
        if not True in [
            gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list
        ]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half,
    )

    if not gpu_is_available():  # CPU
        import warnings

        warnings.warn(
            "Running on CPU now! Make sure your PyTorch version matches your CUDA."
            "The unoptimized RealESRGAN is slow on CPU. "
            "If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.",
            category=RuntimeWarning,
        )
    return upsampler


def image_enhance(job: dict):
    job_input = job["input"]
    image_b64 = job_input["image"]
    upscale = job_input.get("upscale", 1)
    bg_enhance = job_input.get("bg_enhance", False)
    face_upsample = job_input.get("face_upsample", False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    args = InferenceArgs(
        upscale=upscale, bg_upsampler=bg_enhance, face_upsample=face_upsample
    )

    # ------------------------ input & output ------------------------
    w = args.fidelity_weight

    # ------------------ set up background upsampler ------------------
    if args.bg_upsampler == "realesrgan":
        bg_upsampler = set_realesrgan(args)
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan(args)
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(
        url=pretrain_model_url["restoration"],
        model_dir="weights/CodeFormer",
        progress=True,
        file_name=None,
    )
    checkpoint = torch.load(ckpt_path)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned:
        print(f"Face detection model: {args.detection_model}")
    if bg_upsampler is not None:
        print(f"Background upsampling: True, Face upsampling: {args.face_upsample}")
    else:
        print(f"Background upsampling: False, Face upsampling: {args.face_upsample}")

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=args.detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )

    # Enhancement process
    # clean all the intermediate results to process the next image
    face_helper.clean_all()

    img: np.ndarray = stringToRGB(base64_string=image_b64)
    if args.has_aligned:
        # the input faces are already cropped and aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.is_gray = is_gray(img, threshold=10)
        if face_helper.is_gray:
            print("Grayscale input: True")
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        # get face landmarks for each face
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5
        )
        print(f"\tdetect {num_det_faces} faces")
        # align and warp each face
        face_helper.align_warp_face()

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f"\tFailed inference for CodeFormer: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face, cropped_face)

    # paste_back
    if not args.has_aligned:
        # upsample the background
        if bg_upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        if args.face_upsample and face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=args.draw_box,
                face_upsampler=face_upsampler,
            )
        else:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=args.draw_box
            )

    final_img_b64 = base64.b64encode(cv2.imencode(".png", restored_img)[1]).decode(
        "utf-8"
    )
    
    return {"enhanced_image": final_img_b64}
