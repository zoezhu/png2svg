"""
main

@author zz
@date 2023.11.9
"""

import argparse
from utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SVG_TEMP_FOLDER = os.path.join(BASE_DIR, "svg_tmp")
if not os.path.exists(SVG_TEMP_FOLDER):
    os.makedirs(SVG_TEMP_FOLDER)
SR_MODEL_PATH = os.path.join(BASE_DIR, 'Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth')  # 'Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth'

sys.path.append(os.path.join(BASE_DIR, "Real-ESRGAN"))
try:
    from realesrgan import RealESRGANer
except:
    print("[WARNING] realesrgan not import correctly!!! Make sure install it if you need to do sr!!!")


if __name__ == '__main__':
    # 获取参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str, help="Input image folder or single image path.")
    parser.add_argument("-c", '--color', help="How many colors you want to draw", type=int, default=-1)
    parser.add_argument("-sr", '--do_sr', action='store_true', help="Wheather do super resolution for input image.")
    args = parser.parse_args()
    
    if args.do_sr:
        # 初始化sr模型
        gpu_id = "0" if torch.cuda.is_available() else "cpu"
        if os.path.basename(SR_MODEL_PATH)=="RealESRGAN_x4plus.pth":
            scale = 4
            srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        elif os.path.basename(SR_MODEL_PATH)=="RealESRGAN_x2plus.pth":
            scale = 2
            srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        elif os.path.basename(SR_MODEL_PATH)=="RealESRGAN_x4plus_anime_6B.pth":
            scale = 4
            srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        
        upsampler = RealESRGANer(
            scale=scale,  # 倍数
            model_path=SR_MODEL_PATH,
            dni_weight=None,
            model=srmodel,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True)

    for_debug = False
    if os.path.isdir(args.file):
        img_list = glob(os.path.join(args.file, "*.jpg"))+glob(os.path.join(args.file, "*.png"))
    else:
        img_list = [args.file]
    sum_time = 0
    count = 0
    for filename in tqdm(img_list):
        tic = time.time()
        print("process: ", filename)
        # 定义路径
        this_req_folder = os.path.join(SVG_TEMP_FOLDER, os.path.basename(filename).split(".")[0])
        if not os.path.exists(this_req_folder):
            os.makedirs(this_req_folder)
        out_path = ".".join(filename.split(".")[:-1]) + ".svg"
        
        # sr
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        # h,w,_= img.shape
        if img.shape[2] == 4:  # 如果是四通道就换成白底
            background = Image.new('RGBA', img.size, (255, 255, 255))
            img = Image.alpha_composite(background, img)
            img = img[...,::-1]
        if args.do_sr:
            img, _ = upsampler.enhance(img, outscale=scale)
        h,w,_= img.shape
        if for_debug:
            print("img.shape: ", img.shape)
        # h = h//2
        # w = w//2
        # img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
        if for_debug:
            cv2.imwrite("test_sr.png", img)
        # svg
        draw_svg(img, w, h, this_req_folder, out_path, args.color)
        shutil.rmtree(this_req_folder)
        
        sum_time += time.time()-tic
        count += 1

print("Avg process time:", sum_time/count)