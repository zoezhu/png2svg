"""
functions needed for convert

@author zz
@date 2023.11.9
"""

import os, sys
import shutil
import time
import math
from glob import glob
from tqdm import tqdm
import cv2
from skimage import measure, color
from PIL import Image
import numpy as np
from xml.dom import minidom
import xml.etree.ElementTree as ET
try:
    import torch
except:
    print("[WARNING] torch not import correctly!!! Make sure install it if you need to do sr!!!")
from basicsr.archs.rrdbnet_arch import RRDBNet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

for_debug = False


def rgb_to_hex(rgb_color):
    rgb_color = np.uint8(rgb_color)
    r,g,b = rgb_color
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def do_sr(img):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to("cuda")

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output = np.uint8(output)
    print("output.shape: ", output.shape)
    cv2.imwrite("test_sr.png", output)
    return output
    

def get_dominate_colors(img, w, h, palette_size=32, dis_thresh=60):
    global for_debug
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((w,h), resample=Image.LANCZOS)
    global for_debug
    if for_debug:
        img.save("d_color.png")
    
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    # 做颜色合并
    dominate_color = []
    for i in range(len(color_counts)):
        color_count, color_index = color_counts[i]
        now_color = palette[color_index*3:color_index*3+3]
        # if for_debug:
        #     # print("---- new ----")
        #     # print(" now_color: ", rgb_to_hex(now_color))
        #     debug_color = np.ones((20,20,3))
        #     debug_color *= now_color
        #     debug_color = np.array(debug_color, dtype=np.uint8)
        #     debug_color = cv2.cvtColor(debug_color, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(f"debug_color/{rgb_to_hex(now_color)}.png", debug_color)
        add_color = True
        for color in dominate_color:
            # 算颜色距离，应基于LAB进行计算，映射到BGR上可以用对应加权公示进行计算
            b1,g1,r1 = color
            r2,g2,b2 = now_color
            rmean = (r1+r2)/2
            r = r1-r2
            g = g1-g2
            b = b1-b2
            sum_dis = math.sqrt((2+rmean/256)*(r**2)+4*(g**2)+(2+(255-rmean)/256)*(b**2))  # delta_e_cie76
            if sum_dis<dis_thresh:
                add_color = False
        if add_color:
            dominate_color.append((now_color[2],now_color[1],now_color[0]))  # rgb->bgr
            if for_debug:
                color_str = rgb_to_hex(now_color)
                print("==== add color: ", color_str, ", counts: ", color_count)
    
    # print("len(dominate_color): ", len(dominate_color))
    # print(dominate_color)
    return dominate_color


def get_colors(img, w, h, num_color, dominate_color=None):
    global for_debug
    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
    # 获取颜色列表
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    # define criteria, number of clusters(num_color) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    if dominate_color is not None:
        kmeans = KMeans(n_clusters=num_color, init=dominate_color, n_init=10)
        kmeans.fit(Z)
        center =kmeans.cluster_centers_
        if for_debug:
            print("center: ", center)
        res = kmeans.cluster_centers_[kmeans.predict(Z)]
        res = np.uint8(res)
    else:
        ret, label, center = cv2.kmeans(Z, num_color, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        res = np.uint8(center)[label.flatten()]
    res = res.reshape(img.shape)
    if for_debug:
        cv2.imwrite("check_res.png", res)
    
    return res, center


def get_colors_new(img, w, h, num_color_range=[1,8], threshold=0.5):
    """
    循环1-8进行聚类,对与原图的差值进行比较,当差值变动较小的时候认为颜色数量合理
    """
    
    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
    # 获取颜色列表
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    res = None
    center = None
    last_inertia = -1
    last_ratio = -1
    for num_color in range(num_color_range[0], num_color_range[1]):
        kmeans = KMeans(n_clusters=num_color, n_init=10)
        kmeans.fit(Z)
        center = kmeans.cluster_centers_
        res = kmeans.cluster_centers_[kmeans.predict(Z)]
        
        
        this_inertia = kmeans.inertia_
        if last_inertia>0:
            this_ratio = last_inertia/this_inertia
            print(f"In {num_color}: last_inertia: {last_inertia}, this_inertia: {this_inertia}, ratio: {this_ratio}")
            if last_ratio>0:
                if this_ratio<2.5 and abs(this_ratio-last_ratio)<threshold:  # 没有明显的波动
                    break
            last_ratio = this_ratio
        last_inertia = this_inertia
        
    
    res = np.uint8(res)
    res = res.reshape(img.shape)
    
    return res, center


def get_one_color_svg(img, color, this_req_folder, g):
    """
    color是bgr
    """
    color = np.array(color, dtype=np.uint8)
    rgb_color = (color[2],color[1],color[0])
    gray_img = cv2.inRange(img, color, color)
    gray_img = cv2.bitwise_not(gray_img)
    # 消除小区域，先腐蚀后膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    
    color_str = rgb_to_hex(rgb_color)
    this_img_file = os.path.join(this_req_folder, color_str+".bmp")
    this_svg_file = os.path.join(this_req_folder, color_str+".svg")
    cv2.imwrite(this_img_file, gray_img)
    os.system(f"potrace '{this_img_file}' -s -t 15 -a 0.8 -o '{this_svg_file}'")
    
    input_root = ET.parse(this_svg_file).getroot()
    for path in input_root[1]:
        this_v = path.attrib
        no_value = True
        for v in this_v.values():
            if v:
                no_value = False
                break
        if no_value:
            continue
        new_path = ET.SubElement(g, "path")
        new_path.set("fill", color_str)
        for k,v in this_v.items():
            new_path.set(k, v)


def get_connected_svg(img, w, h, this_req_folder, g):
    """
    获取联通域,分别对每块区域进行矢量化
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("img_gray.png", img_gray)
    labels=measure.label(img_gray, connectivity=2)
    cells_color = color.label2rgb(labels, bg_label=0, bg_color=(255, 255, 255))
    plt.figure(figsize=(10, 10))
    plt.imshow(cells_color)
    plt.savefig("color_cells.png")
    regions = measure.regionprops(labels)
    
    
    for id, region in enumerate(regions):
        # 近似整图的联通域是不需要的
        if region.bbox_area > 0.9*h*w:  
            continue
        # # 过滤掉非常小的区域，这种区域可能是边界上的意外连接
        # elif region.area < 100 or region.bbox[2] - region.bbox[0] < 10 or region.bbox[3] - region.bbox[1] < 10:
        #     continue
        

def draw_svg(img, w, h, this_req_folder, out_path, num_color=-1):
    """
    img: str, cv2读入的图片格式
    w: int, 原图的宽
    h: int, 原图的高
    num_color: int, 图片聚类的颜色数量
    out_path: str, svg保存的路径
    """
    
    # 写固定信息
    root = ET.Element("svg")
    root.set("version", "1.0")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("viewBox", f"0 0 {w} {h}")
    root.set("preserveAspectRatio", "xMidYMid meet")
    metadata = ET.SubElement(root, "metadata")
    metadata.text = "Created by zz :)"
    style = ET.SubElement(root, "style")
    style.text = "path { mix-blend-mode: multiply; }"
    g = ET.SubElement(root, "g")
    g.set("transform", f"translate(0.000000,{w}.000000) scale(0.100000,-0.100000)")
    g.set("stroke", "none")
    
    # 双边滤波
    img = cv2.bilateralFilter(img, 25, 150, 100)  # img, d, sigmaColor, sigmaSpace
    global for_debug
    if for_debug:
        cv2.imwrite("check_bifilter.png", img)
    # 获取颜色数量
    if num_color==-1:
        dominate_color = get_dominate_colors(img, w, h)
        dominate_color = np.array([np.array(c, dtype=np.uint8) for c in dominate_color])
        res, colors = get_colors(img, w, h, len(dominate_color))  #res, colors = get_colors(img, w, h, len(dominate_color), dominate_color=dominate_color)
        # res, colors = get_colors_new(img, w, h)
    else:
        res, colors = get_colors(img, w, h, num_color)
    # print("=== len(colors): ", len(colors))
    for color in colors:
        get_one_color_svg(res,color,this_req_folder,g)
    
    # # 抛弃之前的颜色聚类方法，参考vectorizer.ai用的联通域方法
    # get_connected_svg(img, w, h, this_req_folder, g)
    
    
    # 写入文件
    top = '<?xml version="1.0" standalone="no"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n'
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    xmlstr = "\n".join(xmlstr.split("\n")[1:])
    xmlstr = top + xmlstr
    with open(out_path, "w") as fout:
        fout.write(xmlstr)
    # print("Save svg file to: ", out_path)

