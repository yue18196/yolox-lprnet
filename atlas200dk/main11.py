# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import time

# 引入 AclLite 库 (请确保 acllite 文件夹在当前目录或 PYTHONPATH 中)
# 华为 CANN Samples 通常提供这个封装库
from acllite.acllite_model import AclLiteModel
from acllite.acllite_resource import AclLiteResource

# ================= 配置区域 =================
# 1. 模型路径 (请修改为你转好的 .om 文件名)
YOLOX_MODEL_PATH = "./yolox_plate.om" 
LPR_MODEL_PATH = "./lprnet.om"
IMAGE_PATH = "./test.jpg"
OUTPUT_TXT_PATH = "./output.txt"
print(f"📝 识别结果将记录到: {OUTPUT_TXT_PATH}")
# 2. LPRNet 参数 (必须和训练保持一致)
# 如果你最后用的是 160宽，请改为 160
LPR_WIDTH = 160 
LPR_HEIGHT = 24

# 3. 字典 (66类，去掉了 I 和 O)
CHARS = [
    '京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
    '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
    '桂','琼','川','贵','云','藏','陕','甘','青','宁',
    '新',
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K',
    'L','M','N','P','Q','R','S','T','U','V',
    'W','X','Y','Z',
    '-'
]

# ================= 工具函数 =================

def preprocess_yolox(img, input_size=(640, 640)):
    """YOLOX 预处理: Resize + Pad (不归一化)"""
    padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    
    # HWC -> CHW, float32
    blob = padded_img.transpose(2, 0, 1).astype(np.float32)
    # 【Atlas关键】内存必须连续
    blob = np.ascontiguousarray(blob)
    return blob, r

def nms(boxes, scores, nms_thr):
    """非极大值抑制"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep

def decode_lpr(preds):
    """CTC 贪婪解码"""
    res = []
    for i in range(len(preds)):
        idx = preds[i]
        # 65 是空白符 '-' 的索引
        if idx == len(CHARS) - 1: continue 
        # 去重
        if i > 0 and idx == preds[i-1]: continue 
        res.append(CHARS[idx])
    return "".join(res)

# ================= 主逻辑 =================

def main():
    # 0. 资源初始化
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    # 1. 加载模型
    print(f"🚀 加载模型...\n YOLOX: {YOLOX_MODEL_PATH}\n LPRNet: {LPR_MODEL_PATH}")
    if not os.path.exists(YOLOX_MODEL_PATH) or not os.path.exists(LPR_MODEL_PATH):
        print("❌ 模型文件不存在，请检查路径！")
        return

    model_yolo = AclLiteModel(YOLOX_MODEL_PATH)
    model_lpr = AclLiteModel(LPR_MODEL_PATH)
    

    

    # 2. 读取图片
    src_img = cv2.imread(IMAGE_PATH)
    if src_img is None:
        print(f"❌ 读取图片失败: {IMAGE_PATH}")
        return

    # ----------------------------------------------------
    # Step A: YOLOX 检测
    # ----------------------------------------------------
    t0 = time.time()
    yolo_input_size = (640, 640)
    img_in, ratio = preprocess_yolox(src_img, yolo_input_size)
    
    # 推理 (输入必须是 list)
    yolo_result_list = model_yolo.execute([img_in[None, :]]) 
    
    # 获取输出 (1, 8400, 6) -> 已经是解码后的绝对坐标
    predictions = yolo_result_list[0][0] 
    
    # 坐标转换: cx,cy,w,h -> x1,y1,x2,y2
    boxes_xywh = predictions[:, :4]
    boxes_xyxy = np.ones_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2]/2.
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3]/2.
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]/2.
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]/2.
    
    scores = predictions[:, 4] * predictions[:, 5]
    
    # 筛选
    mask = scores > 0.7  # 提高阈值过滤假车牌
    dets = boxes_xyxy[mask]
    scores = scores[mask]
    
    final_boxes = []
    if len(dets) > 0:
        keep = nms(dets, scores, 0.45)
        final_boxes = dets[keep]
        print(f"✅ YOLOX 检测到 {len(final_boxes)} 个目标 (耗时 {time.time()-t0:.4f}s)")
    else:
        print("⚠️ 未检测到车牌")
    
    # ----------------------------------------------------
    # Step B: LPRNet 识别
    # ----------------------------------------------------
    for i, box in enumerate(final_boxes):
        # 1. 坐标还原
        box /= ratio
        x1, y1, x2, y2 = box.astype(int)
        
        # 2. Padding (优化后的策略)
        w_box, h_box = x2 - x1, y2 - y1
        pad_w = int(w_box * 0.06) # 左右多留点，防止切掉省份
        pad_h = int(h_box * 0.04)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(src_img.shape[1], x2 + pad_w)
        y2 = min(src_img.shape[0], y2 + pad_h)
        
        # 3. 抠图
        plate_img = src_img[y1:y2, x1:x2]
        if plate_img.size == 0: continue
        
        # 4. LPR 预处理 (必须和训练一致: 彩色 + Resize + Norm)
        # 注意: 使用 INTER_CUBIC 提升清晰度
        lpr_img = cv2.resize(plate_img, (LPR_WIDTH, LPR_HEIGHT), interpolation=cv2.INTER_CUBIC)
        lpr_img = lpr_img.astype('float32')
        lpr_img -= 127.5
        lpr_img *= 0.0078125
        
        # HWC -> CHW
        lpr_img = lpr_img.transpose(2, 0, 1)
        lpr_img = np.ascontiguousarray(lpr_img) # 内存连续
        
        # 5. 推理
        t_lpr = time.time()
        lpr_result_list = model_lpr.execute([lpr_img[None, :]])
        lpr_output = lpr_result_list[0][0] # [66, 24] 或 [24, 66]
        
        # 6. 维度自动判断
        class_dim = -1
        if lpr_output.shape[0] == 66: class_dim = 0
        elif lpr_output.shape[1] == 66: class_dim = 1
        
        if class_dim == -1: raw_preds = np.argmax(lpr_output, axis=0) # 盲猜
        else: raw_preds = np.argmax(lpr_output, axis=class_dim)
        
        # 7. 解码
        text = decode_lpr(raw_preds)
        
        # 8. 简单过滤
        if len(text) < 7:
            print(f"   [过滤] 结果太短: {text}")
            continue
            
        print(f"🚗 车牌 {i+1}: {text} (LPR耗时 {time.time()-t_lpr:.4f}s)")

        try:
        # 使用 'a' 模式：如果文件不存在则创建，存在则追加写入
            with open(OUTPUT_TXT_PATH, 'a', encoding='utf-8') as f:
            # 写入内容：可以包含时间戳和车牌号
               current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
               record_line = f"[{current_time}] 检测到车牌: {text}\n"
               f.write(record_line)
               print(f"   💾 已记录到文件")
        except Exception as e:
            print(f"   ❌ 写入文件时出错: {e}")
        
        # 9. 绘图 (在板子上跑如果不接显示器，这一步主要是保存看结果)
        cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # PutText 不支持中文，只显示后半部分或拼音，或者干脆不写
        cv2.putText(src_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 保存结果图
    cv2.imwrite("result_atlas.jpg", src_img)
    print("💾 结果已保存至 result_atlas.jpg")

if __name__ == '__main__':
    main()