# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import time
import shutil

# 引入 AclLite
from acllite.acllite_model import AclLiteModel
from acllite.acllite_resource import AclLiteResource

# ================= 配置区域 =================
YOLOX_MODEL_PATH = "./yolox_plate.om"
LPR_MODEL_PATH = "./lprnet.om"

# 监听文件夹配置
INPUT_FOLDER = "./input"       # 把图片放这里
PROCESSED_FOLDER = "./processed" # 处理完的图移到这里(防止重复处理)
OUTPUT_TXT_PATH = "./output.txt" # 结果文件

# LPRNet 参数
LPR_WIDTH = 160
LPR_HEIGHT = 24

# 字典 (66类)
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

# ================= MQTT 预留接口 =================
def send_mqtt_message(plate_text, timestamp):
    """
    预留 MQTT 发送功能
    TODO: 在这里实现 MQTT 发布逻辑
    """
    payload = {
        "plate": plate_text,
        "time": timestamp,
        "device_id": "Atlas200DK_01"
    }
    # print(f"[MQTT Stub] Sending: {payload}")
    pass

# ================= 核心算法函数 (保持不变) =================
def preprocess_yolox(img, input_size=(640, 640)):
    padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    blob = padded_img.transpose(2, 0, 1).astype(np.float32)
    blob = np.ascontiguousarray(blob)
    return blob, r

def nms(boxes, scores, nms_thr):
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
    res = []
    for i in range(len(preds)):
        idx = preds[i]
        if idx == len(CHARS) - 1: continue
        if i > 0 and idx == preds[i-1]: continue
        res.append(CHARS[idx])
    return "".join(res)

# ================= 单张图片处理逻辑 =================
def process_single_image(model_yolo, model_lpr, img_path):
    filename = os.path.basename(img_path)
    print(f"\n📸 处理图片: {filename}")
    
    src_img = cv2.imread(img_path)
    if src_img is None:
        print("❌ 图片损坏或无法读取")
        return

    # --- YOLOX ---
    t0 = time.time()
    img_in, ratio = preprocess_yolox(src_img)
    yolo_res = model_yolo.execute([img_in[None, :]])[0][0]
    
    boxes = yolo_res[:, :4]
    # 还原坐标 xywh -> xyxy
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    
    scores = yolo_res[:, 4] * yolo_res[:, 5]
    mask = scores > 0.7
    dets = boxes_xyxy[mask]
    scores = scores[mask]
    
    keep = nms(dets, scores, 0.45)
    final_boxes = dets[keep]
    
    if len(final_boxes) == 0:
        print("⚠️ 未检测到车牌")
        return

    # --- LPRNet ---
    results = []
    for box in final_boxes:
        box /= ratio
        x1, y1, x2, y2 = box.astype(int)
        
        # Padding
        w, h = x2-x1, y2-y1
        pad_w, pad_h = int(w*0.06), int(h*0.04)
        x1, y1 = max(0, x1-pad_w), max(0, y1-pad_h)
        x2, y2 = min(src_img.shape[1], x2+pad_w), min(src_img.shape[0], y2+pad_h)
        
        plate_img = src_img[y1:y2, x1:x2]
        if plate_img.size == 0: continue
        
        # LPR Preprocess
        lpr_img = cv2.resize(plate_img, (LPR_WIDTH, LPR_HEIGHT), interpolation=cv2.INTER_CUBIC)
        lpr_img = lpr_img.astype('float32')
        lpr_img -= 127.5
        lpr_img *= 0.0078125
        lpr_img = lpr_img.transpose(2, 0, 1)
        lpr_img = np.ascontiguousarray(lpr_img)
        
        # Inference
        lpr_res = model_lpr.execute([lpr_img[None, :]])[0][0]
        
        # Decode
        class_dim = 0 if lpr_res.shape[0] == 66 else 1
        raw_preds = np.argmax(lpr_res, axis=class_dim)
        text = decode_lpr(raw_preds)
        
        if len(text) > 6:
            results.append(text)
            print(f"✅ 识别结果: {text}")
            
            # 画图 (可选，如果不需要保存图片可注释)
            #cv2.rectangle(src_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- 结果写入与后续处理 ---
    if results:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 1. 写入 TXT
        try:
            with open(OUTPUT_TXT_PATH, 'a', encoding='utf-8') as f:
                for res in results:
                    line = f"[{current_time}] File: {filename} | Plate: {res}\n"
                    f.write(line)
            print(f"💾 已记录到 {OUTPUT_TXT_PATH}")
        except Exception as e:
            print(f"❌ 写入失败: {e}")
            
        # 2. 调用 MQTT (预留)
        # for res in results:
        #     send_mqtt_message(res, current_time)
            
        # 3. (可选) 保存处理后的图片
        # save_path = os.path.join(PROCESSED_FOLDER, "result_" + filename)
        # cv2.imwrite(save_path, src_img)

# ================= 主循环 =================
def main():
    # 1. 初始化
    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
    if not os.path.exists(PROCESSED_FOLDER): os.makedirs(PROCESSED_FOLDER)
    
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    model_yolo = AclLiteModel(YOLOX_MODEL_PATH)
    model_lpr = AclLiteModel(LPR_MODEL_PATH)
    
    print(f"🚀 系统就绪！正在监听文件夹: {INPUT_FOLDER}")
    print("⏳ 等待图片传入...")

    try:
        while True:
            # 扫描文件夹内的图片
            # 获取所有文件，按修改时间排序(保证先处理旧的)
            files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if not files:
                time.sleep(1) # 没有图片就休息1秒，省CPU
                continue
                
            # 有图片了！
            for file_name in files:
                file_path = os.path.join(INPUT_FOLDER, file_name)
                
                # 为了防止图片正在拷贝中被读取导致损坏，稍微等一下或者try-catch
                # 在实际工程中，通常会检测文件是否被占用，这里简单处理
                try:
                    process_single_image(model_yolo, model_lpr, file_path)
                    
                    # 处理完后，移动文件到 processed 目录，防止重复处理
                    # 或者直接删除: os.remove(file_path)
                    shutil.move(file_path, os.path.join(PROCESSED_FOLDER, file_name))
                    print("🧹 文件已归档")
                    
                except Exception as e:
                    print(f"❌ 处理文件 {file_name} 出错: {e}")
                    # 出错也要移走，防止卡死循环
                    if os.path.exists(file_path):
                        shutil.move(file_path, os.path.join(PROCESSED_FOLDER, "error_" + file_name))

    except KeyboardInterrupt:
        print("\n🛑 停止监测，释放资源...")
    finally:
        # 这一步其实很难走到，因为是死循环，除非手动Ctrl+C
        pass

if __name__ == '__main__':
    main()