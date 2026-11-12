# 地质灾害航拍影像实例分割数据集 | 泥石流滑坡像素级识别 自然灾害预警 YOLO训练数据
**标签**：#地质灾害监测 #泥石流识别 #滑坡分割 #实例分割 #自然灾害预警 #无人机遥感影像 #YOLO分割数据集 #智慧防灾 #遥感图像处理 #深度学习 #应急减灾


山地、丘陵区域常年受强降雨、地质运动影响，泥石流、山体滑坡等突发性地质灾害频发，传统防灾模式依赖人工现场巡查、遥感影像目视解译，不仅巡查范围有限、响应速度滞后，还难以精准勾勒灾害区域边界、量化受灾面积，极易延误预警与抢险时机。随着无人机遥感与深度学习实例分割技术普及，像素级灾害区域自动提取成为智慧防灾的主流技术路线，但**适配野外复杂地形、灾害形态多样、标注规范的实例分割数据集**十分稀缺，成为算法落地与模型迭代的主要阻碍。本次分享的地质灾害航拍实例分割数据集，聚焦泥石流、滑坡两大典型灾害场景，图像清晰度高、标注完整规范，原生适配YOLO实例分割格式，可直接用于分割模型训练、灾害识别算法研发、应急预警系统搭建，为地质灾害智能化监测、灾后灾情评估提供坚实的数据支撑。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/441586c388bd44b7895d6f71938d6dff.png)

---

## 一、项目总览
### 1.1 数据集定位
本数据集是面向**无人机/航拍遥感场景的地质灾害实例分割专用数据集**，主打泥石流、山体滑坡两类高发地质灾害的像素级区域分割任务。数据集依托野外真实航拍影像制作，充分还原山地复杂地形、植被遮挡、地形起伏等真实干扰因素，标注精准且格式统一，不仅适用于YOLO系列实例分割模型训练、验证、测试，也可迁移至U-Net、Mask R-CNN等主流分割框架，是地质灾害智能识别、遥感影像智能解译、自然灾害预警领域的优质基准数据。

### 1.2 核心基础信息表
| 信息维度 | 详细参数 |
| ---- | ---- |
| 核心任务 | 实例分割（像素级区域划分）、地质灾害目标检测 |
| 标注格式 | 标准YOLO实例分割格式，开箱即用，无需二次转换 |
| 数据总规模 | 合计2435张高清航拍图像，样本体量可支撑中小型分割模型全流程训练 |
| 识别类别 | 泥石流、山体滑坡两大典型地质灾害 |
| 图像特性 | 野外实景拍摄，覆盖不同地形、光照、灾害规模，场景多样性充足 |
| 适用技术框架 | YOLOv5-seg、YOLOv8-seg、Mask R-CNN、U-Net、SAM等分割模型 |
| 核心应用领域 | 地质灾害实时预警、灾后灾情评估、遥感影像自动化处理、防灾科研教学、工程项目开发 |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4f79ff9bb66b4d0d94befba3817fafda.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81253b5212db444f9b6b0a9dbfaaecf3.png)

### 1.3 数据集核心优势与技术难点
1. **核心优势**
    - 标注质量高：针对泥石流、滑坡不规则形态完成像素级精准标注，边界贴合灾害实际轮廓，降低模型学习噪声；
    - 格式兼容性强：原生适配YOLO实例分割标准格式，主流深度学习框架均可快速接入，缩短项目开发周期；
    - 场景贴合实战：图像取自真实野外灾害区域，包含裸地、植被、陡坡等复杂背景，模型训练后泛化能力更强；
    - 样本分布合理：涵盖小型零星灾害、大面积连片灾害等不同形态，有效缓解模型对单一尺度目标的拟合偏向。
2. **行业技术难点**
    - 形态无规则：泥石流、滑坡轮廓杂乱无固定形状，区别于常规规则目标，对模型边界提取能力要求极高；
    - 背景干扰强：山地植被、岩石、阴影易与灾害区域纹理混淆，提升分割难度；
    - 尺度差异大：单张图像内灾害区域面积跨度大，小范围隐患与大面积灾害并存，考验模型多尺度特征提取能力。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d97bc1aed0c54a1fa9d91cba2f893142.png)

## 二、数据集目录结构与标注规范
### 2.1 标准目录结构
采用深度学习分割任务通用目录架构，区分图像、分割标注、配置文件，支持训练集、验证集、测试集自主划分，解压后可直接导入项目工程：
```
geohazard_seg_dataset/
├── images/                # 航拍原图文件夹
│   ├── train/             # 训练集图像
│   ├── val/               # 验证集图像
│   └── test/              # 测试集图像
├── labels/                # YOLO实例分割标注文件，与图像文件名一一对应
│   ├── train/
│   ├── val/
│   └── test/
├── geohazard_seg.yaml     # 数据集核心配置文件（YOLO分割专用）
├── class_list.txt         # 类别名称与ID映射文档
└── README.md              # 数据采集说明、标注规则、使用协议
```

### 2.2 标注格式规则
- 格式标准：遵循YOLO实例分割统一规范，单行内容由**类别ID + 目标框归一化坐标 + 分割轮廓点坐标**组成，所有坐标基于图像尺寸做归一化处理，取值范围 `[0, 1]`；
- ID映射：`0 = 泥石流`，`1 = 滑坡`，全数据集ID与类别严格对应，无错乱标注；
- 匹配规则：图像文件（jpg/png）与标注文件（txt）文件名完全一致，一一绑定，保障数据加载准确性。

---

## 三、深度学习全流程实战代码
基于工业界主流的 **YOLOv8-seg** 实例分割框架开发全套代码，包含环境部署、数据集合法性校验、配置文件编写、模型训练、批量图像分割推理、模型性能评估五大模块。所有代码结合**地质灾害航拍、不规则目标分割**场景添加专属经验注释，针对泥石流、滑坡的特征特性完成超参调优，可直接运行使用。

### 3.1 环境依赖安装
适配 Python 3.8 ~ 3.11 版本，推荐GPU环境运行，一键安装分割任务所需全部依赖库：
```bash
# 适用于Windows、Linux服务器，GPU环境需提前配置CUDA
pip install ultralytics torch opencv-python numpy pyyaml tqdm matplotlib
```

### 3.2 数据集配置文件 geohazard_seg.yaml
YOLO系列实例分割模型必备配置文件，定义数据路径、类别数量、类别名称，直接复制至数据集根目录即可：
```yaml
# 泥石流、滑坡实例分割数据集配置文件（YOLO-seg专用）
path: ./geohazard_seg_dataset  # 数据集根目录（支持绝对路径/相对路径）
train: images/train            # 训练集图像路径
val: images/val                # 验证集图像路径
test: images/test              # 测试集图像路径

# 分割任务核心参数
nc: 2  # 共计2个灾害类别
names:
  0: Mudslide   # 泥石流
  1: Landslide  # 山体滑坡
```

### 3.3 数据集完整性与标注校验代码
**场景注释**：地质灾害图像轮廓点数量多，极易出现标注坐标越界、轮廓点缺失、图像与标注不匹配、类别ID错误等问题。航拍图像还存在文件名异常、格式混杂等情况。本脚本批量校验全量数据，同时统计两类灾害样本分布，提前排查问题，避免训练中途中断。
```python
from pathlib import Path
from tqdm import tqdm

# 全局路径配置，根据本地数据集存放路径修改
ROOT_DIR = Path("./geohazard_seg_dataset")
IMG_PATH = ROOT_DIR / "images"
LABEL_PATH = ROOT_DIR / "labels"
CLASS_NUM = 2  # 固定类别：泥石流、滑坡

def check_seg_dataset():
    """YOLO实例分割数据集校验：文件匹配、标注格式、坐标合法性、类别统计"""
    total_error = 0
    class_stat = [0] * CLASS_NUM

    # 遍历训练集、验证集、测试集
    for split in ["train", "val", "test"]:
        img_sub = IMG_PATH / split
        label_sub = LABEL_PATH / split
        if not img_sub.exists() or not label_sub.exists():
            print(f"【致命错误】{split} 文件夹缺失，终止当前子集校验")
            total_error += 1
            continue

        img_files = list(img_sub.glob("*.jpg")) + list(img_sub.glob("*.png"))
        print(f"\n开始校验 {split} 子集，图像总数：{len(img_files)}")

        for img_file in tqdm(img_files):
            label_file = label_sub / f"{img_file.stem}.txt"
            # 校验标注文件是否存在
            if not label_file.exists():
                print(f"【标注缺失】图像 {img_file.name} 无对应分割标注")
                total_error += 1
                continue

            # 逐行解析分割标注
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    coords = list(map(float, line.split()))
                    # 首个数值为类别ID
                    cls_id = int(coords[0])
                    # 校验类别ID合法性
                    if not (0 <= cls_id < CLASS_NUM):
                        print(f"【ID错误】{label_file.name} 第{line_idx+1}行 非法类别ID：{cls_id}")
                        total_error += 1
                        continue
                    class_stat[cls_id] += 1
                    # 校验所有归一化坐标范围（分割轮廓坐标必须在0-1之间）
                    for pos in coords[1:]:
                        if not (0 <= pos <= 1):
                            print(f"【坐标越界】{label_file.name} 第{line_idx+1}行 轮廓坐标超出范围")
                            total_error += 1
                            break

    # 输出最终校验结果
    print("\n===== 数据集整体校验结果 =====")
    if total_error == 0:
        print("✅ 全量数据校验通过，标注格式、文件匹配均正常，可启动模型训练！")
    else:
        print(f"❌ 累计检测到 {total_error} 处数据异常，请修复后再使用！")

    # 输出两类灾害样本统计
    print("\n===== 灾害类别样本数量统计 =====")
    print(f"0 | 泥石流  样本总数：{class_stat[0]}")
    print(f"1 | 山体滑坡 样本总数：{class_stat[1]}")

if __name__ == "__main__":
    check_seg_dataset()
```

### 3.4 YOLOv8-seg 模型训练代码（地质灾害专项调优）
**场景注释**：泥石流、滑坡属于不规则大尺度目标，且背景纹理复杂。训练时提升迭代轮数保证轮廓特征充分收敛；开启分割专属`copy_paste`增强，强化不规则区域特征学习；选用AdamW优化器提升复杂特征拟合能力；设置合理早停参数，避免样本有限导致的过拟合；调低学习率，保障复杂场景下训练稳定性。
```python
from ultralytics import YOLO

def train_geohazard_seg_model():
    """训练泥石流、滑坡实例分割模型（基于YOLOv8-seg）"""
    # 选用YOLOv8s-seg，平衡精度、推理速度，适配航拍不规则目标分割
    model = YOLO("yolov8s-seg.pt")

    # 面向地质灾害分割的超参数调优
    train_results = model.train(
        data="./geohazard_seg_dataset/geohazard_seg.yaml",
        epochs=120,          # 不规则目标需更多迭代轮数收敛轮廓特征
        batch=12,            # 8G显存推荐8-12，16G显存可设为16
        imgsz=640,           # 航拍图像标准输入尺寸
        device=0,            # GPU训练，无GPU改为 device="cpu"
        lr0=0.0008,          # 低初始学习率，复杂场景训练更平稳
        optimizer="AdamW",
        augment=True,        # 全局数据增强，适配光照、角度变化
        copy_paste=0.2,      # 分割任务专属增强，提升不规则区域分割效果
        mosaic=1.0,
        patience=25,         # 早停策略，连续25轮无精度提升则停止，防止过拟合
        conf=0.2,            # 降低置信度阈值，避免小型灾害漏检漏分割
        project="geohazard_seg_run",
        name="yolov8_seg_disaster",
        save_best=True,      # 仅保留最优精度权重
        plots=True           # 自动生成loss、mAP、分割指标可视化图表
    )
    print(f"🎉 训练完成！最优模型权重路径：{train_results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_geohazard_seg_model()
```

### 3.5 批量图像分割推理与可视化代码
**场景注释**：模拟无人机航拍、遥感影像批量处理的实战场景，模型输出灾害目标框+像素级分割掩码。防灾场景优先保证**不漏判、边界完整**，因此设置较低置信度阈值；推理结果可对接防灾平台，自动统计灾害面积、位置，支撑预警与灾情上报工作。
```python
import cv2
from pathlib import Path
from ultralytics import YOLO

# 推理全局配置参数
BEST_WEIGHT = "./geohazard_seg_run/yolov8_seg_disaster/weights/best.pt"
TEST_IMAGES = Path("./geohazard_seg_dataset/images/test")
CONF_THRESH = 0.2
IOU_THRESH = 0.45

def batch_seg_infer():
    """批量对航拍图像执行泥石流、滑坡实例分割，可视化并输出结果"""
    # 加载训练完成的最优分割模型
    model = YOLO(BEST_WEIGHT)
    img_list = list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png"))
    print(f"开始批量分割推理，待处理图像数量：{len(img_list)}")

    for img_file in img_list:
        # 执行分割推理
        results = model(
            source=str(img_file),
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            save=True,        # 保存带分割掩码、检测框的结果图
            show_labels=True,
            show_conf=True
        )

        # 解析分割结果，结构化输出用于后台统计
        for res in results:
            if res.masks is None:
                print(f"【{img_file.name}】未检测到泥石流、滑坡灾害")
                continue
            # 遍历单张图像内所有分割目标
            for idx, mask in enumerate(res.masks):
                cls_id = int(res.boxes.cls[idx])
                conf = float(res.boxes.conf[idx])
                cls_name = res.names[cls_id]
                print(f"【{img_file.name}】检测到{cls_name}，置信度：{conf:.2f}，已完成区域分割")

if __name__ == "__main__":
    batch_seg_infer()
```

### 3.6 模型分割性能评估代码
**场景注释**：地质灾害预警属于安全优先级极高的场景，漏分割、边界残缺会直接影响防灾决策。本代码重点评估分割任务核心指标mAP、召回率、掩码精度（mask mAP），量化模型在复杂地形下的分割可靠性，也可用于不同模型横向对比。
```python
from ultralytics import YOLO

def evaluate_seg_model():
    """评估实例分割模型综合性能（检测+分割双维度指标）"""
    model = YOLO("./geohazard_seg_run/yolov8_seg_disaster/weights/best.pt")
    # 在验证集上执行模型评估
    metrics = model.val(
        data="./geohazard_seg_dataset/geohazard_seg.yaml",
        conf=0.2,
        iou=0.45
    )

    # 打印核心评估指标
    print("===== 地质灾害实例分割模型评估指标 =====")
    print(f"目标检测 mAP@0.5：{metrics.box.map50:.4f}")
    print(f"分割掩码 mAP@0.5：{metrics.seg.map50:.4f}")
    print(f"平均精确率 Precision：{metrics.box.precision:.4f}")
    print(f"平均召回率 Recall：{metrics.box.recall:.4f}")

if __name__ == "__main__":
    evaluate_seg_model()
```

---

## 四、数据集应用价值与落地场景
### 4.1 核心应用价值
1. **填补数据缺口**：针对地质灾害分割样本稀缺的行业现状，提供2435张高清标注图像，支撑分割算法研发与模型精度迭代；
2. **提升防灾效率**：模型部署后可实现航拍/遥感影像自动化解译，秒级完成灾害区域定位与边界分割，相比人工解译效率提升数十倍，缩短预警响应时间；
3. **量化灾情数据**：像素级分割结果可精准计算灾害面积、影响范围，为灾后救援、灾害等级评定、治理方案制定提供量化数据支撑；
4. **降低作业风险**：替代人工进入高危山区巡查，减少地质灾害区域人员作业隐患，实现非接触式智能监测。

### 4.2 主流落地场景
- **地质灾害预警**：山区、矿区、库区无人机常态化巡检，实时识别泥石流、滑坡隐患，推送预警信息；
- **遥感影像解译**：卫星、航拍遥感数据批量处理，完成大范围区域灾害普查与历史影像分析；
- **灾后应急评估**：地震、强降雨灾害发生后，快速统计受灾范围，辅助应急指挥与救援调度；
- **科研与教学**：遥感科学、地质工程、人工智能交叉学科实验、毕业设计、分割算法竞赛；
- **设备研发**：防灾监测无人机、地面智能监测终端的算法适配与功能测试。

---

## 五、使用规范与优化建议（GitHub Tips）
> [!NOTE]
> 1. 本数据集仅限**学术研究、算法开发、项目测试**使用，禁止二次售卖、商用倒卖及违规传播；
> 2. 实例分割任务算力需求高于普通目标检测，2435张图像建议使用独立GPU训练，完整训练耗时约3~5小时，CPU训练效率极低；
> 3. 如需适配Mask R-CNN、U-Net等非YOLO框架，可使用标注转换工具将YOLO分割格式转为VOC、COCO或语义分割掩码格式；
> 4. 数据集灾害形态多变，不建议使用过度裁剪、强下采样操作，避免分割轮廓特征丢失。

> [!TIP]
> 1. 边缘端部署：无人机、嵌入式设备优先选用YOLOv8-n-seg轻量化模型，配合TensorRT、ONNX做模型量化，提升推理速度；
> 2. 精度优化：针对植被遮挡区域，可额外增加随机遮挡、色彩扰动等数据增强，提升模型抗干扰能力；
> 3. 模型选型：追求极致分割精度可选用YOLOv8l-seg/YOLOv8x-seg，资源有限优先使用轻量模型。

---

## 六、深度拓展方向（技术延伸思考）
1. **多任务融合**：在实例分割基础上增加**灾害等级分类**（轻度/中度/重度）、灾害面积估算，实现“分割+分级+量化”一体化分析；
2. **时序监测分析**：结合多期航拍影像，构建时序分析模型，监测滑坡、泥石流隐患区域的变化趋势，实现中长期风险预判；
3. **多源数据融合**：叠加DEM高程、坡度等地理数据，结合影像特征训练模型，进一步提升复杂山地场景下的分割精度；
4. **视频流实时分割**：基于OpenCV对接无人机实时视频流，实现动态航拍画面逐帧分割，适配野外移动巡检场景。

---

## 文末全量标签
#地质灾害监测 #泥石流分割 #滑坡识别 #YOLO实例分割 #无人机遥感 #自然灾害预警 #智慧防灾 #遥感影像处理 #深度学习数据集 #应急减灾 #像素级分割

