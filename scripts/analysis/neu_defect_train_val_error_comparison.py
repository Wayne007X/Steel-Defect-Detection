# -*- coding: utf-8 -*-
"""NEU-DET clean 与 mosaic000 训练集/验证集同口径错误对照。仅推理，不训练。"""
from pathlib import Path
from collections import defaultdict
import csv, gc, sys, traceback
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------- 路径 ----------------
ROOT = Path(r"D:\yolov8\NEU-DET")
PROJECT = Path(r"E:\yolo_project")
SPLITS = {
    "train": (ROOT / "train" / "images", ROOT / "train" / "labels"),
    "val": (ROOT / "validation" / "images", ROOT / "validation" / "labels"),
}
MODELS = [
    ("neu_defect_clean", PROJECT / "runs" / "neu_defect_clean" / "weights" / "best.pt"),
    ("neu_defect_mosaic000", PROJECT / "runs" / "neu_defect_mosaic000" / "weights" / "best.pt"),
]
OUT = PROJECT / "runs" / "neu_defect_train_val_error_comparison"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMGSZ, BATCH, DEVICE = 640, 4, "0"
CANDIDATE_CONF, WORK_CONF, NMS_IOU, MATCH_IOU, MAX_DET = 0.001, 0.25, 0.70, 0.50, 300


def csv_write(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def list_images(root):
    if not root.is_dir(): raise FileNotFoundError(f"图片目录不存在：{root}")
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS), key=lambda p: str(p).lower())


def read_labels(path, width, height):
    if not path.is_file(): raise FileNotFoundError(f"缺少标签文件：{path}")
    txt = path.read_text(encoding="utf-8-sig").strip()
    out = []
    if not txt: return out
    for ln, line in enumerate(txt.splitlines(), 1):
        a = line.split()
        if len(a) < 5: raise ValueError(f"标签格式错误：{path} 第 {ln} 行")
        cid = int(float(a[0])); xc, yc, bw, bh = map(float, a[1:5])
        if cid < 0 or cid >= len(CLASSES): raise ValueError(f"类别 ID 超界：{path} 第 {ln} 行 class={cid}")
        x1=max(0.,(xc-bw/2)*width); y1=max(0.,(yc-bh/2)*height)
        x2=min(float(width),(xc+bw/2)*width); y2=min(float(height),(yc+bh/2)*height)
        out.append({"class_id":cid,"class_name":CLASSES[cid],"x1":x1,"y1":y1,"x2":x2,"y2":y2,"label_path":str(path),"label_line":ln})
    return out


def iou(a,b):
    x1=max(a["x1"],b["x1"]); y1=max(a["y1"],b["y1"])
    x2=min(a["x2"],b["x2"]); y2=min(a["y2"],b["y2"])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    aa=max(0.,a["x2"]-a["x1"])*max(0.,a["y2"]-a["y1"])
    ab=max(0.,b["x2"]-b["x1"])*max(0.,b["y2"]-b["y1"])
    return inter/(aa+ab-inter) if aa+ab-inter > 0 else 0.


def div(a,b): return a/b if b else 0.


def predict_split(model, images):
    pred_map = {}
    results = model.predict(source=[str(p) for p in images], stream=True, imgsz=IMGSZ, batch=BATCH,
                           conf=CANDIDATE_CONF, iou=NMS_IOU, max_det=MAX_DET, device=DEVICE,
                           augment=False, save=False, verbose=False)
    for n, r in enumerate(results, 1):
        boxes=[]
        if r.boxes is not None and len(r.boxes):
            xyxy=r.boxes.xyxy.cpu().numpy(); confs=r.boxes.conf.cpu().numpy(); ids=r.boxes.cls.cpu().numpy().astype(int)
            for b,c,k in zip(xyxy,confs,ids):
                if 0 <= k < len(CLASSES):
                    boxes.append({"class_id":int(k),"class_name":CLASSES[int(k)],"confidence":float(c),
                                  "x1":float(b[0]),"y1":float(b[1]),"x2":float(b[2]),"y2":float(b[3])})
        pred_map[str(Path(r.path).resolve()).lower()] = boxes
        if n % 100 == 0 or n == len(images): print(f"    推理 {n}/{len(images)}")
    return pred_map


def analyze_image(model_name, split, image, gts, preds):
    # 对每个候选预测，计算与真实框的最大同类/异类 IoU。
    for p in preds:
        p["max_same_iou"]=0.; p["max_other_iou"]=0.
        for g in gts:
            ov=iou(p,g)
            if p["class_id"] == g["class_id"]: p["max_same_iou"]=max(p["max_same_iou"],ov)
            else: p["max_other_iou"]=max(p["max_other_iou"],ov)

    active=[i for i,p in enumerate(preds) if p["confidence"] >= WORK_CONF]
    active.sort(key=lambda i: preds[i]["confidence"], reverse=True)
    matched_gt={}; matched_pred=set()
    # 同类别、一对一、按置信度降序贪心匹配。
    for pi in active:
        p=preds[pi]; options=[]
        for gi,g in enumerate(gts):
            if gi in matched_gt or p["class_id"] != g["class_id"]: continue
            ov=iou(p,g)
            if ov >= MATCH_IOU: options.append((ov,gi))
        if options:
            ov,gi=max(options); matched_gt[gi]=pi; matched_pred.add(pi)
            p["status"]="TP"; p["error_type"]="true_positive"; p["matched_gt_index"]=gi; p["matched_iou"]=ov

    # 未匹配的高置信度预测分类为不同 FP 类型。
    for pi in active:
        if pi in matched_pred: continue
        p=preds[pi]
        same_overlaps=[(iou(p,g),gi) for gi,g in enumerate(gts) if p["class_id"]==g["class_id"]]
        other_max=max((iou(p,g) for g in gts if p["class_id"]!=g["class_id"]),default=0.)
        same_high=[(ov,gi) for ov,gi in same_overlaps if ov>=MATCH_IOU]
        if same_high and all(gi in matched_gt for _,gi in same_high): et="duplicate_or_competing_prediction"
        elif other_max >= MATCH_IOU: et="class_confusion_prediction"
        elif max((ov for ov,_ in same_overlaps),default=0.) >= .10: et="localization_related_prediction"
        else: et="background_prediction"
        p["status"]="FP"; p["error_type"]=et; p["matched_gt_index"]=""; p["matched_iou"]=0.

    gt_rows=[]; pred_rows=[]; image_rows=[]; gt_err=defaultdict(int); pred_err=defaultdict(int)
    for gi,g in enumerate(gts):
        best_iou=0.; best_conf=0.; best_class=""; et="TP"
        if gi in matched_gt:
            p=preds[matched_gt[gi]]; best_iou=iou(p,g); best_conf=p["confidence"]; best_class=p["class_name"]
        else:
            same=[]; other=[]
            for p in preds:
                ov=iou(p,g)
                if p["class_id"]==g["class_id"]: same.append((ov,p["confidence"],p["class_name"]))
                else: other.append((ov,p["confidence"],p["class_name"]))
            high=[x for x in same if x[0]>=MATCH_IOU and x[1]>=WORK_CONF]
            low=[x for x in same if x[0]>=MATCH_IOU and CANDIDATE_CONF<=x[1]<WORK_CONF]
            if high: et="duplicate_or_competing_ground_truth"; best_iou,best_conf,best_class=max(high,key=lambda x:(x[0],x[1]))
            elif low: et="low_confidence_candidate"; best_iou,best_conf,best_class=max(low,key=lambda x:(x[0],x[1]))
            else:
                ms=max(same,default=(0.,0.,"")); mo=max(other,default=(0.,0.,""))
                if ms[0]>=.10: et="localization_related_miss"; best_iou,best_conf,best_class=ms
                elif mo[0]>=MATCH_IOU: et="class_confusion_miss"; best_iou,best_conf,best_class=mo
                else: et="no_candidate_miss"; best_iou,best_conf,best_class=ms
        gt_rows.append({"model":model_name,"split":split,"image":str(image),"class_id":g["class_id"],"class_name":g["class_name"],
                        "gt_index":gi,"gt_x1":round(g["x1"],3),"gt_y1":round(g["y1"],3),"gt_x2":round(g["x2"],3),"gt_y2":round(g["y2"],3),
                        "label_path":g["label_path"],"label_line":g["label_line"],"gt_status":"TP" if et=="TP" else "FN",
                        "gt_error_type":et,"best_candidate_iou":round(best_iou,6),"best_candidate_confidence":round(best_conf,6),"best_candidate_class":best_class})
        if et!="TP": gt_err[(g["class_id"],et)]+=1

    for pi,p in enumerate(preds):
        status=p.get("status","below_working_threshold") if p["confidence"]>=WORK_CONF else "below_working_threshold"
        et=p.get("error_type","low_confidence_candidate") if p["confidence"]>=WORK_CONF else "low_confidence_candidate"
        pred_rows.append({"model":model_name,"split":split,"image":str(image),"prediction_index":pi,"class_id":p["class_id"],"class_name":p["class_name"],
                          "confidence":round(p["confidence"],6),"x1":round(p["x1"],3),"y1":round(p["y1"],3),"x2":round(p["x2"],3),"y2":round(p["y2"],3),
                          "status_at_conf_025":status,"prediction_error_type":et,"max_same_class_iou":round(p["max_same_iou"],6),
                          "max_other_class_iou":round(p["max_other_iou"],6),"matched_gt_index":p.get("matched_gt_index",""),"matched_iou":round(p.get("matched_iou",0.),6)})
        if p["confidence"]>=WORK_CONF and status=="FP": pred_err[(p["class_id"],et)]+=1

    for cid,cname in enumerate(CLASSES):
        cls_g=[g for g in gts if g["class_id"]==cid]
        cls_p=[p for p in preds if p["class_id"]==cid and p["confidence"]>=WORK_CONF]
        tp=sum(p.get("status")=="TP" for p in cls_p); fp=sum(p.get("status")=="FP" for p in cls_p); fn=len(cls_g)-tp
        image_rows.append({"model":model_name,"split":split,"image":str(image),"class_id":cid,"class_name":cname,"gt_count":len(cls_g),
                           "pred_count_at_conf_025":len(cls_p),"TP":tp,"FP":fp,"FN":fn})
    return gt_rows,pred_rows,image_rows,gt_err,pred_err


def analyze_split(model_name, model, split, image_root, label_root):
    images=list_images(image_root)
    if not images: raise RuntimeError(f"未找到图片：{image_root}")
    print(f"\n[{model_name}] {split}：{len(images)} 张图片")
    pred_map=predict_split(model,images)
    gt_details=[]; pred_details=[]; image_details=[]; gt_errors=defaultdict(int); pred_errors=defaultdict(int)
    totals=defaultdict(lambda: {"GT":0,"TP":0,"FP":0,"FN":0})
    missing=[]
    for n,img in enumerate(images,1):
        label=label_root / img.relative_to(image_root).with_suffix(".txt")
        if not label.is_file(): missing.append(str(label)); continue
        with Image.open(img) as im: w,h=im.size
        gts=read_labels(label,w,h); preds=pred_map.get(str(img.resolve()).lower(),[])
        gr,pr,ir,ge,pe=analyze_image(model_name,split,img,gts,preds)
        gt_details.extend(gr); pred_details.extend(pr); image_details.extend(ir)
        for k,v in ge.items(): gt_errors[k]+=v
        for k,v in pe.items(): pred_errors[k]+=v
        for r in ir:
            t=totals[r["class_id"]]; t["GT"]+=r["gt_count"]; t["TP"]+=r["TP"]; t["FP"]+=r["FP"]; t["FN"]+=r["FN"]
        if n%100==0 or n==len(images): print(f"    错误分类 {n}/{len(images)}")
    if missing:
        raise FileNotFoundError(f"{split} 有 {len(missing)} 张图片缺标签。第一个：{missing[0]}。脚本不会把缺失标签当背景图处理。")

    summary=[]
    for cid,cname in enumerate(CLASSES):
        t=totals[cid]; p=div(t["TP"],t["TP"]+t["FP"]); r=div(t["TP"],t["TP"]+t["FN"]); f=div(2*p*r,p+r)
        row={"model":model_name,"split":split,"class_id":cid,"class_name":cname,"images":len(images),**t,
             "precision":round(p,6),"recall":round(r,6),"F1":round(f,6)}
        for name in ["background_prediction","localization_related_prediction","duplicate_or_competing_prediction","class_confusion_prediction"]:
            row["FP_"+name.replace("_prediction","")]=gt_unused=pred_errors[(cid,name)]
        for name in ["low_confidence_candidate","localization_related_miss","class_confusion_miss","no_candidate_miss","duplicate_or_competing_ground_truth"]:
            row["FN_"+name.replace("_miss","").replace("_ground_truth","")]=gt_errors[(cid,name)]
        summary.append(row)
    tp=sum(x["TP"] for x in summary); fp=sum(x["FP"] for x in summary); fn=sum(x["FN"] for x in summary)
    p=div(tp,tp+fp); r=div(tp,tp+fn); f=div(2*p*r,p+r)
    overall={"model":model_name,"split":split,"class_id":"","class_name":"all","images":len(images),"GT":sum(x["GT"] for x in summary),
             "TP":tp,"FP":fp,"FN":fn,"precision":round(p,6),"recall":round(r,6),"F1":round(f,6)}
    for k in summary[0]:
        if k.startswith("FP_") or k.startswith("FN_"): overall[k]=sum(int(x.get(k,0)) for x in summary)
    summary.append(overall)
    return summary,gt_details,pred_details,image_details


def main():
    for p in [*SPLITS["train"],*SPLITS["val"]]:
        if not p.is_dir(): raise FileNotFoundError(f"目录不存在：{p}")
    for name,p in MODELS:
        if not p.is_file(): raise FileNotFoundError(f"模型权重不存在：{p}")
    if OUT.exists(): raise FileExistsError(f"输出目录已存在，为避免覆盖旧结果，停止运行：{OUT}\n如需重跑，请修改脚本中的 OUT 目录名。")
    OUT.mkdir(parents=True)
    all_summary=[]; all_gt=[]; all_pred=[]; all_images=[]
    for name,weight in MODELS:
        print("\n"+"="*76+f"\n模型：{name}\n权重：{weight}\n"+"="*76)
        model=YOLO(str(weight)); print(f"模型类别：{model.names}")
        for split,(image_root,label_root) in SPLITS.items():
            s,g,p,i=analyze_split(name,model,split,image_root,label_root)
            all_summary.extend(s); all_gt.extend(g); all_pred.extend(p); all_images.extend(i)
        del model; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass

    csv_write(OUT/"metrics_summary.csv",all_summary)
    # train/val 差值，按模型与类别配对。
    gaps=[]
    for name,_ in MODELS:
        for cname in ["all"]+CLASSES:
            tr=next(x for x in all_summary if x["model"]==name and x["split"]=="train" and x["class_name"]==cname)
            va=next(x for x in all_summary if x["model"]==name and x["split"]=="val" and x["class_name"]==cname)
            row={"model":name,"class_name":cname,"train_images":tr["images"],"val_images":va["images"],"train_GT":tr["GT"],"val_GT":va["GT"]}
            for metric in ["precision","recall","F1"]:
                row["train_"+metric]=tr[metric]; row["val_"+metric]=va[metric]; row["val_minus_train_"+metric]=round(va[metric]-tr[metric],6)
            for metric in sorted(k for k in tr if k.startswith("FP_") or k.startswith("FN_")):
                row["train_"+metric]=tr.get(metric,0); row["val_"+metric]=va.get(metric,0); row["val_minus_train_"+metric]=int(va.get(metric,0))-int(tr.get(metric,0))
            gaps.append(row)
    csv_write(OUT/"train_val_gap.csv",gaps)
    csv_write(OUT/"gt_error_details.csv",all_gt)
    csv_write(OUT/"prediction_details.csv",all_pred)
    csv_write(OUT/"image_summary.csv",all_images)
    (OUT/"run_info.txt").write_text(
        f"imgsz={IMGSZ}\ncandidate_conf={CANDIDATE_CONF}\nworking_conf={WORK_CONF}\nNMS_IOU={NMS_IOU}\nMATCH_IOU={MATCH_IOU}\nmax_det={MAX_DET}\n"
        "TP/FP/FN 使用 conf=0.25、同类别一对一匹配、IoU=0.50。候选诊断从 conf=0.001 开始。\n"
        "训练集结果只用于诊断，不是独立测试结果。\n",encoding="utf-8")
    print("\n对照完成，没有训练模型。结果目录：",OUT)
    for f in ["metrics_summary.csv","train_val_gap.csv","gt_error_details.csv","prediction_details.csv","image_summary.csv","run_info.txt"]:
        print(" -",OUT/f)

if __name__=="__main__":
    try: main()
    except Exception as e:
        print("\n脚本运行失败：",e); traceback.print_exc(); sys.exit(1)
