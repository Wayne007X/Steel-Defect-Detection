"""Prepare specific annotation-review questions and conditional paired uncertainty."""
import csv
import json
from pathlib import Path
import numpy as np
from optimize_neu import OUT, DATA, NAMES, dump
from audit_weak_errors import DEST, assignments
from compare_predictions import summarize


def main():
    cases = [
        ("crazing_251", "模型倾向横向覆盖，GT 含纵向或分段框；复核缺陷团划分及边界约定。"),
        ("crazing_253", "多个相邻 GT 与模型上下大框的范围不同；区分定位学习失败与划分口径差异。"),
        ("crazing_257", "模型上下大框跨越部分 GT；复核相连裂纹是否应作为同一目标。"),
        ("crazing_278", "四个 GT 多数未达到当前置信度阈值；检查原图纹理与低分候选。"),
        ("crazing_299", "当前阈值下无输出，存在多个 GT；作为低对比度和低分漏检复核样本。"),
        ("crazing_300", "预测范围与 GT 分段明显不同；复核完整裂纹网络的标注范围。"),
        ("rolled-in_scale_245", "上下两个 GT 对应一个长预测框；复核相邻缺陷团的分隔标准。"),
        ("rolled-in_scale_247", "三个相邻 GT 被一个较大预测覆盖；复核分组口径及 NMS 影响。"),
        ("rolled-in_scale_254", "单个 GT 位于下部，模型覆盖更大的区域；请领域人员确认上下可见纹理是否都属于目标。"),
        ("rolled-in_scale_273", "两个 GT 与多个预测边界不同；定位与背景纹理混淆均需检查。"),
        ("rolled-in_scale_286", "模型倾向多个窄框，GT 较宽；复核分散压入区域应如何组成一个框。"),
        ("rolled-in_scale_292", "四个 GT 被一个大预测覆盖；优先复核目标分组与框范围。"),
    ]
    rows = []
    for stem, question in cases:
        cls = stem.rsplit("_",1)[0]
        rows.append(dict(image=str(DATA / "validation/images" / cls / (stem+".jpg")),
                         xml=str(DATA / "validation/annotations" / (stem+".xml")),
                         yolo=str(DATA / "validation/labels" / cls / (stem+".txt")),
                         comparison=f"images/{stem}.png", review_question=question,
                         status="visual_flag_only_not_confirmed_label_error",
                         confirmed_action="", reviewer=""))
    with (DEST / "annotation_review_queue.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    dump(DEST / "visual_review_notes.json", dict(
        images_screened=120, method="Inspected all ten 12-image contact sheets of the two weak validation classes.",
        observations=["部分样本的相邻 GT 与预测框存在分组或范围差异；这不是已经确认的标签错误。",
                      "低分候选是主要漏检模式之一；降低阈值或放松 NMS 会同时增加误报。",
                      "框尺寸审计不支持将这两个弱类别笼统视为小目标问题。"],
        no_source_labels_edited=True, priority_cases=rows))
    arrays = []
    records = []
    for name in ("global_nms030", "adaptive_weak050"):
        model_records=np.load(OUT / "round2" / (name+".npy"),allow_pickle=True).tolist()
        records.append(model_records)
        array=[]
        for r in model_records:
            keep, matches, claimed, _=assignments(r)
            array.append([len(matches),len(keep)-len(matches),len(r["gt_cls"])-len(claimed)])
        arrays.append(np.array(array))
    assert [r["path"] for r in records[0]] == [r["path"] for r in records[1]]
    groups={}
    for i,r in enumerate(records[0]):
        groups.setdefault(Path(r["path"]).parent.name,[]).append(i)
    rng=np.random.default_rng(20261030)
    samples=np.concatenate([rng.choice(ids,size=(5000,len(ids)),replace=True) for ids in groups.values()],axis=1)
    values=[]
    for array in arrays:
        aggregate=array[samples].sum(1)
        t,fp,fn=aggregate.T
        values.append(np.stack([t/(t+fp), t/(t+fn), 2*t/(2*t+fp+fn)],axis=1))
    delta=values[1]-values[0]
    dump(DEST / "adaptive_paired_bootstrap.json",dict(
        method="5000 paired image bootstrap draws, stratified by six image classes; fixed trained seed42 models.",
        limitation="Conditional uncertainty on this validation sample; does not account for model selection, training seeds, shared batches or deployment shift.",
        candidate_minus_baseline={metric:dict(mean=float(delta[:,j].mean()),ci95=list(map(float,np.quantile(delta[:,j],[.025,.975]))))
                                 for j,metric in enumerate(["precision","recall","F1"])}))
    print("Prepared 12 concrete review cases and paired diagnostic intervals.")


if __name__ == "__main__":
    main()
