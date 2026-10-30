import json
import re
import csv
from pathlib import Path
from collections import Counter


# ============================================================
# ============================================================



OUTPUT_CSV_FILE = Path(
    "./outputs/all_evaluate.csv"
)

IOU_THRESHOLD = 0.5


# ============================================================
# ============================================================

def parse_json_string(text):
    if text is None:
        return None

    if isinstance(text, dict):
        return text

    text = str(text)

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    ).strip()

    start = text.find("{")

    if start == -1:
        return None

    text = text[start:]

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    try:

        decoder = json.JSONDecoder()

        obj, _ = decoder.raw_decode(text)

        return obj

    except json.JSONDecodeError:

        return None


# ============================================================
# ============================================================

def _valid_bbox(box):
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool)
            for x in box
        )
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _bbox_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _select_largest_bbox(r):
    bbox = r.get("bbox")
    bboxes = r.get("bboxes")

    if _valid_bbox(bbox):
        return list(bbox)

    candidates = []
    if isinstance(bboxes, (list, tuple)):
        candidates.extend(box for box in bboxes if _valid_bbox(box))

    if isinstance(bbox, (list, tuple)) and bbox and any(
        isinstance(item, (list, tuple)) for item in bbox
    ):
        candidates.extend(box for box in bbox if _valid_bbox(box))

    if candidates:
        return list(max(candidates, key=_bbox_area))

    return None


def normalize_risk(obj):
    if not isinstance(obj, dict):
        return {
            "risk": False,
            "risks": []
        }

    risk = bool(obj.get("risk", False))
    risks = obj.get("risks", [])

    if not isinstance(risks, list):
        risks = []

    valid_risks = []

    for r in risks:
        if not isinstance(r, dict):
            continue

        category = r.get("risk_category")
        des = r.get("des")

        if not isinstance(category, str) or not isinstance(des, str):
            continue

        bbox = _select_largest_bbox(r)
        if bbox is None:
            continue

        valid_risks.append({
            "risk_category": category,
            "des": des,
            "bbox": bbox
        })

    return {
        "risk": risk,
        "risks": valid_risks
    }


def load_ground_truth(path):

    data = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for index, line in enumerate(f):

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            messages = item["messages"]

            gt_text = None

            image_path = None

            for msg in messages:

                if msg["role"] == "user":

                    for content in msg["content"]:

                        if content.get(
                            "type"
                        ) == "image":

                            image_path = content[
                                "image"
                            ]

                elif msg["role"] == "assistant":

                    for content in msg["content"]:

                        if content.get(
                            "type"
                        ) == "text":

                            gt_text = content[
                                "text"
                            ]

            gt = normalize_risk(
                parse_json_string(
                    gt_text
                )
            )

            data[index] = {

                "image": image_path,

                "gt": gt

            }

    return data


# ============================================================
# ============================================================

def load_predictions(path):

    data = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            index = item["index"]

            prediction = normalize_risk(
                parse_json_string(
                    item.get(
                        "prediction"
                    )
                )
            )

            data[index] = {

                "image": item.get(
                    "image"
                ),

                "prediction": prediction

            }

    return data


# ============================================================
# IoU
# ============================================================

def calculate_iou(
    box1,
    box2
):

    if (
        not isinstance(box1, list)
        or not isinstance(box2, list)
        or len(box1) != 4
        or len(box2) != 4
    ):

        return 0.0

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    inter_w = max(
        0,
        x2 - x1
    )

    inter_h = max(
        0,
        y2 - y1
    )

    intersection = (
        inter_w * inter_h
    )

    area1 = (

        max(
            0,
            box1[2] - box1[0]
        )

        *

        max(
            0,
            box1[3] - box1[1]
        )

    )

    area2 = (

        max(
            0,
            box2[2] - box2[0]
        )

        *

        max(
            0,
            box2[3] - box2[1]
        )

    )

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection / union
    )


# ============================================================
# Category + Des matching
# ============================================================

def semantic_match(
    gt_risks,
    pred_risks
):
    candidates = []

    for pi, pred in enumerate(
        pred_risks
    ):

        for gi, gt in enumerate(
            gt_risks
        ):

            if (
                pred[
                    "risk_category"
                ]
                ==
                gt[
                    "risk_category"
                ]
                and
                pred["des"]
                ==
                gt["des"]
            ):

                candidates.append(
                    (
                        pi,
                        gi
                    )
                )

    matches = []

    used_gt = set()

    used_pred = set()

    for pi, gi in candidates:

        if pi in used_pred:

            continue

        if gi in used_gt:

            continue

        matches.append({

            "pred_idx": pi,

            "gt_idx": gi

        })

        used_pred.add(pi)

        used_gt.add(gi)

    return matches


# ============================================================
# Des-only matching
# ============================================================

def des_only_match(
    gt_risks,
    pred_risks
):

    candidates = []

    for pi, pred in enumerate(
        pred_risks
    ):

        for gi, gt in enumerate(
            gt_risks
        ):

            if (
                pred["des"]
                !=
                gt["des"]
            ):

                continue

            iou = calculate_iou(

                pred["bbox"],

                gt["bbox"]

            )

            candidates.append(

                (
                    iou,
                    pi,
                    gi
                )

            )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matches = []

    used_pred = set()

    used_gt = set()

    for iou, pi, gi in candidates:

        if pi in used_pred:

            continue

        if gi in used_gt:

            continue

        matches.append({

            "pred_idx": pi,

            "gt_idx": gi,

            "iou": iou

        })

        used_pred.add(pi)

        used_gt.add(gi)

    return matches


# ============================================================
# Detection matching
# ============================================================

def detection_match(
    gt_risks,
    pred_risks,
    iou_threshold=0.5
):

    candidates = []

    for pi, pred in enumerate(
        pred_risks
    ):

        for gi, gt in enumerate(
            gt_risks
        ):

            if (
                pred[
                    "risk_category"
                ]
                !=
                gt[
                    "risk_category"
                ]
            ):

                continue

            if (
                pred["des"]
                !=
                gt["des"]
            ):

                continue

            iou = calculate_iou(

                pred["bbox"],

                gt["bbox"]

            )

            candidates.append(

                (
                    iou,
                    pi,
                    gi
                )

            )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matches = []

    used_pred = set()

    used_gt = set()

    for iou, pi, gi in candidates:

        if pi in used_pred:

            continue

        if gi in used_gt:

            continue

        if iou >= iou_threshold:

            matches.append({

                "pred_idx": pi,

                "gt_idx": gi,

                "iou": iou

            })

            used_pred.add(pi)

            used_gt.add(gi)

    return matches


# ============================================================
# P / R / F1
# ============================================================

def calculate_prf(
    tp,
    fp,
    fn
):

    precision = (

        tp / (tp + fp)

        if tp + fp > 0

        else 0

    )

    recall = (

        tp / (tp + fn)

        if tp + fn > 0

        else 0

    )

    f1 = (

        2 * precision * recall
        /
        (precision + recall)

        if precision + recall > 0

        else 0

    )

    return (
        precision,
        recall,
        f1
    )


# ============================================================
# ============================================================

def get_model_and_checkpoint(
    pred_file
):

    checkpoint = (
        pred_file.parent.name
    )

    model = (
        pred_file.parent.parent.name
    )

    return (
        model,
        checkpoint
    )


# ============================================================
# ============================================================

def evaluate(
    pred_file,
    output_detail_file
):

    print()
    print("=" * 80)
    print(
        f"Evaluating: {pred_file}"
    )
    print("=" * 80)

    # ========================================================
    # Loading
    # ========================================================

    gt_data = load_ground_truth(
        GT_FILE
    )

    pred_data = load_predictions(
        pred_file
    )

    print(
        f"GT samples       : "
        f"{len(gt_data)}"
    )

    print(
        f"Pred samples     : "
        f"{len(pred_data)}"
    )

    all_indices = sorted(

        set(gt_data.keys())
        |
        set(pred_data.keys())

    )

    common_indices = sorted(

        set(gt_data.keys())
        &
        set(pred_data.keys())

    )

    missing_prediction = sorted(

        set(gt_data.keys())
        -
        set(pred_data.keys())

    )

    print(
        f"Common samples   : "
        f"{len(common_indices)}"
    )

    print(
        f"Missing prediction: "
        f"{len(missing_prediction)}"
    )

    # ========================================================
    # Risk-level
    # ========================================================

    risk_tp = 0

    risk_fp = 0

    risk_fn = 0

    risk_tn = 0

    # ========================================================
    # Category + Des Semantic
    # ========================================================

    semantic_tp = 0

    semantic_fp = 0

    semantic_fn = 0

    # ========================================================
    # Des-only Semantic
    # ========================================================

    des_only_tp = 0

    des_only_fp = 0

    des_only_fn = 0

    # ========================================================
    # Detection
    # ========================================================

    detection_tp = 0

    detection_fp = 0

    detection_fn = 0

    # ========================================================
    # Localization
    # ========================================================

    semantic_ious = []

    des_only_ious = []

    detection_ious = []

    # ========================================================
    # Category
    # ========================================================

    category_gt = Counter()

    category_pred = Counter()

    category_tp = Counter()

    category_fp = Counter()

    category_fn = Counter()

    # ========================================================
    # Details
    # ========================================================

    details = []

    # ========================================================
    # Loop
    # ========================================================

    for index in all_indices:

        # --------------------------------------------
        # GT
        # --------------------------------------------

        if index in gt_data:

            gt = gt_data[
                index
            ]["gt"]

        else:

            gt = {

                "risk": False,

                "risks": []

            }

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        if index in pred_data:

            pred = pred_data[
                index
            ]["prediction"]

        else:

            pred = {

                "risk": False,

                "risks": []

            }

        gt_risks = gt[
            "risks"
        ]

        pred_risks = pred[
            "risks"
        ]

        # ====================================================
        # Risk-level
        # ====================================================

        gt_has_risk = (
            len(gt_risks) > 0
        )

        pred_has_risk = (
            len(pred_risks) > 0
        )

        if (
            gt_has_risk
            and
            pred_has_risk
        ):

            risk_tp += 1

        elif (
            not gt_has_risk
            and
            pred_has_risk
        ):

            risk_fp += 1

        elif (
            gt_has_risk
            and
            not pred_has_risk
        ):

            risk_fn += 1

        else:

            risk_tn += 1

        # ====================================================
        # Category statistics
        # ====================================================

        for r in gt_risks:

            category_gt[
                r["risk_category"]
            ] += 1

        for r in pred_risks:

            category_pred[
                r["risk_category"]
            ] += 1

        # ====================================================
        # Category + Des Semantic
        # ====================================================

        semantic_matches = semantic_match(

            gt_risks,

            pred_risks

        )

        semantic_tp_img = len(
            semantic_matches
        )

        semantic_fp_img = (

            len(pred_risks)
            -
            semantic_tp_img

        )

        semantic_fn_img = (

            len(gt_risks)
            -
            semantic_tp_img

        )

        semantic_tp += (
            semantic_tp_img
        )

        semantic_fp += (
            semantic_fp_img
        )

        semantic_fn += (
            semantic_fn_img
        )

        # ----------------------------------------------------
        # Category + Des semantic IoU
        # ----------------------------------------------------

        semantic_match_info = []

        for m in semantic_matches:

            gt_risk = gt_risks[
                m["gt_idx"]
            ]

            pred_risk = pred_risks[
                m["pred_idx"]
            ]

            iou = calculate_iou(

                gt_risk["bbox"],

                pred_risk["bbox"]

            )

            semantic_ious.append(
                iou
            )

            semantic_match_info.append({

                "gt_idx":
                    m["gt_idx"],

                "pred_idx":
                    m["pred_idx"],

                "iou":
                    iou

            })

        # ====================================================
        # Des-only Semantic
        # ====================================================

        des_only_matches = des_only_match(

            gt_risks,

            pred_risks

        )

        des_only_tp_img = len(
            des_only_matches
        )

        des_only_fp_img = (

            len(pred_risks)
            -
            des_only_tp_img

        )

        des_only_fn_img = (

            len(gt_risks)
            -
            des_only_tp_img

        )

        des_only_tp += (
            des_only_tp_img
        )

        des_only_fp += (
            des_only_fp_img
        )

        des_only_fn += (
            des_only_fn_img
        )

        # ----------------------------------------------------
        # Des-only IoU
        # ----------------------------------------------------

        des_only_match_info = []

        for m in des_only_matches:

            des_only_ious.append(
                m["iou"]
            )

            des_only_match_info.append({

                "gt_idx":
                    m["gt_idx"],

                "pred_idx":
                    m["pred_idx"],

                "iou":
                    m["iou"]

            })

        # ====================================================
        # Detection
        # ====================================================

        detection_matches = detection_match(

            gt_risks,

            pred_risks,

            IOU_THRESHOLD

        )

        detection_tp_img = len(
            detection_matches
        )

        detection_fp_img = (

            len(pred_risks)
            -
            detection_tp_img

        )

        detection_fn_img = (

            len(gt_risks)
            -
            detection_tp_img

        )

        detection_tp += (
            detection_tp_img
        )

        detection_fp += (
            detection_fp_img
        )

        detection_fn += (
            detection_fn_img
        )

        # ----------------------------------------------------
        # Detection IoU
        # ----------------------------------------------------

        for m in detection_matches:

            detection_ious.append(
                m["iou"]
            )

        # ====================================================
        # Category PRF
        # ====================================================

        matched_gt_category = set()

        matched_pred_category = set()

        for m in semantic_matches:

            gt_risk = gt_risks[
                m["gt_idx"]
            ]

            category = gt_risk[
                "risk_category"
            ]

            category_tp[
                category
            ] += 1

            matched_gt_category.add(
                m["gt_idx"]
            )

            matched_pred_category.add(
                m["pred_idx"]
            )

        for gi, r in enumerate(
            gt_risks
        ):

            if gi not in matched_gt_category:

                category_fn[
                    r["risk_category"]
                ] += 1

        for pi, r in enumerate(
            pred_risks
        ):

            if pi not in matched_pred_category:

                category_fp[
                    r["risk_category"]
                ] += 1

        # ====================================================
        # Detail
        # ====================================================

        detail = {

            "index":
                index,

            "image":
                gt_data.get(
                    index,
                    {}
                ).get(
                    "image"
                ),

            "gt_num":
                len(gt_risks),

            "pred_num":
                len(pred_risks),

            # category + des
            "semantic_tp":
                semantic_tp_img,

            "semantic_fp":
                semantic_fp_img,

            "semantic_fn":
                semantic_fn_img,

            # des only
            "des_only_tp":
                des_only_tp_img,

            "des_only_fp":
                des_only_fp_img,

            "des_only_fn":
                des_only_fn_img,

            # detection
            "detection_tp":
                detection_tp_img,

            "detection_fp":
                detection_fp_img,

            "detection_fn":
                detection_fn_img,

            "semantic_matches":
                semantic_match_info,

            "des_only_matches":
                des_only_match_info,

            "detection_matches":
                detection_matches,

            "gt_risks":
                gt_risks,

            "pred_risks":
                pred_risks

        }

        details.append(
            detail
        )

    # ========================================================
    # Metrics
    # ========================================================

    risk_precision, \
    risk_recall, \
    risk_f1 = calculate_prf(

        risk_tp,

        risk_fp,

        risk_fn

    )

    semantic_precision, \
    semantic_recall, \
    semantic_f1 = calculate_prf(

        semantic_tp,

        semantic_fp,

        semantic_fn

    )

    des_only_precision, \
    des_only_recall, \
    des_only_f1 = calculate_prf(

        des_only_tp,

        des_only_fp,

        des_only_fn

    )

    detection_precision, \
    detection_recall, \
    detection_f1 = calculate_prf(

        detection_tp,

        detection_fp,

        detection_fn

    )

    # ========================================================
    # Semantic Localization
    # ========================================================

    if semantic_ious:

        semantic_mean_iou = (

            sum(semantic_ious)
            /
            len(semantic_ious)

        )

        semantic_iou50 = (

            sum(
                iou >= 0.5
                for iou in semantic_ious
            )
            /
            len(semantic_ious)

        )

        semantic_iou75 = (

            sum(
                iou >= 0.75
                for iou in semantic_ious
            )
            /
            len(semantic_ious)

        )

    else:

        semantic_mean_iou = 0

        semantic_iou50 = 0

        semantic_iou75 = 0

    # ========================================================
    # Des-only Localization
    # ========================================================

    if des_only_ious:

        des_only_mean_iou = (

            sum(des_only_ious)
            /
            len(des_only_ious)

        )

        des_only_iou50 = (

            sum(
                iou >= 0.5
                for iou in des_only_ious
            )
            /
            len(des_only_ious)

        )

        des_only_iou75 = (

            sum(
                iou >= 0.75
                for iou in des_only_ious
            )
            /
            len(des_only_ious)

        )

    else:

        des_only_mean_iou = 0

        des_only_iou50 = 0

        des_only_iou75 = 0

    # ========================================================
    # Detection IoU
    # ========================================================

    if detection_ious:

        detection_mean_iou = (

            sum(detection_ious)
            /
            len(detection_ious)

        )

        detection_iou75 = (

            sum(
                iou >= 0.75
                for iou in detection_ious
            )
            /
            len(detection_ious)

        )

    else:

        detection_mean_iou = 0

        detection_iou75 = 0

    # ========================================================
    # Risk Count
    # ========================================================

    total_gt = sum(

        len(
            gt_data[i]["gt"]["risks"]
        )

        for i in gt_data

    )

    total_pred = sum(

        len(
            pred_data[i]["prediction"]["risks"]
        )

        for i in pred_data

    )

    gt_risks_per_image = (

        total_gt / len(gt_data)

        if len(gt_data) > 0

        else 0

    )

    pred_risks_per_image = (

        total_pred / len(pred_data)

        if len(pred_data) > 0

        else 0

    )

    prediction_gt_ratio = (

        total_pred / total_gt

        if total_gt > 0

        else 0

    )

    # ========================================================
    # Model / Checkpoint
    # ========================================================

    model_name, checkpoint = (
        get_model_and_checkpoint(
            pred_file
        )
    )

    # ========================================================
    # ========================================================

    result = {

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "Model":
            model_name,

        "Checkpoint":
            checkpoint,

        "Prediction_File":
            str(pred_file),

        "Detail_File":
            str(output_detail_file),

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------

        "GT_Samples":
            len(gt_data),

        "Pred_Samples":
            len(pred_data),

        "Common_Samples":
            len(common_indices),

        "Missing_Prediction":
            len(missing_prediction),

        # ----------------------------------------------------
        # Image-level Risk
        # ----------------------------------------------------

        "Risk_GT_Positive":
            risk_tp + risk_fn,

        "Risk_Pred_Positive":
            risk_tp + risk_fp,

        "Risk_TP":
            risk_tp,

        "Risk_FP":
            risk_fp,

        "Risk_FN":
            risk_fn,

        "Risk_TN":
            risk_tn,

        "Risk_Precision":
            risk_precision,

        "Risk_Recall":
            risk_recall,

        "Risk_F1":
            risk_f1,

        # ----------------------------------------------------
        # Category + Des Semantic
        # ----------------------------------------------------

        "Semantic_GT":
            semantic_tp + semantic_fn,

        "Semantic_Pred":
            semantic_tp + semantic_fp,

        "Semantic_TP":
            semantic_tp,

        "Semantic_FP":
            semantic_fp,

        "Semantic_FN":
            semantic_fn,

        "Semantic_Precision":
            semantic_precision,

        "Semantic_Recall":
            semantic_recall,

        "Semantic_F1":
            semantic_f1,

        # ----------------------------------------------------
        # Des-only Semantic
        # ----------------------------------------------------

        "DesOnly_GT":
            des_only_tp + des_only_fn,

        "DesOnly_Pred":
            des_only_tp + des_only_fp,

        "DesOnly_TP":
            des_only_tp,

        "DesOnly_FP":
            des_only_fp,

        "DesOnly_FN":
            des_only_fn,

        "DesOnly_Precision":
            des_only_precision,

        "DesOnly_Recall":
            des_only_recall,

        "DesOnly_F1":
            des_only_f1,

        # ----------------------------------------------------
        # Semantic Localization
        # ----------------------------------------------------

        "Semantic_Matched":
            len(semantic_ious),

        "Semantic_Mean_IoU":
            semantic_mean_iou,

        "Semantic_IoU50":
            semantic_iou50,

        "Semantic_IoU75":
            semantic_iou75,

        # ----------------------------------------------------
        # Des-only Localization
        # ----------------------------------------------------

        "DesOnly_Matched":
            len(des_only_ious),

        "DesOnly_Mean_IoU":
            des_only_mean_iou,

        "DesOnly_IoU50":
            des_only_iou50,

        "DesOnly_IoU75":
            des_only_iou75,

        # ----------------------------------------------------
        # Complete Detection
        # ----------------------------------------------------

        "Detection_TP":
            detection_tp,

        "Detection_FP":
            detection_fp,

        "Detection_FN":
            detection_fn,

        "Detection_Precision":
            detection_precision,

        "Detection_Recall":
            detection_recall,

        "Detection_F1":
            detection_f1,

        "Detection_Mean_IoU":
            detection_mean_iou,

        "Detection_IoU75":
            detection_iou75,

        # ----------------------------------------------------
        # Risk Count
        # ----------------------------------------------------

        "GT_Total_Risks":
            total_gt,

        "Pred_Total_Risks":
            total_pred,

        "GT_Risks_Per_Image":
            gt_risks_per_image,

        "Pred_Risks_Per_Image":
            pred_risks_per_image,

        "Prediction_GT_Ratio":
            prediction_gt_ratio

    }

    # ========================================================
    # Category-level
    # ========================================================

    all_categories = sorted(

        set(category_gt.keys())
        |
        set(category_pred.keys())

    )

    for category in all_categories:

        tp = category_tp[
            category
        ]

        fp = category_fp[
            category
        ]

        fn = category_fn[
            category
        ]

        precision, recall, f1 = calculate_prf(

            tp,

            fp,

            fn

        )

        category_name = (
            category
            .replace(",", "_")
            .replace(" ", "_")
        )

        result[
            f"{category_name}_GT"
        ] = category_gt[
            category
        ]

        result[
            f"{category_name}_Pred"
        ] = category_pred[
            category
        ]

        result[
            f"{category_name}_TP"
        ] = tp

        result[
            f"{category_name}_Precision"
        ] = precision

        result[
            f"{category_name}_Recall"
        ] = recall

        result[
            f"{category_name}_F1"
        ] = f1

    # ========================================================
    # ========================================================

    print()
    print(
        "[RESULT]"
    )

    print(
        f"Semantic F1       : "
        f"{semantic_f1:.4f} "
        f"({semantic_f1 * 100:.2f}%)"
    )

    print(
        f"Des-only F1       : "
        f"{des_only_f1:.4f} "
        f"({des_only_f1 * 100:.2f}%)"
    )

    print(
        f"Semantic Mean IoU : "
        f"{semantic_mean_iou:.4f}"
    )

    print(
        f"Semantic IoU@0.50 : "
        f"{semantic_iou50:.4f} "
        f"({semantic_iou50 * 100:.2f}%)"
    )

    print(
        f"Detection F1      : "
        f"{detection_f1:.4f} "
        f"({detection_f1 * 100:.2f}%)"
    )

    print(
        f"GT risks          : "
        f"{total_gt}"
    )

    print(
        f"Pred risks        : "
        f"{total_pred}"
    )

    # ========================================================
    # ========================================================

    output_detail_file.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        output_detail_file,

        "w",

        encoding="utf-8"

    ) as f:

        for item in details:

            f.write(

                json.dumps(

                    item,

                    ensure_ascii=False

                )

                + "\n"

            )

    print(
        f"Detail saved      : "
        f"{output_detail_file}"
    )

    # ========================================================
    # ========================================================

    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    GT_FILE = Path(
        "../data/sample_50000_ratioS5000/test_5.0.jsonl"
    )

    PRED_FILES = [
        # ====================================================
        Path("../outputs/qwen3.5-4b-safety-lora-unsloth_ration0929c755000/checkpoint-3344/predictions_gpu0.jsonl"),
    ]
    # evaluation_details.jsonl
    OUTPUT_DETAIL_FILES = [
        Path(str(p).replace('predictions_gpu0', 'evaluation_details'))
        for p in PRED_FILES
    ]

    # ========================================================
    # ========================================================

    if len(PRED_FILES) != len(
        OUTPUT_DETAIL_FILES
    ):

        raise ValueError(

            "PRED_FILES and "
            "OUTPUT_DETAIL_FILES"

        )

    # ========================================================
    # ========================================================

    all_results = []

    for pred_file, output_detail_file in zip(

        PRED_FILES,

        OUTPUT_DETAIL_FILES

    ):

        if not pred_file.exists():

            print()
            print(
                "[WARNING] Prediction file "
                "not found:"
            )

            print(
                pred_file
            )

            continue

        result = evaluate(

            pred_file,

            output_detail_file

        )

        all_results.append(
            result
        )

    # ========================================================
    # ========================================================

    OUTPUT_CSV_FILE.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    if all_results:

        # ----------------------------------------------------
        # ----------------------------------------------------

        fieldnames = []

        for result in all_results:

            for key in result.keys():

                if key not in fieldnames:

                    fieldnames.append(
                        key
                    )

        # ----------------------------------------------------
        # ----------------------------------------------------

        with open(

            OUTPUT_CSV_FILE,

            "w",

            newline="",

            encoding="utf-8-sig"

        ) as f:

            writer = csv.DictWriter(

                f,

                fieldnames=fieldnames,

                extrasaction="ignore"

            )

            writer.writeheader()

            writer.writerows(
                all_results
            )

        print()
        print("=" * 80)
        print(
            "ALL EVALUATION RESULTS SAVED"
        )
        print("=" * 80)

        print(
            OUTPUT_CSV_FILE
        )

        print(
            f"Total evaluated models: "
            f"{len(all_results)}"
        )

        print("=" * 80)

    else:

        print()
        print(
            "[ERROR] No evaluation "
            "results generated."
        )
