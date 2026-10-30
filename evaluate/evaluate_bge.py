import json
import re
import argparse
import sys
import unicodedata
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from collections import Counter


# ============================================================
# ============================================================

GT_FILE = Path("./test_5.0S5000.jsonl")

PRED_FILE = Path("./predictions_gpu01536.jsonl") 

REPORT_FILE = Path("./evaluation_report.txt")

IOU_THRESHOLD = 0.5
SEMANTIC_MODEL = "BAAI/bge-small-zh-v1.5"
CATEGORY_SIMILARITY_THRESHOLD = 0.65
DES_SIMILARITY_THRESHOLD = 0.60
SEMANTIC_SCORE_THRESHOLD = 0.65


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

    # raw_decode
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


# ============================================================
# ============================================================

def load_ground_truth(path):

    data = {}

    with open(path, "r", encoding="utf-8") as f:

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

                        if content.get("type") == "image":
                            image_path = content["image"]

                elif msg["role"] == "assistant":

                    for content in msg["content"]:

                        if content.get("type") == "text":
                            gt_text = content["text"]

            gt = normalize_risk(
                parse_json_string(gt_text)
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

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            index = item["index"]

            prediction = normalize_risk(
                parse_json_string(
                    item.get("prediction")
                )
            )

            data[index] = {
                "image": item.get("image"),
                "prediction": prediction
            }

    return data


# ============================================================
# IoU
# ============================================================

def calculate_iou(box1, box2):

    if (
        not isinstance(box1, list)
        or not isinstance(box2, list)
        or len(box1) != 4
        or len(box2) != 4
    ):
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Semantic similarity and matching
# ============================================================

def normalize_text(text):
    """Normalize text for exact-match shortcuts and lexical fallback."""

    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


SEMANTIC_CONCEPT_PATTERNS = (
    (r"edge|floor edge|building top edge|bridge deck edge|edge", "edge"),
    (r"opening|aperture|hole", "opening"),
    (r"guardrail|safety guardrail|railing|protective facility|edge protection|protection", "protection"),
    (r"missing|gap|inadequate|incomplete|not complete|none|no|not installed|not equipped|not provided", "missing"),
    (r"safety helmet|helmet|hard hat", "safety helmet"),
    (r"safety belt|fall arrest belt", "safety belt"),
    (r"not worn|not wearing|not put on|not fastened|without wearing", "not worn"),
    (r"work platform|operating platform|platform", "work platform"),
    (r"springboard|scaffold board|footboard", "springboard"),
    (r"scaffold|steel support|support", "scaffold"),
    (r"dense mesh net|safety flat net|safety net", "safety net"),
    (r"lifting and hoisting|hoisting|lifting machinery|lifting equipment", "lifting and hoisting"),
    (r"warning zone|warning line|danger area", "warning area"),
    (r"illegal passenger carrying|lifting people|carrying people", "illegal passenger carrying"),
    (r"reflective clothing|reflective vest", "reflective clothing"),
    (r"material stacking|stacking|piling|building materials", "material stacking"),
    (r"chaotic|messy|scattered|unsorted", "chaotic stacking"),
    (r"cleanup|work completion and site cleanup|construction waste|debris", "not cleaned up"),
    (r"exposed|hardened|green coverage", "ground not hardened"),
    (r"rain shelter|rain protection measure", "rain protection measure"),
    (r"fire extinguisher|firefighting equipment", "fire extinguisher")
)


def concept_tokens(text):
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return {
        concept
        for pattern, concept in SEMANTIC_CONCEPT_PATTERNS
        if re.search(pattern, normalized)
    }


def character_similarity(text_a, text_b):
    """Dependency-free fallback using domain concepts and character n-grams."""

    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    concepts_a = concept_tokens(text_a)
    concepts_b = concept_tokens(text_b)
    shared_concepts = concepts_a & concepts_b

    concept_score = 0.0
    if len(shared_concepts) >= 2:
        concept_score = len(shared_concepts) / min(
            len(concepts_a),
            len(concepts_b)
        )

    def grams(text):
        values = list(text)
        values.extend(
            values[i] + values[i + 1]
            for i in range(len(values) - 1)
        )
        return Counter(values)

    grams_a = grams(a)
    grams_b = grams(b)
    common = set(grams_a) & set(grams_b)

    dot = sum(grams_a[key] * grams_b[key] for key in common)
    norm_a = sum(value * value for value in grams_a.values()) ** 0.5
    norm_b = sum(value * value for value in grams_b.values()) ** 0.5

    if not norm_a or not norm_b:
        return 0.0

    character_score = dot / (norm_a * norm_b)
    return max(character_score, concept_score)


class SemanticMatcher:
    """Match Chinese risk records using sentence embeddings when available."""

    def __init__(
        self,
        model_name=SEMANTIC_MODEL,
        backend="auto",
        category_threshold=CATEGORY_SIMILARITY_THRESHOLD,
        des_threshold=DES_SIMILARITY_THRESHOLD,
        score_threshold=SEMANTIC_SCORE_THRESHOLD,
        device=None,
        batch_size=16  # 64
    ):
        self.model_name = model_name
        self.backend = backend
        self.category_threshold = category_threshold
        self.des_threshold = des_threshold
        self.score_threshold = score_threshold
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.embeddings = {
            "risk_category": {},
            "des": {}
        }

    def prepare(self, gt_data, pred_data):
        """Encode every distinct field once before the evaluation loop."""

        texts = {
            "risk_category": set(),
            "des": set()
        }

        for source in (gt_data.values(), pred_data.values()):
            for item in source:
                risks = item.get("gt", item.get("prediction", {})).get(
                    "risks", []
                )
                for risk in risks:
                    texts["risk_category"].add(risk["risk_category"])
                    texts["des"].add(risk["des"])

        if self.backend == "lexical":
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            self.backend = "embedding"

            for field, values in texts.items():
                values = sorted(values)
                if not values:
                    continue

                encoded = self.model.encode(
                    values,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                self.embeddings[field] = {
                    text: vector
                    for text, vector in zip(values, encoded)
                }

        except Exception as exc:
            if self.backend == "embedding":
                raise RuntimeError(
                    "Please add sentence-transformers，"
                    "Is --model exist?"
                ) from exc

            self.backend = "lexical"
            print(
                "WARNING: sentence-transformers/model unavailable; "
                "using character n-gram fallback instead. "
                f"Reason: {exc}",
                file=sys.stderr
            )

    def similarity(self, field, text_a, text_b):
        if normalize_text(text_a) == normalize_text(text_b):
            return 1.0

        if self.backend == "embedding":
            vector_a = self.embeddings[field][text_a]
            vector_b = self.embeddings[field][text_b]
            return max(0.0, min(1.0, float(vector_a @ vector_b)))

        return character_similarity(text_a, text_b)

    def pair_score(self, gt, pred):
        category_similarity = self.similarity(
            "risk_category",
            gt["risk_category"],
            pred["risk_category"]
        )
        des_similarity = self.similarity(
            "des",
            gt["des"],
            pred["des"]
        )
        score = (
            0.35 * category_similarity
            + 0.65 * des_similarity
        )

        return category_similarity, des_similarity, score

    def semantic_candidates(self, gt_risks, pred_risks):
        candidates = []

        for pi, pred in enumerate(pred_risks):
            for gi, gt in enumerate(gt_risks):
                category_similarity, des_similarity, score = self.pair_score(
                    gt,
                    pred
                )

                if category_similarity < self.category_threshold:
                    continue

                if des_similarity < self.des_threshold:
                    continue

                if score < self.score_threshold:
                    continue

                candidates.append({
                    "pred_idx": pi,
                    "gt_idx": gi,
                    "category_similarity": category_similarity,
                    "des_similarity": des_similarity,
                    "semantic_score": score,
                    "iou": calculate_iou(
                        pred["bbox"],
                        gt["bbox"]
                    )
                })

        candidates.sort(
            key=lambda item: (
                item["semantic_score"],
                item["iou"],
                item["des_similarity"],
                item["category_similarity"]
            ),
            reverse=True
        )
        return candidates

    def semantic_match(self, gt_risks, pred_risks):
        matches = []
        used_gt = set()
        used_pred = set()

        for candidate in self.semantic_candidates(gt_risks, pred_risks):
            pi = candidate["pred_idx"]
            gi = candidate["gt_idx"]

            if pi in used_pred or gi in used_gt:
                continue

            matches.append(candidate)
            used_pred.add(pi)
            used_gt.add(gi)

        return matches

    def des_only_match(self, gt_risks, pred_risks):
        candidates = []

        for pi, pred in enumerate(pred_risks):
            for gi, gt in enumerate(gt_risks):
                des_similarity = self.similarity(
                    "des",
                    gt["des"],
                    pred["des"]
                )

                if des_similarity < self.des_threshold:
                    continue

                candidates.append({
                    "pred_idx": pi,
                    "gt_idx": gi,
                    "des_similarity": des_similarity,
                    "iou": calculate_iou(
                        pred["bbox"],
                        gt["bbox"]
                    )
                })

        candidates.sort(
            key=lambda item: (
                item["des_similarity"],
                item["iou"]
            ),
            reverse=True
        )

        matches = []
        used_pred = set()
        used_gt = set()

        for candidate in candidates:
            pi = candidate["pred_idx"]
            gi = candidate["gt_idx"]

            if pi in used_pred or gi in used_gt:
                continue

            matches.append(candidate)
            used_pred.add(pi)
            used_gt.add(gi)

        return matches

    def detection_match(self, gt_risks, pred_risks, iou_threshold=0.5):
        matches = []

        for candidate in self.semantic_match(gt_risks, pred_risks):
            if candidate["iou"] >= iou_threshold:
                matches.append(candidate)

        return matches


# ============================================================
# P / R / F1
# ============================================================

def calculate_prf(tp, fp, fn):

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
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return precision, recall, f1


# ============================================================
# ============================================================

def evaluate(
    pred_file,
    output_detail_file,
    gt_file=GT_FILE,
    matcher=None
):

    print("=" * 70)
    print("Loading data")
    print("=" * 70)

    gt_data = load_ground_truth(gt_file)
    pred_data = load_predictions(pred_file)

    if matcher is None:
        matcher = SemanticMatcher()

    matcher.prepare(gt_data, pred_data)

    print(f"GT samples       : {len(gt_data)}")
    print(f"Pred samples     : {len(pred_data)}")

    all_indices = sorted(
        set(gt_data.keys()) |
        set(pred_data.keys())
    )

    common_indices = sorted(
        set(gt_data.keys()) &
        set(pred_data.keys())
    )

    missing_prediction = sorted(
        set(gt_data.keys()) -
        set(pred_data.keys())
    )

    print(f"Common samples   : {len(common_indices)}")
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
    # Detection: Des-only + IoU
    # ========================================================

    des_only_detection_tp = 0
    des_only_detection_fp = 0
    des_only_detection_fn = 0

    # ========================================================
    # Localization
    # ========================================================

    semantic_ious = []
    des_only_ious = []

    # Category + Des + IoU
    detection_ious = []

    # Des-only + IoU
    des_only_detection_ious = []

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

            gt = gt_data[index]["gt"]

        else:

            gt = {
                "risk": False,
                "risks": []
            }

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        if index in pred_data:

            pred = pred_data[index]["prediction"]

        else:

            pred = {
                "risk": False,
                "risks": []
            }

        gt_risks = gt["risks"]
        pred_risks = pred["risks"]

        # ====================================================
        # Risk-level
        # ====================================================

        gt_has_risk = len(gt_risks) > 0
        pred_has_risk = len(pred_risks) > 0

        if gt_has_risk and pred_has_risk:

            risk_tp += 1

        elif not gt_has_risk and pred_has_risk:

            risk_fp += 1

        elif gt_has_risk and not pred_has_risk:

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
        # Category + Des Semantic Matching
        # ====================================================

        semantic_matches = matcher.semantic_match(
            gt_risks,
            pred_risks
        )

        semantic_tp_img = len(
            semantic_matches
        )

        semantic_fp_img = (
            len(pred_risks)
            - semantic_tp_img
        )

        semantic_fn_img = (
            len(gt_risks)
            - semantic_tp_img
        )

        semantic_tp += semantic_tp_img
        semantic_fp += semantic_fp_img
        semantic_fn += semantic_fn_img

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

            semantic_ious.append(iou)

            semantic_match_info.append({
                "gt_idx": m["gt_idx"],
                "pred_idx": m["pred_idx"],
                "iou": iou,
                "category_similarity": m["category_similarity"],
                "des_similarity": m["des_similarity"],
                "semantic_score": m["semantic_score"],
                "gt_risk_category": gt_risk["risk_category"],
                "pred_risk_category": pred_risk["risk_category"],
                "gt_des": gt_risk["des"],
                "pred_des": pred_risk["des"]
            })

        # ====================================================
        # Des-only Semantic Matching
        # ====================================================

        des_only_matches = matcher.des_only_match(
            gt_risks,
            pred_risks
        )

        des_only_tp_img = len(
            des_only_matches
        )

        des_only_fp_img = (
            len(pred_risks)
            - des_only_tp_img
        )

        des_only_fn_img = (
            len(gt_risks)
            - des_only_tp_img
        )

        des_only_tp += des_only_tp_img
        des_only_fp += des_only_fp_img
        des_only_fn += des_only_fn_img

        # ----------------------------------------------------
        # Des-only IoU
        # ----------------------------------------------------

        des_only_match_info = []

        for m in des_only_matches:

            des_only_ious.append(
                m["iou"]
            )

            des_only_match_info.append({
                "gt_idx": m["gt_idx"],
                "pred_idx": m["pred_idx"],
                "iou": m["iou"],
                "des_similarity": m["des_similarity"],
                "gt_des": gt_risks[m["gt_idx"]]["des"],
                "pred_des": pred_risks[m["pred_idx"]]["des"]
            })

        # ====================================================
        # Detection matching
        # ====================================================

        detection_matches = matcher.detection_match(
            gt_risks,
            pred_risks,
            IOU_THRESHOLD
        )

        detection_tp_img = len(
            detection_matches
        )

        detection_fp_img = (
            len(pred_risks)
            - detection_tp_img
        )

        detection_fn_img = (
            len(gt_risks)
            - detection_tp_img
        )

        detection_tp += detection_tp_img
        detection_fp += detection_fp_img
        detection_fn += detection_fn_img

        # ----------------------------------------------------
        # Detection IoU
        # ----------------------------------------------------

        for m in detection_matches:

            detection_ious.append(
                m["iou"]
            )

        # ====================================================
        # Des-only Detection matching
        # ====================================================
        # Matching rule:
        # semantic des + IoU >= threshold
        #
        # risk_category is completely ignored.
        # ====================================================

        des_only_detection_matches = []

        for m in des_only_matches:

            if m["iou"] >= IOU_THRESHOLD:
                des_only_detection_matches.append(m)

        des_only_detection_tp_img = len(
            des_only_detection_matches
        )

        des_only_detection_fp_img = (
                len(pred_risks)
                - des_only_detection_tp_img
        )

        des_only_detection_fn_img = (
                len(gt_risks)
                - des_only_detection_tp_img
        )

        des_only_detection_tp += (
            des_only_detection_tp_img
        )

        des_only_detection_fp += (
            des_only_detection_fp_img
        )

        des_only_detection_fn += (
            des_only_detection_fn_img
        )

        # ----------------------------------------------------
        # Des-only Detection IoU
        # ----------------------------------------------------

        for m in des_only_detection_matches:
            des_only_detection_ious.append(
                m["iou"]
            )

        # ====================================================
        # Category PRF
        # ====================================================

        # category + des semantic matching
        matched_gt_category = set()
        matched_pred_category = set()

        for m in semantic_matches:

            gt_risk = gt_risks[
                m["gt_idx"]
            ]

            pred_risk = pred_risks[
                m["pred_idx"]
            ]

            category = gt_risk[
                "risk_category"
            ]

            category_tp[category] += 1

            matched_gt_category.add(
                m["gt_idx"]
            )

            matched_pred_category.add(
                m["pred_idx"]
            )

        for gi, r in enumerate(gt_risks):

            if gi not in matched_gt_category:

                category_fn[
                    r["risk_category"]
                ] += 1

        for pi, r in enumerate(pred_risks):

            if pi not in matched_pred_category:

                category_fp[
                    r["risk_category"]
                ] += 1

        # ====================================================
        # Detail
        # ====================================================

        detail = {
            "index": index,

            "image": gt_data.get(
                index,
                {}
            ).get("image"),

            "gt_num": len(gt_risks),
            "pred_num": len(pred_risks),

            # category + des
            "semantic_tp": semantic_tp_img,
            "semantic_fp": semantic_fp_img,
            "semantic_fn": semantic_fn_img,

            # des only
            "des_only_tp": des_only_tp_img,
            "des_only_fp": des_only_fp_img,
            "des_only_fn": des_only_fn_img,

            # detection: category + des + IoU
            "detection_tp": detection_tp_img,
            "detection_fp": detection_fp_img,
            "detection_fn": detection_fn_img,

            # detection: des-only + IoU
            "des_only_detection_tp": des_only_detection_tp_img,
            "des_only_detection_fp": des_only_detection_fp_img,
            "des_only_detection_fn": des_only_detection_fn_img,

            "semantic_matches":
                semantic_match_info,

            "des_only_matches":
                des_only_match_info,

            "detection_matches":
                detection_matches,

            "des_only_detection_matches":
                des_only_detection_matches,

            "gt_risks": gt_risks,
            "pred_risks": pred_risks
        }

        details.append(detail)

    # ========================================================
    # Metrics
    # ========================================================

    risk_precision, risk_recall, risk_f1 = calculate_prf(
        risk_tp,
        risk_fp,
        risk_fn
    )

    semantic_precision, semantic_recall, semantic_f1 = calculate_prf(
        semantic_tp,
        semantic_fp,
        semantic_fn
    )

    des_only_precision, des_only_recall, des_only_f1 = calculate_prf(
        des_only_tp,
        des_only_fp,
        des_only_fn
    )

    detection_precision, detection_recall, detection_f1 = calculate_prf(
        detection_tp,
        detection_fp,
        detection_fn
    )

    des_only_detection_precision, des_only_detection_recall, des_only_detection_f1 = calculate_prf(
        des_only_detection_tp,
        des_only_detection_fp,
        des_only_detection_fn
    )

    # ========================================================
    # Localization metrics
    # ========================================================

    if semantic_ious:

        semantic_mean_iou = (
            sum(semantic_ious)
            / len(semantic_ious)
        )

        semantic_iou50 = (
            sum(
                iou >= 0.5
                for iou in semantic_ious
            )
            / len(semantic_ious)
        )

        semantic_iou75 = (
            sum(
                iou >= 0.75
                for iou in semantic_ious
            )
            / len(semantic_ious)
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
            / len(des_only_ious)
        )

        des_only_iou50 = (
            sum(
                iou >= 0.5
                for iou in des_only_ious
            )
            / len(des_only_ious)
        )

        des_only_iou75 = (
            sum(
                iou >= 0.75
                for iou in des_only_ious
            )
            / len(des_only_ious)
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
            / len(detection_ious)
        )

        detection_iou75 = (
            sum(
                iou >= 0.75
                for iou in detection_ious
            )
            / len(detection_ious)
        )

    else:

        detection_mean_iou = 0
        detection_iou75 = 0

    # ========================================================
    # Des-only Detection IoU
    # ========================================================

    if des_only_detection_ious:

        des_only_detection_mean_iou = (
                sum(des_only_detection_ious)
                / len(des_only_detection_ious)
        )

        des_only_detection_iou75 = (
                sum(
                    iou >= 0.75
                    for iou in des_only_detection_ious
                )
                / len(des_only_detection_ious)
        )

    else:

        des_only_detection_mean_iou = 0
        des_only_detection_iou75 = 0

    # ========================================================
    # ========================================================

    print()
    print("=" * 70)
    print("                    EVALUATION RESULT")
    print("=" * 70)

    # ========================================================
    # 1 Risk-level
    # ========================================================

    print()
    print("[1] IMAGE-LEVEL RISK")
    print("-" * 70)

    print(
        f"GT positive          : "
        f"{risk_tp + risk_fn}"
    )

    print(
        f"Pred positive        : "
        f"{risk_tp + risk_fp}"
    )

    print(
        f"TP                   : {risk_tp}"
    )

    print(
        f"FP                   : {risk_fp}"
    )

    print(
        f"FN                   : {risk_fn}"
    )

    print(
        f"TN                   : {risk_tn}"
    )

    print(
        f"Precision            : "
        f"{risk_precision:.4f}"
    )

    print(
        f"Recall               : "
        f"{risk_recall:.4f}"
    )

    print(
        f"F1                   : "
        f"{risk_f1:.4f}"
    )

    # ========================================================
    # 2 Category + Des Semantic
    # ========================================================

    print()
    print("[2] SEMANTIC RECOGNITION")
    print("-" * 70)

    print(
        "Matching rule: semantic similarity of risk_category + des"
    )

    print(
        f"Backend              : {matcher.backend}"
    )

    print(
        f"Thresholds           : category >= {matcher.category_threshold:.2f}, "
        f"des >= {matcher.des_threshold:.2f}, "
        f"combined >= {matcher.score_threshold:.2f}"
    )

    print(
        f"GT risks             : "
        f"{semantic_tp + semantic_fn}"
    )

    print(
        f"Pred risks           : "
        f"{semantic_tp + semantic_fp}"
    )

    print(
        f"TP                   : "
        f"{semantic_tp}"
    )

    print(
        f"FP                   : "
        f"{semantic_fp}"
    )

    print(
        f"FN                   : "
        f"{semantic_fn}"
    )

    print(
        f"Precision            : "
        f"{semantic_precision:.4f} "
        f"({semantic_precision * 100:.2f}%)"
    )

    print(
        f"Recall               : "
        f"{semantic_recall:.4f} "
        f"({semantic_recall * 100:.2f}%)"
    )

    print(
        f"F1                   : "
        f"{semantic_f1:.4f} "
        f"({semantic_f1 * 100:.2f}%)"
    )

    # ========================================================
    # 3 Des-only Semantic
    # ========================================================

    print()
    print("[3] DES-ONLY SEMANTIC RECOGNITION")
    print("-" * 70)

    print(
        "Matching rule: semantic similarity of des only"
    )

    print(
        "risk_category and bbox are ignored; IoU is used only as a tie-breaker."
    )

    print(
        f"GT risks             : "
        f"{des_only_tp + des_only_fn}"
    )

    print(
        f"Pred risks           : "
        f"{des_only_tp + des_only_fp}"
    )

    print(
        f"TP                   : "
        f"{des_only_tp}"
    )

    print(
        f"FP                   : "
        f"{des_only_fp}"
    )

    print(
        f"FN                   : "
        f"{des_only_fn}"
    )

    print(
        f"Precision            : "
        f"{des_only_precision:.4f} "
        f"({des_only_precision * 100:.2f}%)"
    )

    print(
        f"Recall               : "
        f"{des_only_recall:.4f} "
        f"({des_only_recall * 100:.2f}%)"
    )

    print(
        f"F1                   : "
        f"{des_only_f1:.4f} "
        f"({des_only_f1 * 100:.2f}%)"
    )

    # ========================================================
    # 4 Localization
    # ========================================================

    print()
    print("[4] LOCALIZATION")
    print("-" * 70)

    print(
        "Localization is calculated "
        "on semantic matches."
    )

    print()
    print("[Category + Des]")
    print(
        f"Matched              : "
        f"{len(semantic_ious)}"
    )

    print(
        f"Mean IoU             : "
        f"{semantic_mean_iou:.4f}"
    )

    print(
        f"IoU@0.50             : "
        f"{semantic_iou50:.4f} "
        f"({semantic_iou50 * 100:.2f}%)"
    )

    print(
        f"IoU@0.75             : "
        f"{semantic_iou75:.4f} "
        f"({semantic_iou75 * 100:.2f}%)"
    )

    print()
    print("[Des Only]")
    print(
        f"Matched              : "
        f"{len(des_only_ious)}"
    )

    print(
        f"Mean IoU             : "
        f"{des_only_mean_iou:.4f}"
    )

    print(
        f"IoU@0.50             : "
        f"{des_only_iou50:.4f} "
        f"({des_only_iou50 * 100:.2f}%)"
    )

    print(
        f"IoU@0.75             : "
        f"{des_only_iou75:.4f} "
        f"({des_only_iou75 * 100:.2f}%)"
    )

    # ========================================================
    # 5 Detection
    # ========================================================

    print()
    print("[5] COMPLETE DETECTION")
    print("-" * 70)

    print(
        "Matching rule:"
    )

    print(
        "semantic category + semantic des + IoU >= "
        f"{IOU_THRESHOLD}"
    )

    print(
        f"TP                   : "
        f"{detection_tp}"
    )

    print(
        f"FP                   : "
        f"{detection_fp}"
    )

    print(
        f"FN                   : "
        f"{detection_fn}"
    )

    print(
        f"Precision            : "
        f"{detection_precision:.4f} "
        f"({detection_precision * 100:.2f}%)"
    )

    print(
        f"Recall               : "
        f"{detection_recall:.4f} "
        f"({detection_recall * 100:.2f}%)"
    )

    print(
        f"F1                   : "
        f"{detection_f1:.4f} "
        f"({detection_f1 * 100:.2f}%)"
    )

    print()
    print(
        f"Mean IoU             : "
        f"{detection_mean_iou:.4f}"
    )

    print(
        f"IoU@0.75             : "
        f"{detection_iou75:.4f}"
    )

    # ========================================================
    # 6 Complete Detection without Category
    # ========================================================

    print()
    print("[6] COMPLETE DETECTION (DES-ONLY)")
    print("-" * 70)

    print(
        "Matching rule:"
    )

    print(
        "semantic des + IoU >= "
        f"{IOU_THRESHOLD}"
    )

    print(
        "risk_category is ignored."
    )

    print(
        f"TP                   : "
        f"{des_only_detection_tp}"
    )

    print(
        f"FP                   : "
        f"{des_only_detection_fp}"
    )

    print(
        f"FN                   : "
        f"{des_only_detection_fn}"
    )

    print(
        f"Precision            : "
        f"{des_only_detection_precision:.4f} "
        f"({des_only_detection_precision * 100:.2f}%)"
    )

    print(
        f"Recall               : "
        f"{des_only_detection_recall:.4f} "
        f"({des_only_detection_recall * 100:.2f}%)"
    )

    print(
        f"F1                   : "
        f"{des_only_detection_f1:.4f} "
        f"({des_only_detection_f1 * 100:.2f}%)"
    )

    print()

    print(
        f"Matched              : "
        f"{len(des_only_detection_ious)}"
    )

    print(
        f"Mean IoU             : "
        f"{des_only_detection_mean_iou:.4f}"
    )

    print(
        f"IoU@0.75             : "
        f"{des_only_detection_iou75:.4f}"
    )

    # ========================================================
    # 7 Risk count
    # ========================================================

    total_gt = sum(
        len(gt_data[i]["gt"]["risks"])
        for i in gt_data
    )

    total_pred = sum(
        len(pred_data[i]["prediction"]["risks"])
        for i in pred_data
    )

    print()
    print("[7] RISK COUNT")
    print("-" * 70)

    print(
        f"GT total             : {total_gt}"
    )

    print(
        f"Prediction total     : {total_pred}"
    )

    print(
        f"GT / image           : "
        f"{total_gt / len(gt_data):.2f}"
    )

    print(
        f"Pred / image         : "
        f"{total_pred / len(pred_data):.2f}"
    )

    print(
        f"Prediction / GT      : "
        f"{total_pred / total_gt:.4f}"
    )

    # ========================================================
    # 8 Category
    # ========================================================

    print()
    print("[8] CATEGORY-LEVEL SEMANTIC RESULTS")
    print("-" * 70)

    all_categories = sorted(
        set(category_gt.keys()) |
        set(category_pred.keys())
    )

    print(
        f"{'Category':<25}"
        f"{'GT':>8}"
        f"{'Pred':>8}"
        f"{'TP':>8}"
        f"{'Prec':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print("-" * 85)

    for category in all_categories:

        tp = category_tp[category]
        fp = category_fp[category]
        fn = category_fn[category]

        precision, recall, f1 = calculate_prf(
            tp,
            fp,
            fn
        )

        print(
            f"{category:<25}"
            f"{category_gt[category]:>8}"
            f"{category_pred[category]:>8}"
            f"{tp:>8}"
            f"{precision:>10.3f}"
            f"{recall:>10.3f}"
            f"{f1:>10.3f}"
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

    print()
    print("=" * 70)

    print(
        "Detailed results saved to:"
    )

    print(
        output_detail_file
    )

    print("=" * 70)


# ============================================================
# main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate construction-site risks with semantic matching."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=GT_FILE,
        help=f"Ground-truth JSONL (default: {GT_FILE})"
    )
    parser.add_argument(
        "--pred",
        type=Path,
        nargs="+",
        default=[PRED_FILE],
        help="One or more prediction JSONL files."
    )
    parser.add_argument(
        "--output",
        type=Path,
        nargs="*",
        default=None,
        help="Output JSONL files; omitted means evaluation_details.jsonl beside each prediction."
    )
    parser.add_argument(
        "--model",
        default=SEMANTIC_MODEL,
        help=f"SentenceTransformer model name or local path (default: {SEMANTIC_MODEL})"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "embedding", "lexical"),
        default="auto",
        help="Semantic backend. auto falls back to character n-grams if needed."
    )
    parser.add_argument(
        "--category-threshold",
        type=float,
        default=CATEGORY_SIMILARITY_THRESHOLD
    )
    parser.add_argument(
        "--des-threshold",
        type=float,
        default=DES_SIMILARITY_THRESHOLD
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=SEMANTIC_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--device",
        default=None,
        help="SentenceTransformer device, for example cpu or cuda."
    )
    return parser.parse_args()


def resolve_output_files(pred_files, output_files):
    if output_files is None or not output_files:
        return [pred_file.parent / "evaluation_details.jsonl" for pred_file in pred_files]

    if len(output_files) == len(pred_files):
        return output_files

    if len(output_files) == 1:
        return [output_files[0] for _ in pred_files]

    raise ValueError(
        "--output must contain either one path or the same number of paths as --pred."
    )


if __name__ == "__main__":
    args = parse_args()
    output_files = resolve_output_files(args.pred, args.output)

    for pred_file, output_detail_file in zip(args.pred, output_files):
        print()
        print()
        print("#" * 80)
        print(f"Evaluating: {pred_file}")
        print("#" * 80)

        matcher = SemanticMatcher(
            model_name=args.model,
            backend=args.backend,
            category_threshold=args.category_threshold,
            des_threshold=args.des_threshold,
            score_threshold=args.score_threshold,
            device=args.device
        )

        evaluate(
            pred_file,
            output_detail_file,
            gt_file=args.gt,
            matcher=matcher
        )

