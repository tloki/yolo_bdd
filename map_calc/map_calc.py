import math

import numpy as np
import torch


class MAPCalculator:

    def __init__(self, min_overlap=0.5, classes=None, prune_decimals=2):

        # classes

        self.min_overlap = min_overlap
        self.classes = classes
        self.preds = []
        self.gts = []

        # self.print = not no_print

        self.gt_counter_per_class = {}
        self.gt_counter_images_per_class = {}

        self.pair_count = 0

        self.GTS = []
        self.PRDS = dict()
        self.already_seen_classes_gt = []
        self.gt_classes = None
        self.n_classes = None

        self.prune_decimals = prune_decimals

        self.predicteds = dict()

    def calculate(self, print_=True):
        report = ""
        gt_classes = list(self.gt_counter_per_class.keys())
        self.gt_classes = sorted(gt_classes)
        self.n_classes = len(gt_classes)

        for class_name in self.gt_classes:
            if class_name in self.PRDS:
                self.PRDS[class_name].sort(key=lambda x: float(x['confidence']), reverse=True)

        sum_AP = 0.0
        ap_dictionary = {}
        lamr_dictionary = {}

        count_true_positives = {}

        for class_index, class_name in enumerate(self.gt_classes):
            count_true_positives[class_name] = 0

            if class_name in self.PRDS:
                dr_data = self.PRDS[class_name]
            else:
                dr_data = []

            nd = len(dr_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd

            for idx, detection in enumerate(dr_data):
                file_id = detection["id"]
                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                # gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = self.GTS[file_id]
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = detection["bbox"]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = obj["bbox"]
                        # L U R D
                        # 0 1 2 3
                        bi = [max(bb[0], bbgt[0]), min(bb[1], bbgt[1]), min(bb[2], bbgt[2]), max(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[1] - bi[3] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[1] - bb[3] + 1) + \
                                 (bbgt[2] - bbgt[0] + 1) * (bbgt[1] - bbgt[3] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign detection as true positive/don't care/false positive
                # set minimum overlap
                min_overlap = self.min_overlap
                # if specific_iou_flagged:
                #     if class_name in specific_iou_classes:
                #         index = specific_iou_classes.index(class_name)
                #         min_overlap = float(iou_list[index])
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

            cum_sum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cum_sum
                cum_sum += val
            cum_sum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cum_sum
                cum_sum += val
            # print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_name]
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            # print(prec)

            ap, mean_recall, mean_precision = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(ap * 100) + " = " + str(class_name) + " AP "
            # class_name + " AP = {0:.2f}%".format(ap*100)
            """
             Write to results.txt
            """
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]

            class_report = text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n\n"
            report += class_report
            if print_:
                print(class_report)

            ap_dictionary[class_name] = ap

            n_images = self.gt_counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

        mAP = sum_AP / len(self.classes)
        text = "mAP = {0:.3f}%".format(mAP * 100)
        report += text
        if print_:
            print(text)
        return mAP, report

    def add_gt_pred_pair(self, gts_single, preds_single):
        if len(gts_single) == 0 and len(preds_single) == 0:
            return



        if self.prune_decimals is not None:
            if len(gts_single) > 0:
                gts_single = torch.round(gts_single * 10 ** self.prune_decimals) / (10 ** self.prune_decimals)
            if len(preds_single) > 0:
                preds_single = torch.round(preds_single * 10 ** self.prune_decimals) / (10 ** self.prune_decimals)

        # expects x, y, width, height, confidence, class index

        gt_collector = list()
        # gt_collector["bbox"] = list()
        # gt_collector["class"] = list()
        # gt_collector["used"] = list()

        already_seen_classes = []

        for gt in gts_single:
            # is_difficult = False

            l, d, w, h = gt[0:4]
            r = l + w
            u = d + h

            class_index = round(float(gt[5]))

            if self.classes is not None:
                try:
                    class_name = self.classes[class_index]
                except IndexError:
                    print("lel")
                    exit(-1)
            else:
                class_name = class_index

            if class_name in self.gt_counter_per_class:
                self.gt_counter_per_class[class_name] += 1
            else:
                self.gt_counter_per_class[class_name] = 1

            # gt_collector["bbox"].append((l, u, r, d))
            # gt_collector["class"].append(class_name)
            # gt_collector["used"].append(False)

            gt_collector.append({"class_name": class_name, "bbox": (l, u, r, d), "used": False})

            if class_name not in already_seen_classes:
                if class_name in self.gt_counter_images_per_class:
                    self.gt_counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    self.gt_counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)

        self.GTS.append(gt_collector)

        pred_collector = list()
        # pred_collector['confidence'] = []
        # pred_collector['bbox'] = []
        # pred_collector['id'] = []

        # TODO: optimize?
        for class_index, class_name in enumerate(self.classes):
            for pred in preds_single:
                l, d, w, h, confidence, pred_class_index = pred[0:6]
                pred_class_index = round(float(pred_class_index))
                r = l + w
                u = d + h

                if class_index == pred_class_index:
                    # pred_collector['confidence'].append(confidence)
                    # pred_collector['bbox'].append((l, u, r, d))
                    # pred_collector['id'].append(self.pair_count)
                    if not class_name in self.PRDS:
                        self.PRDS[class_name] = []

                    self.PRDS[class_name].append({"confidence": confidence,
                                                  "id": self.pair_count,
                                                  "bbox": (l, u, r, d)})

        # self.PRDS.append(pred_collector)
        self.pair_count += 1

        # TODO: sort


bounding_boxes = []


def log_average_miss_rate(precision, fp_cum_sum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fp_pi = 0
        return lamr, mr, fp_pi

    fp_pi = fp_cum_sum / float(num_images)
    mr = (1 - precision)

    fp_pi_tmp = np.insert(fp_pi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fp_pi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fp_pi


"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre
