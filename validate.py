import pandas as pd

def parse_segment(segment):
    start, end = map(int, segment.split("-"))
    return start, end


def iou(segment_q, segment_t):
    start_q, stop_q = parse_segment(segment_q)
    start_t, stop_t = parse_segment(segment_t)
    
    intersection_start = max(start_q, start_t)
    intersection_end = min(stop_q, stop_t)

    intersection_length = max(0, intersection_end - intersection_start)
    union_length = (stop_q - start_q) + (stop_t - start_t) - intersection_length

    iou = intersection_length / union_length if union_length > 0 else 0
    return iou


def f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_res = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1_res

def final_metric(f, final_iou):
    return 2 * (final_iou * f) / (final_iou + f + 1e-6)

def vaildate(result_path, target_path):
    orig, target = pd.read_csv(target_path, index_col=0), pd.read_csv(result_path, index_col=0)

    orig_dict = orig.groupby(['ID-piracy', 'ID-license']).count().to_dict()['SEG-piracy']
    target_dict = target.groupby(['ID-piracy', 'ID-license']).count().to_dict()['SEG-piracy']

    fn, fp, tp = 0, 0, 0
    ious = []

    for ids, count in orig_dict.items():
        if ids not in target_dict:
            fn += count # модель не нашла что то из оригинальной таблицы
        elif target_dict[ids] > count:
            fp += target_dict[ids] - count # модель нашла больше совпадений чем в оригинальной таблице
            tp += min(target_dict[ids], count) # тогда для истинных совпадений совпадений берем наименьшее количество
        elif target_dict[ids] < count:
            fn += count - target_dict[ids] # модель нашла меньше совпадений чем в оригинальной таблице
            tp += min(target_dict[ids], count) # тогда для истинных совпадений совпадений берем наименьшее количество
        else:
            tp += count

    for ids, count in target_dict.items():
        if ids not in orig_dict:
            fp += count # модель нашла то, чего не было в оригинальной таблице

    # Подсчет IOU для каждой отдельной строки из orig   
    for i, row in orig.iterrows():
        max_iou = 0
        merged = pd.merge(
            row.to_frame().T,
            target,
            'left',
            left_on=['ID-piracy', 'ID-license'],
            right_on = ['ID-piracy', 'ID-license']
        ).dropna()
        
        # Выбор наилучшего IOU по всем совпадениям из target
        if len(merged) > 0:
            for j, row1 in merged.iterrows():
                final_iou = iou(row1['SEG-piracy_x'], row1['SEG-piracy_y']) * iou(row1['SEG-license_x'], row1['SEG-license_y'])
                if final_iou > max_iou:
                    max_iou = final_iou
        
        ious.append(max_iou)

    iou_final = sum(ious) / (len(ious) + fp)
    precision, recall, f1_res = f1(tp, fp, fn)
    metric_final = final_metric(f1_res, iou_final)

    return {'Precision': precision,
            'Recall': recall,
            'F1': f1_res,
            'IOU': iou_final,
            'Metric': metric_final}
