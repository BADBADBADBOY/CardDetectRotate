import os
import numpy as np

def order_points_new(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0, 1] != leftMost[1, 1]:
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    else:
        leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
    (tl, bl) = leftMost
    if rightMost[0, 1] != rightMost[1, 1]:
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    else:
        rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
    (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="float32")


if __name__ == "__main__":

    target_gt_dir = "./gt_dir/"
    ori_label_file = "./labelv2.txt"

    if not os.path.exists(target_gt_dir):
        os.mkdir(target_gt_dir)

    with open(ori_label_file, 'r', encoding='utf-8') as fid:
        line_str = ""
        for line in fid.readlines():
            line_str += line
        line_str = line_str.replace('\n', ' ').split('#')
        for item in line_str:
            item = item.split(' ')
            s_item = []
            for it in item:
                if len(it) > 0:
                    s_item.append(it)
            if len(s_item) <= 0:
                continue
            file_path = s_item[0]
            w, h = s_item[1], s_item[2]
            s_item = np.array(s_item[3:]).reshape(-1, 17).astype(np.float32).astype(np.int32)
            with open(os.path.join(target_gt_dir, file_path.split('/')[-1].split('.')[0] + '.txt'), 'w+') as f_w:
                for item in s_item:
                    x1, y1, x2, y2 = item[:4]
                    p1, p2, p3, p4, p5, p6, p7, p8 = item[7], item[8], item[10], item[11], item[13], item[14], item[4], \
                                                     item[5]
                    poly = np.array([p1, p2, p3, p4, p5, p6, p7, p8]).reshape(4, 2)
                    poly_new = order_points_new(poly)
                    if poly[0][0] == poly_new[0][0] and poly[0][1] == poly_new[0][1]:
                        ang_cls = 0
                    elif poly[0][0] == poly_new[1][0] and poly[0][1] == poly_new[1][1]:
                        ang_cls = 1
                    elif poly[0][0] == poly_new[2][0] and poly[0][1] == poly_new[2][1]:
                        ang_cls = 2
                    elif poly[0][0] == poly_new[3][0] and poly[0][1] == poly_new[3][1]:
                        ang_cls = 3

                    save_item = [x1, y1, x2, y2, p1, p2, p3, p4, p5, p6, p7, p8, str(ang_cls), '0']
                    save_item = ",".join(list(map(str, save_item))) + '\n'
                    f_w.write(save_item)