# coding=utf-8
import os
from PIL import Image, ImageDraw


def _filter_out_china_plate():
    """
    过滤掉国内车牌,并且把剩余的车牌存储到指定目录
    """
    cur_path = os.path.dirname(__file__)
    src_dir = os.path.join(cur_path, 'test', 'src')
    out_dir = os.path.join(cur_path, 'test', 'test_filter_china_plate')
    all_plates = os.listdir(src_dir)
    for plate in all_plates:
        isChinaPlate = _has_chinese(plate)
        if isChinaPlate:
            print('ignore china plate: {}'.format(plate))
            continue
        src_plate_path = os.path.join(src_dir, plate)
        out_plate_path = os.path.join(out_dir, plate)
        print('src_plate_path: {} - out_plate_path: {}'.format(src_plate_path, out_plate_path))
        Image.open(src_plate_path).save(out_plate_path)


def _calculate_china_count():
    cur_path = os.path.dirname(__file__)
    # src_dir = os.path.join(cur_path, 'test', 'src')
    src_dir = '/home/duany049/data/CarPlate/my_plate'
    all_plates = os.listdir(src_dir)
    count = 0
    for plate in all_plates:
        isChinaPlate = _has_chinese(plate)
        if isChinaPlate:
            count += 1
            print('cur china plate count: {}'.format(count))
    print('count of china plate: {}'.format(count))


def _copy_plate():
    cur_path = os.path.dirname(__file__)
    src_dir = os.path.join(cur_path, 'my_plate')
    out_dir = os.path.join(cur_path, 'plate_tem')
    all_plates = os.listdir(src_dir)
    all_plates_end = all_plates[160000:231000]
    for plate in all_plates_end:
        isChinaPlate = _has_chinese(plate)
        if isChinaPlate:
            print('ignore china plate: {}'.format(plate))
            src_plate_path = os.path.join(src_dir, plate)
            out_plate_path = os.path.join(out_dir, plate)
            Image.open(src_plate_path).save(out_plate_path)


def _has_chinese(plate):
    for element in plate:
        if '\u4e00' <= element <= '\u9fff':
            return True
    return False


if __name__ == '__main__':
    _copy_plate()
