import numpy as np
import collections
import cv2
def getColorList():
    dict = collections.defaultdict(list)
    '''蓝色色'''
    lower_green = np.array([69, 116, 69])
    upper_green = np.array([130, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['blue'] = color_list
    '''红色'''
    lower_blue = np.array([164, 43, 46])
    upper_blue = np.array([255, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['red'] = color_list
    '''白色'''
    lower_white = np.array([200, 155, 176])
    upper_white = np.array([255, 255, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    '''绿色'''
    lower_red = np.array([0, 0, 35])
    upper_red = np.array([69, 101, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['green'] = color_list
    return dict
def get_color(frame):
    print('go in get_color')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts ,hierarchy= cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d
    turn_green_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    import random
    strings = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    random_str = random.sample(strings, 6)
    mingzi=r'D:\python\2021828yolov5\yolov5\hsv_color/%s.jpg',color+random_str
    cv2.imwrite(mingzi, turn_green_img)
    return color
if __name__ == '__main__':
    # filename = 'image.png'
    # filename = 'redcar.jpg'
    # filename = 'bluecar.jpg'
    # frame = cv2.imread(filename)
    # print(get_color(frame))
    pass