import cv2, os, random, colorsys, onnxruntime, string, time, argparse, uuid, logging
import numpy as np
from utils import Processing
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("roads")
parser.add_argument('-i',"--input", type = str, required = True, default = False, help = "path image ...")
logging.basicConfig(filename=f'log/ocr.log', filemode='w', format='%(asctime)s - %(message)s', level = logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]




class Detection(Processing):
    def __init__(self, path_model:str, path_classes:str, image_shape:list, padding:int):
        self.path_model   = path_model
        self.path_classes = path_classes
        self.session = onnxruntime.InferenceSession(self.path_model, providers = providers)
        self.class_labels, self.num_names = self.get_classes(self.path_classes)
        self.image_shape = image_shape
        self.font = ImageFont.truetype('weights/font.otf', 12)
        self.class_colors = self.colors(self.class_labels)
      
    
    def boxes_detection(self, image, size):
        ort_inputs = {self.session.get_inputs()[0].name:image, self.session.get_inputs()[1].name:size}
        box_out, scores_out, classes_out = self.session.run(None, ort_inputs)
        return box_out, scores_out, classes_out
    

    
    def draw_detection(self, image, boxes_out, scores_out, classes_out):
        image_pred = image.copy()
        for i, c in reversed(list(enumerate(classes_out))):
            draw = ImageDraw.Draw(image_pred)
            predicted_class = self.class_labels[c]
            box = boxes_out[i]
            score = scores_out[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            label = '{}: {:.2f}%'.format(predicted_class, score*100)
            print(label)
            logging.info(f'{label}')
            label_size = draw.textsize(label, self.font)
            if top - label_size[1] >= 0:
            	text_origin = np.array([left, top - label_size[1]])
            else:
            	text_origin = np.array([left, top + 1])
            draw.rectangle([left, top, right, bottom], outline= tuple(self.class_colors[c]), width=1)
            draw.text(text_origin, label, fill = (255,255,0), font = self.font)      
            del draw
        return np.array(image_pred)



    def draw_line(self, image, x, y, x1, y1, color, l = 15, t = 2):
        cv2.line(image, (x, y), (x + l, y), color, t)
        cv2.line(image, (x, y), (x, y + l), color, t)    
        cv2.line(image, (x1, y), (x1 - l, y), color, t)
        cv2.line(image, (x1, y), (x1, y + l), color, t)    
        cv2.line(image, (x, y1), (x + l, y1), color, t)
        cv2.line(image, (x, y1), (x, y1 - l), color, t)   
        cv2.line(image, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(image, (x1, y1), (x1, y1 - l), color, t)    
        return image
    

    def draw_visual(self, image, boxes_out, scores_out, classes_out, lines = True):
        image = np.array(image)
        for i, c in reversed(list(enumerate(classes_out))):
            predicted_class = self.class_labels[c]
            box = boxes_out[i]
            score = scores_out[i]
            predicted_class_label = '{}: {:.2f}%'.format(predicted_class, score*100)
            box_color = self.class_colors[c]
            box_color = list(map(int, box_color))
            box = list(map(int, box))
            y_min, x_min, y_max, x_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 1)
            if lines: self.draw_line(image, x_min, y_min, x_max, y_max, box_color)
            cv2.putText(image, predicted_class_label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, [255,255,0], 1)
        return image
    
    
    
    def __call__(self, input_image:str):
        image = Image.open(input_image)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)       
        image = self.cvtColor(image)
        image_data  = self.resize_image(image, (self.image_shape[1], self.image_shape[0]))
        image_data  = np.expand_dims(self.preprocess_input(np.array(image_data, dtype='float32')), 0)
        box_out, scores_out, classes_out = self.boxes_detection(image_data,input_image_shape)
        image_pred = self.draw_visual(image, box_out, scores_out, classes_out)
        return image_pred





if __name__ == '__main__':
	args = parser.parse_args()
	opt = {"path_model":"weights/model.onnx","path_classes":"classes.txt","image_shape":[416,416],"padding":0}
	detector = Detection(**opt)
	image_pred = detector(args.input)
	image = cv2.cvtColor(image_pred, cv2.COLOR_BGR2RGB)
	cv2.imwrite("out.jpg", image)
