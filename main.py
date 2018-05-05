import cv2
from moviepy.editor import VideoFileClip
import re
from os.path import basename, abspath
from copy import copy, deepcopy
import glob


srcdir = './src/'
dstdir = './dst/'
dstdir_img = dstdir + 'images/'
dstdir_ann = dstdir + 'annots/'

file_format = '{0:05d}'
absolute_paths = False

path_annotation_template='annotation_template.xml'
path_annotation_template_object='annotation_template_object.xml'

defaults = {
    'folder': 'drones',
    'database': 'cvpr15',
    'width': 640,
    'height': 480,
    'depth': 3,
    'segmented': 0
}
mandatory = ['filename','path','objects']

defaults_object = {
    'name': 'drone',
    'pose': 'Unspecified',
    'truncated': 0,
    'difficult': 0
}
mandatory_object = ['xmin','ymin','xmax','ymax']


def main():
    # load annotation templates
    template = load_file(path_annotation_template)
    template_object = load_file(path_annotation_template_object)
    
    index_path_list = glob.glob(srcdir + '*.txt')
    for index_path in index_path_list:
        name = basename(index_path).split('.')[0]
        print(name, index_path)
        index = load_original_index(index_path)
        # add 'path' and filter empty
        
        video_src = srcdir + name + '.avi'
        image_dst = dstdir_img + name + '-' + file_format + '.jpeg'
        index = extract_images_from_video(video_src, image_dst, index=index,  debug=True)
        
        ann_dst = dstdir_ann + name + '-' + file_format + '.xml'
        write_yolo_annotations(template,template_object,index, ann_dst)
    
    
def write_yolo_annotations(template, template_object, index, ann_dst):
    for key in index:
        entry = index[key]
        entry_txt = ''
        #print(entry)
        boxes = entry['boxes']
        if len(boxes) > 0:
            objects_txt = ''
            for box in boxes:
                #print('compose template: object')
                xml = compose_template(template_object, box, defaults_object, mandatory_object)
                if xml is None:
                    objects_txt = None
                else:
                    objects_txt += xml + '\n'
            
            if objects_txt is not None:
                entry['objects'] = objects_txt
                #print('compose template: main')
                xml = compose_template(template, entry, defaults, mandatory)
                if xml is not None:
                    #print(entry['id'], ann_dst)
                    path = ann_dst.format(entry['id'])
                    write_file(path, xml)

def compose_template(template, params, defaults, mandatory):
    param_list = copy(defaults)
    for key in params:
        param_list[key] = params[key]
    for mandatory_key in mandatory:
        if mandatory_key not in param_list:
            return None
    #print(template)
    #print(param_list)
    return template.format(**param_list)
    
    

def load_original_index(filename):
    index_content = load_file(filename)
    
    p = re.compile('time_layer: (\d+) detections:(.*)\n')
    matches=p.findall(index_content)
    #print(matches)
    
    index = {}
    for frame_id, boxes in matches:
        frame_id = int(frame_id)
        #print(frame_id)
        boxes = re.sub(r'[^0-9(),]', '', boxes)
        if len(boxes) == 0:
            index[frame_id] = []
        else:
            if boxes.endswith(','): boxes = boxes[:-1]
            boxes = re.sub(r'\),\(', '|', boxes)
            boxes = re.sub(r'[()]', '', boxes)
            boxes = boxes.split('|')
            for i,box in enumerate(boxes):
                boxes[i] = box.split(',')
                boxes[i] = list(map(lambda x:int(x), boxes[i]))
            
        index[frame_id] = { 'id':frame_id, 'boxes': boxes }
    return index

def extract_images_from_video(filename, dst_img, index, debug=False):
    idx = {}
    name = str(basename(filename).split('.')[0])
    clip = VideoFileClip(filename)
    max_frame = max(index.keys())
    print('max_frame', max_frame)
    count = 1
    for frame in clip.iter_frames():
        if count in index:
            boxes = index[count]['boxes']
            if len(boxes) == 0:
                print("skip:",count)
            else:
                idx[count] = index[count]
                if debug:
                    yolo_boxes = []
                    for box in boxes:
                        # TODO: verify point order in annotation !!!
                        y1, y2, x1, x2 = box
                        xmin,ymin,xmax,ymax = min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)
                        yolo_box = {
                            'xmin':xmin,
                            'xmax':xmax,
                            'ymin':ymin,
                            'ymax':ymax
                        }
                        yolo_boxes.append(yolo_box)
                        p1,p2 = (x1,y1),(x2,y2)
                        
                        #print(p1,p2)
                        cv2.rectangle(frame,p1,p2,(0,255,0))
                    idx[count]['boxes'] = yolo_boxes
                
                path = dst_img.format(count)
                idx[count]['filename'] = basename(path)
                idx[count]['path'] = abspath(path) if absolute_paths else path
                cv2.imwrite(path, frame)
           
        #print(frame.shape)
        count += 1
    print(name,count)
    return idx


def load_file(filename):
    with open(filename) as f:
        return f.read()

def write_file(filename,content):
    with open(filename,'w') as f:
        return f.write(content)


if __name__ == "__main__":
    main()
   
