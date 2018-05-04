
import argparse

def extract_label(img_name):
    return img_name.split('_')[1]

def make_box(img_shape):
    return (str(0),str(0),img_shape.split('x')[0].strip(),img_shape.split('x')[1].strip())

def render_line(line):

    img_name, img_shape = line.split(',')
    label = extract_label(img_name).upper()
    box = make_box(img_shape)
    return "{0},{1},{2},{3},{4},{5}\n".format(img_name,box[0],box[1],box[2],box[3],label)

def filter_by_label(lst_to_write):
    allowed_labels = []
    with open("class_to_id.csv") as f:
        allowed_labels = [ line.split(",")[0] for line in f.readlines()]
    ret = []
    for el in lst_to_write:
        if el.strip().split(",")[5] in allowed_labels:
            ret.append(el)
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='render line')
    parser.add_argument('fname',type=str,help='file you want to be rendered')
        
    args = parser.parse_args()
    fname = args.fname
    
    lst_to_write = []
    with open(fname,'r') as f:
        for line in f:
            lst_to_write.append(render_line(line))
    
    lst_to_write = filter_by_label(lst_to_write)
    with open(fname,'w') as f:
        for line in lst_to_write:
            f.write(line)
