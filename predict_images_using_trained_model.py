#!/usr/bin/python3
# Image Recognition Using Tensorflow Exmaple.
# Code based on example at:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/label_image.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import threading
import queue
import time
import sys
import socket
import math
# sudo apt install python3-pip
# sudo python3 -m pip install --upgrade pip
# sudo python3 -m pip install --upgrade setuptools
# sudo python3 -m pip install --upgrade tensorflow==1.15





def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def predict_image(q, sess, graph, image_bytes, img_full_path, labels, input_operation, output_operation):
    image = read_tensor_from_image_bytes(image_bytes)
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: image
    })
    results = np.squeeze(results)
    prediction = results.argsort()[-5:][::-1][0]
    q.put( {'img_full_path':img_full_path, 'prediction':labels[prediction].title(), 'percent':results[prediction]} )

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def read_tensor_from_image_bytes(imagebytes, input_height=299, input_width=299, input_mean=0, input_std=255):
    image_reader = tf.image.decode_png( imagebytes, channels=3, name="png_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)
    return result

def main():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        #TAG_IMAGE_FILE = 'apple10.png'
        #file_name = r"D:/rec_tf_final_park/TF/unknown_images/"+TAG_IMAGE_FILE
        #print('file_name', file_name)

        #net_filename = file_name.encode()
        #sock.sendto(net_filename, ("192.168.34.233", 9999))
        #print("Sending {} ...".format(file_name))

        #f = open(file_name,"rb")
        #data = f.read(1024)
        #while(data):
        #    if(sock.sendto(data,("192.168.34.233", 9999))):
        #        data = f.read(1024)
        #        time.sleep(0.02)

      
        
        # Loading the Trained Machine Learning Model created from running retrain.py on the training_images directory
        graph = load_graph('/tmp/retrain_tmp/output_graph.pb')
        labels = load_labels("/tmp/retrain_tmp/output_labels.txt")
        #graph = load_graph('D:\AI/output_graph.pb')
        #labels = load_labels("D:\AI/output_labels.txt")

        # Load up our session
        input_operation = graph.get_operation_by_name("import/Placeholder")
        output_operation = graph.get_operation_by_name("import/final_result")
        sess = tf.compat.v1.Session(graph=graph)

        # Can use queues and threading to spead up the processing
        q = queue.Queue()
 
        #첫번째 디렉토리
        unknown_images_dir = "C:/Img2"
        unknown_images = os.listdir(unknown_images_dir)

        normal_images = [file for file in unknown_images if file.startswith("predict")]###unknown_images -> normal_images


        if not normal_images:###unknown_images -> normal_images
            #print("no files in unknown_images!!!")###unknown_images -> normal_images
            print("no 'Normal.' files in unknown_images!!!")###unknown_images -> normal_images
        else:
            #Going to interate over each of our images.
            #첫번째 디렉토리 predict
            for image in normal_images:###unknown_images -> normal_images
                img_full_path = '{}/{}'.format(unknown_images_dir, image)
        
                print('Processing Image {}'.format(img_full_path))
                # We don't want to process too many images at once. 10 threads max
                while len(threading.enumerate()) > 10:
                    time.sleep(0.0001)

                #predict_image function is expecting png image bytes so we read image as 'rb' to get a bytes object
                image_bytes = open(img_full_path,'rb').read()
                threading.Thread(target=predict_image, args=(q, sess, graph, image_bytes, img_full_path, labels, input_operation, output_operation)).start()

            

            print('Waiting For Threads to Finish...')
         
            while q.qsize() < len(normal_images):###unknown_images -> normal_images
                time.sleep(0.001)
        

            #getting a list of all threads returned results
            prediction_results = [q.get() for x in range(q.qsize())]
        

            #do something with our results... Like print them to the screen.
            #for prediction in prediction_results:
            #    print('TensorFlow Predicted {img_full_path} is a {prediction} with {percent:.2%} Accuracy'.format(**prediction))

            #percent 값 send 후 img 파일 삭제
            for index, prediction in enumerate(prediction_results):
                division = prediction_results[index]['prediction'][0:1]
                #division = division[0:1]##보낼 구분자
                
                #percent = format(prediction_results[index]['percent']*100,".2f")
                percent = format(prediction_results[index]['percent']*100, ".0f")

                print("구분 : "+division)
                print("퍼센트 : "+percent)
                
                division_percent = division+percent
                
                sock.sendto(division_percent.encode() ,("127.0.0.1", 15202))
  
               
                print('TensorFlow Predicted {img_full_path} is a {prediction} with {percent:.2%} Accuracy'.format(**prediction))
                #print(index, prediction)

                os.remove(prediction_results[index]['img_full_path'])
            
            
            
if __name__ == "__main__":
    main()
