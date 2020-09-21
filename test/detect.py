
import tensorflow as tf
print("start setting the gpu")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("continue importing")
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
# flags.DEFINE_string('weights', './data/yolov4.weights', 'pretrained weights')
flags.DEFINE_string('weights', None, None)
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/1_1.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

class Detector:
    
    def __init__(self,arg):
        self.im_data = []
        
        

        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = FLAGS.size
        image_path = FLAGS.image
    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
    
        images_data = []
        for i in range(1):
            self.im_data.append(image_data)
            self.im_data = np.asarray(images_data).astype(np.float32)

    def train(self):
        trainset = Dataset(FLAGS, is_training=True)
        testset = Dataset(FLAGS, is_training=False)
        logdir = "./data/log"
        isfreeze = False
        steps_per_epoch = len(trainset)
        first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
        total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
        input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
        feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
        if FLAGS.tiny:
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        else:
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                elif i == 1:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)

        if FLAGS.weights == None:
            print("Training from scratch")
        else:
            if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
            else:
                model.load_weights(FLAGS.weights)
                print('Restoring weights from: %s ... ' % FLAGS.weights)
        optimizer = tf.keras.optimizers.RMSprop(
                        learning_rate=0.002,
                        rho=0.9,
                        momentum=0.0,
                        epsilon=1e-07,
                        centered=False,
                        name="RMSprop")
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        writer = tf.summary.create_file_writer(logdir)

        def train_step(image_data, target):
            with tf.GradientTape() as tape:
                pred_result = model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

            # optimizing process
                for i in range(len(freeze_layers)):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)                        giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]
                total_loss = giou_loss + conf_loss + prob_loss
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                             "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
                    # update learning rate
                global_steps.assign_add(1)
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
                else:
                    lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * ((1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
                optimizer.lr.assign(lr.numpy())

                    # writing summary data
                with writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                    tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                    tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                    tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                writer.flush()
                
        def test_step(image_data, target):
            with tf.GradientTape() as tape:
                pred_result = model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

            # optimizing process
                for i in range(len(freeze_layers)):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]
                total_loss = giou_loss + conf_loss + prob_loss
                tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                         "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))
        for epoch in range(first_stage_epochs + second_stage_epochs):
            if epoch < first_stage_epochs:
                if not isfreeze:
                    isfreeze = True
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
            elif epoch >= first_stage_epochs:
                if isfreeze:
                    isfreeze = False
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        unfreeze_all(freeze)
            for image_data, target in trainset:
                train_step(image_data, target)

            for image_data, target in testset:
                test_step(image_data, target)
            tf.saved_model.save(model, './checkpoints/yolov4')
            
    def detect(self):
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
            box = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4))
            score = tf.reshape(pred_conf, (tf.shape(pred_conf)[0], box.shape[1], -1))
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(boxes=box,
                                                                                         scores=score,
                                                                                         max_output_size_per_class=50,
                                                                                         max_total_size=50,
                                                                                         iou_threshold=FLAGS.iou,
                                                                                         score_threshold=FLAGS.score)
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        return pred_bbox


