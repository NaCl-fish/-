import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import matplotlib.pyplot as plt
import input_data
import model
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 雅黑字体

#保存准确率数据，制成csv
result_list=[]

def get_image(img_dir):
    """
        参数解释：
            img_dir：图片路径
            image：图片
    """
    image = Image.open(img_dir)        # 打开img_dir路径下的图片
    image = image.resize([128, 128])   # 改变图片的大小，定为宽高都为128像素
    image = np.array(image)            # 转成多维数组，向量的格式
    return image

N_CLASSES = 2  # 分类数，猫和狗
IMG_W = 128  # resize图片宽高
IMG_H = 128
BATCH_SIZE = 1  # 每批次读取数据的数量
CAPACITY = 2000  # 队列最大容量

test_dir = './data/test/'                 # 训练集的文件夹路径

# 获取要训练的图片和对应的图片标签, 这里返回的train_img是存放猫狗图片路径的列表，train_label是存放对train对应标签的列表(0是猫，1是狗)
test_img, test_label = input_data.get_files(test_dir)

# 读取队列中的数据
test_batch, test_label_batch = input_data.get_batch(test_img, test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# 调用model方法得到返回值, 进行变量赋值
test_logits = model.cnn_inference(test_batch, BATCH_SIZE, N_CLASSES, False)
test_acc = model.evaluation(test_logits, test_label_batch)

pd_list = []

def evaluate_n_image(path):
    # 修改成自己测试集的文件夹路径
    test_dir = './data/test/' #

    global_step = path.split('/')[-1].split('-')[-1]  # 通过切割获取ckpt变量中的步长

    num = 0
    correct = 0

    file_array = input_data.get_files(test_dir)
    test_img = file_array[0]   # 获取测试集的图片路径列表
    test_lab = file_array[1]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, path)  # 加载到当前环境中
        print('模型加载成功, 训练的步数为： %s' % global_step)

        coord = tf.train.Coordinator()  # 创建线程协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        test_accuracy_list = []
        test_loss_list = []
        # test
        for j in tqdm(range(5000)):
            if coord.should_stop():   # 队列中的所有数据已被读出，无数据可读时终止训练
                break
            num += 1
            te_acc = sess.run(test_acc)
            correct += te_acc

    ratio = correct * 100 / num
    print('测试集共接收测试数据', num, '张，其中识别正确', correct, '张，正确率', ratio, '%')
    result_list.append(ratio)
    pd_list.append([global_step, ratio])

if __name__ == '__main__':
    # 调用方法，开始测试
    xl = []
    xl.append(0)
    result_list.append(0)
    pd_list.append([0,0])

    # 修改成自己训练好的模型路径
    logs_train_dir = './log/'

    # print("从指定路径中加载模型...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)  # 读取路径下的checkpoint
    # 载入模型，不需要提供模型的名字，会通过 checkpoint 文件定位到最新保存的模型
    if ckpt and ckpt.model_checkpoint_path:  # checkpoint存在且其存放的变量不为空
        for path in tqdm(ckpt.all_model_checkpoint_paths):
            global_step = path.split('/')[-1].split('-')[-1]
            xl.append(int(global_step) + 1)
            evaluate_n_image(path)
    else:
        print('模型加载失败，checkpoint文件没找到！')

    #测试准确率图
    name = ['global_step', 'ratio']
    pd1 = pd.DataFrame(columns = name, data = pd_list)
    pd1.to_csv('./result.csv')

    plt.figure()  # 建立可视化图像框
    plt.plot(xl, result_list, color='green', label='test_accuracy')
    plt.xlabel("Training_Step(单位：步)")  # x轴取名
    plt.ylabel("Test_Accuracy(%)")  # y轴取名
    plt.legend()  # 给图加上图例
    plt.savefig('./image/Test_Accuracy')
    plt.show()  # 显示图片