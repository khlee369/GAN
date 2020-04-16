import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import xavier_initializer

class GAN:
    '''
    <Configuration info>
    ID : Model ID
    n_iter : Total # of iterations
    n_prt : Loss print cycle
    n_input : Dimension of input
    n_output : Dimension of output
    n_batch : Size of batch
    n_save : Model save cycle
    n_history : Train/Test loss save cycle
    LR : Learning rate
    
    <Configuration example>
    config = {
        'ID' : 'test_NN',
        'n_iter' : 5000,
        'g_iter' : 1,
        'd_iter' : 1,
        'n_dist' : 10,
        'n_prt' : 100,
        'n_input' : 784,
        'n_output' : 10,
        'n_batch' : 50,
        'n_save' : 1000,
        'n_history' : 50,
        'LR' : 0.0001
    }
    '''
    
    def __init__(self, config):
        self.ID = config['ID']
        self.n_iter = config['n_iter']
        self.g_iter = config['g_iter']
        self.d_iter = config['d_iter']
        self.n_dist = config['n_dist']
        self.n_prt = config['n_prt']
        self.n_input = config['n_input']
        self.n_output = config['n_output']
        self.n_batch = config['n_batch']
        self.n_save = config['n_save']
        self.n_history = config['n_history']
        self.LR = config['LR']
        
        self.history = {
            'loss_d' : [],
            'loss_g' : []
        }
        
        self.checkpoint = 0
        self.path = './{}'.format(self.ID)
        try: 
            os.mkdir(self.path)
            os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
        except FileExistsError:
            msg = input('[FileExistsError] Will you remove directory? [Y/N] ')
            if msg == 'Y': # or debug 
                shutil.rmtree(self.path)
                os.mkdir(self.path)
                os.mkdir('{0}/{1}'.format(self.path, 'checkpoint'))
            else: 
                print('Please choose another ID')
                assert 0
        
        self.fake = 0
        self.real = 1
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.dist = tf.placeholder(tf.float32, [None, self.n_dist])
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            
            self.generated_img = self.generator(self.dist)['output']
            self.discr_g = self.discriminator(self.generated_img)['output']
            self.discr_x = self.discriminator(self.x, reuse=True)['output']
            
            self.loss_g = self.compute_loss_g(self.discr_g)
            self.loss_d = self.compute_loss_d(self.discr_g, self.discr_x)
            
            self.optm = tf.train.AdamOptimizer(self.LR, name='optm')
            
            self.optm_g = self.optm.minimize(self.loss_g, var_list=self.graph.get_collection('variables', 'generator'))
            self.optm_d = self.optm.minimize(self.loss_d, var_list=self.graph.get_collection('variables', 'discriminator'))
            
            self.saver = tf.train.Saver(max_to_keep=None)
            self.init = tf.global_variables_initializer()
     
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(self.init)
    
        print('Model ID : {}'.format(self.ID))
        print('Model saved at : {}'.format(self.path))
        
    def fit(self, data):
        for step in range(1, self.n_iter+1):
            train_x, train_y = data.train.next_batch(self.n_batch)
            train_dist = np.random.multivariate_normal(np.zeros(self.n_dist), np.eye(self.n_dist), self.n_batch)
                        
            for i in range(self.d_iter):
                self.sess.run(self.optm_d, feed_dict={self.dist : train_dist, self.x : train_x})
        
            for i in range(self.g_iter):
                self.sess.run(self.optm_g, feed_dict={self.dist : train_dist})
            
            if step % self.n_prt == 0:
                loss_g = self.get_loss_g(train_dist)
                loss_d = self.get_loss_d(train_dist, train_x)
                print('Your G_loss ({0}/{1}) : {2}'.format(step, self.n_iter, loss_g))
                print('Your D_loss ({0}/{1}) : {2}\n'.format(step, self.n_iter, loss_d))

                gen_img = self.sess.run(self.generated_img, feed_dict={self.dist : train_dist})
                plt.imshow(gen_img[0].reshape(28,28), 'gray')
                plt.title("Generated Img")
                plt.show()
                
            if step % self.n_save == 0:
                self.checkpoint += self.n_save
                self.save('{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, self.checkpoint))
                
            if step % self.n_history == 0:
                loss_g = self.get_loss_g(train_dist)
                loss_d = self.get_loss_d(train_dist, train_x)
                self.history['loss_g'].append(loss_g)
                self.history['loss_d'].append(loss_d)
    
    def fc_layer(self, input_tensor, name, n_out, activate_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weight = tf.get_variable('weight', [n_in, n_out], tf.float32, xavier_initializer())
            bias = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(input_tensor, weight), bias, name='logits')
            if activate_fn is None : return logits
            else: return activate_fn(logits, name='activation')
            
    def generator(self, x):
        with tf.variable_scope('generator'):
            generator1 = self.fc_layer(x, 'generator1', 100)
            generator2 = self.fc_layer(generator1, 'generator2', 300)
            generator3 = self.fc_layer(generator2, 'generator3', 500)
            output = self.fc_layer(generator3, 'output', self.n_input, activate_fn=None)
        return {
            'generator1' : generator1,
            'generator2' : generator2,
            'generator3' : generator3,
            'output' : output,
        }

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            discriminator1 = self.fc_layer(x, 'discriminator1', 500)
            discriminator2 = self.fc_layer(discriminator1, 'discriminator2', 300)
            discriminator3 = self.fc_layer(discriminator2, 'discriminator3', 100)
            output = self.fc_layer(discriminator3, 'output', 1, activate_fn=None)
        return {
            'discriminator1' : discriminator1,
            'discriminator2' : discriminator2,
            'discriminator3' : discriminator3,
            'output' : output,
        }
            
    def compute_loss_g(self, output):
        with tf.variable_scope('compute_loss_g'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=tf.ones_like(output))
            loss = tf.reduce_mean(loss)
        return loss
    
    def compute_loss_d(self, output, real_img):
        with tf.variable_scope('compute_loss_d'):
            loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=tf.zeros_like(output))
            loss1 = tf.reduce_mean(loss1)
            loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_img, labels=tf.ones_like(real_img))
            loss2 = tf.reduce_mean(loss2)
            loss = tf.add(loss1, loss2)
        return loss
 
    def get_loss_g(self, dist):
        return self.sess.run(self.loss_g, feed_dict={self.dist : dist})

    def get_loss_d(self, dist, x):
        return self.sess.run(self.loss_d, feed_dict={self.dist : dist, self.x : x})
    
    ## Save/Restore
    def save(self, path):
        self.saver.save(self.sess, path)
        
    def load(self, path):
        self.saver.restore(self.sess, path)
        checkpoint = path.split('_')[-1]
        self.checkpoint = int(checkpoint)
        print('Model loaded from file : {}'.format(path))