# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import utilities
from tcn import TCN
import time
import tensorflow 
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import pandas
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
class TCNAE:
    model = None
    def __init__(self,
         ts_dimension = 51,
         ts_seq=12,
         dilations = (1, 2, 4, 8, 16),
         nb_filters = 40,
         kernel_size = 40,
         nb_stacks = 1,
         padding = 'same',
         dropout_rate = 0.00,
         filters_conv1d = 20,
         activation_conv1d = 'linear',
         latent_sample_rate = 2,
         pooler = MaxPooling1D,
         lr = 0.001,
         conv_kernel_init = 'glorot_normal',
         loss = 'mse',
         use_early_stopping = False,
         error_window_length = 128,
         verbose = 1
        ):
       
        self.ts_dimension = ts_dimension
        self.dilations = dilations
        self.ts_seq = ts_seq
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.filters_conv1d = filters_conv1d
        self.activation_conv1d = activation_conv1d
        self.latent_sample_rate = latent_sample_rate
        self.pooler = pooler
        self.lr = lr
        self.conv_kernel_init = conv_kernel_init
        self.loss = loss
        self.use_early_stopping = use_early_stopping
        self.error_window_length = error_window_length
        self.build_model()
        
    
    def build_model(self, verbose = 1):
                
        tensorflow.keras.backend.clear_session()
        sampling_factor = self.latent_sample_rate
        print("sampling factor",sampling_factor)
        i = Input(batch_shape=(None, self.ts_seq, self.ts_dimension))
        
        print("shape of i")
        print(i.shape)

        # Put signal through TCN. Output-shape: (batch,sequence length, nb_filters)
        tcn_enc = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                      padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                      kernel_initializer=self.conv_kernel_init, name='tcn-enc')(i)
        enc_flat = Conv1D(filters=self.filters_conv1d, kernel_size=1, activation=self.activation_conv1d, padding=self.padding)(tcn_enc)
        # Now, adjust the number of channels...
        print("shape after tcn_enc",tcn_enc.shape)
        
        tcn_enc1 = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                      padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                      kernel_initializer=self.conv_kernel_init, name='tcn-enc1')(tcn_enc)
        print("shape after tcn_enc1",tcn_enc1.shape)
        enc_flat1 = Conv1D(filters=self.filters_conv1d, kernel_size=1, activation=self.activation_conv1d, padding=self.padding)(tcn_enc1)
        print("shape of enc_flat",enc_flat.shape)
        ## Do some average (max) pooling to get a compressed representation of the time series (e.g. a sequence of length 8)
        # Change the pool_size and strides based on your requirements and input size
        enc_pooled = self.pooler(pool_size= 2, strides= 2, padding='valid', data_format='channels_last')(enc_flat)
        print("ecn_pooled shape",enc_pooled.shape)
        #enc_pooled = self.pooler(pool_size=sampling_factor, strides= None, padding='valid', data_format='channels_last')(enc_flat)
        
        # If you want, maybe put the pooled values through a non-linear Activation
        enc_out = Activation("linear")(enc_pooled)

        # Now we should have a short sequence, which we will upsample again and then try to reconstruct the original series
        dec_upsample = UpSampling1D(size=sampling_factor)(enc_out)
        print("shape of dec_upsampe",dec_upsample.shape)
        dec_reconstructed = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                                padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                                kernel_initializer=self.conv_kernel_init, name='tcn-dec')(dec_upsample)
        print("shape after dec_rec",dec_reconstructed.shape)
        dec_reconstructed1 = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                                padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                                kernel_initializer=self.conv_kernel_init, name='tcn-dec1')(dec_reconstructed)

        print("shape after dec_rec1",dec_reconstructed1.shape)
        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        o = Dense(self.ts_dimension, activation='linear')(dec_reconstructed1)
        print("shape of output",o.shape)
        print("Model Input Shape:", i.shape)
        print("Model Output Shape:", o.shape)

        model = Model(inputs=[i], outputs=[o])
        adam = optimizers.Adam()
        
        model.compile(loss=self.loss, optimizer=adam, metrics=[self.loss])
        if verbose > 1:
            model.summary()
        self.model = model
        
       
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, val_loss: {:.4f}".format(epoch, result.history['loss'][-1], result.history['val_loss'][-1]))
    
    def fit(self, train_X, train_Y, batch_size, epochs=5, verbose = 1):
        history = []
        my_callbacks = None
        if self.use_early_stopping:
            my_callbacks = [EarlyStopping(monitor='val_loss', patience=2, min_delta=1e-4, restore_best_weights=True)]
        start = time.time()
        print("Shape of Train_X:",train_X.shape)
        print("Shape of Train_Y:",train_Y.shape)
        print("batch size:",batch_size)
  
        keras_verbose = 0
        if verbose > 0:
            print("> Starting the Training...")
            keras_verbose = 2
        '''sampling_factor = self.latent_sample_rate
        target_length = 252 
        print("target_length:", target_length)
        train_Y_padded = np.pad(train_Y, ((0, 0), (0, target_length - train_Y.shape[1]), (0, 0)))
        print("Shape of train_Y_padded:", train_Y_padded.shape)'''
        history = self.model.fit(train_X, train_Y, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_split=0.001, 
                            shuffle=True,
                            callbacks=my_callbacks,
                            verbose=keras_verbose)
        self.epoch_end(epochs, history)
        if verbose > 0:
            print("> Training Time :", round(time.time() - start), "seconds.")
        return history
  
    def predict(self, test_X): 
        predictions = self.model.predict(test_X)
        reconstruction_loss = np.mean(np.square(test_X - predictions), axis=(1, 2))
        return reconstruction_loss

    def fit_threshold(self, train_X, train_Y):
        # Assuming train_X and train_Y are the training dataset
        predictions = self.model.predict(train_X)
        reconstruction_loss = np.mean(np.square(train_Y - predictions), axis=(1, 2))
        print("reconstruction loss",reconstruction_loss)
        self.mean_reconstruction = np.mean(reconstruction_loss)
        self.covariance_matrix = np.cov(reconstruction_loss.reshape(-1, 1), rowvar=False)

        print("Mean:", self.mean_reconstruction)
        print("cov:", self.covariance_matrix)
        # Set threshold, for example, using a percentile
        print("reconstruction loss based on percentile")
        self.threshold = np.percentile(reconstruction_loss, 95)
        print("Threshold  percentile:", self.threshold)
        self.threshold = np.mean(reconstruction_loss) + 2 * np.std(reconstruction_loss)
        print("Threshold mean and std:", self.threshold)
        
       

        # Assuming 'reconstruction_loss' is your array of reconstruction losses
        X = reconstruction_loss.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_

        # Set the threshold at the midpoint between the two clusters
        self.threshold = (centroids[0] + centroids[1]) / 2
        print("threshold based on cluster is:",self.threshold)
        return self.threshold

    def detect_anomalies(self, test_X,anomaly_scores):      
        # Assuming anomaly_scores is obtained from the TCN-based Autoencoder
        lof = LocalOutlierFactor(n_neighbors=1000, contamination=0.15)  # Adjust parameters
        anomaly_labels = (lof.fit_predict(anomaly_scores.reshape(-1, 1)) == -1).astype(float)
        '''reconstruction_loss = self.predict(test_X)
        
        # Calculate Mahalanobis distance
        gmm = GaussianMixture(n_components=6, covariance_type='full', random_state=42)

        #gmm = GaussianMixture(n_components=6)  # You may need to adjust the number of components
        gmm.fit(reconstruction_loss.reshape(-1, 1))
        print("finding anoamly score")
        # Anomaly score is the negative log likelihood under the GMM
        anomaly_scores = gmm.score_samples(reconstruction_loss.reshape(-1, 1))
        print("done finsind anomaly score")
        cond_number = np.linalg.cond(self.covariance_matrix)
        print("Condition Number of Covariance Matrix:", cond_number)
        regularization_term = 1e-5
        self.covariance_matrix += regularization_term * np.eye(len(self.covariance_matrix))


        mahalanobis_distances = np.matmul(np.matmul((reconstruction_loss - self.mean_reconstruction),
                                                   np.linalg.inv(self.covariance_matrix)),
                                         (reconstruction_loss - self.mean_reconstruction))

        # Anomaly score is the square of Mahalanobis distance
        anomaly_scores = mahalanobis_distances
        print(anomaly_scores.shape)'''
        return anomaly_labels
       
    '''def predict(self, test_X):
        X_rec =  self.model.predict(test_X)
        print("test_X shape",test_X.shape)
        #print("X_rec shape:",X_rec)
        # do some padding in the end, since not necessarily the whole time series is reconstructed
        X_rec = np.pad(X_rec, ((0,0),(0, test_X.shape[1] - X_rec.shape[1] ), (0,0)), 'constant') 
        E_rec = (X_rec - test_X).squeeze()
        print("E_rec shape",E_rec.shape)
        reshaped_E_rec = E_rec.reshape(-1, E_rec.shape[-1])
        print("reshaped_E_rec shape",reshaped_E_rec.shape)
        # Convert to DataFrame
        df_E_rec = pd.DataFrame(reshaped_E_rec, columns=[f'feature_{i}' for i in range(reshaped_E_rec.shape[1])])
        print("df_E_rec shape",df_E_rec.shape)
        Err = utilities.slide_window(df_E_rec, self.error_window_length, verbose=0)
        print("Err shape",Err.shape)
        print("Err shape 0",Err.shape[0])
        sel = np.random.choice(range(Err.shape[0]),int(Err.shape[0]*0.98))
        selected_err = Err[sel].reshape(-1, Err.shape[2])  # Reshape to 2D array
        print("selected err",selected_err.shape)
        if selected_err.ndim > 1:
             selected_err = selected_err.reshape(-1, selected_err.shape[-1])

        mu = np.mean(selected_err, axis=0)
        print("mu is :",mu)
        
        cov = np.cov(selected_err, rowvar = False)
        
        print("cov is ",cov)
        
        try:
            inv_cov = np.linalg.inv(cov)
        except numpy.linalg.LinAlgError as err:
            print("Error, probably singular matrix!")
            inv_cov = np.eye(cov.shape[0])
    
        X_diff_mu = Err[:] - mu
        print("X_diff_mu shape:", X_diff_mu.shape)
        print("inv_cov shape:", inv_cov.shape)
        M = np.sum(X_diff_mu @ inv_cov.T * X_diff_mu, axis=2)
        sq_mahalanobis = M
        
        sq_mahalanobis_flat = sq_mahalanobis.reshape(-1)
        
        print("sq_mahalanobis and sq_mahalanobis_flat shape", sq_mahalanobis.shape,sq_mahalanobis_flat.shape)
        #sq_mahalanobis = utilities.mahalanobis_distance(X=Err[:], cov=cov, mu=mu)
        # moving average over mahalanobis distance. Only slightly smooths the signal
        anomaly_score = np.convolve(sq_mahalanobis_flat, np.ones((50,))/50, mode='same')
        anomaly_score = np.sqrt(anomaly_score)
        return anomaly_score'''


# +
import tensorflow as tf

print("TensorFlow version:", tf.__version__)


# -
# !pip install tcn



