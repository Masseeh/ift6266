## First model : Convolutional auto-encoder

For the starting point, I chose convolutional auto-encoder. It was my first experience to implement neural net using Theano. The network was trained using SGD (Adam) algorithm. 
For the architecture I used three layers of (Conv , Maxpooling) with filter size of 5 and 32 filters and pool size of 2. And for the bottle-neck there is a full connected neural net with 800 hidden units which represent latent variable.
The resulted images are as follow:

![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/0-org.png) 
![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/0-tobe.png)


![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/1-org.png) 
![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/1-tobe.png)


![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/2-org.png) 
![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/2-tobe.png)


![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/3-org.png) 
![alt text](https://github.com/Masseeh/ift6266/blob/master/docs/images/convae/3-tobe.png)

The hyperparameters were : learning rate = 0.0009 , number of epochs = 200 , batch size = 64
For the training phase I used early stopping with patience of 2. Training process only took 12 epoch, so it was pretty fast.

### Summary:

-- Without using captions
-- Fast training
-- blurry results


### Future:

-- Try generative models like GANs , VAEs conditioning on captions and incomplete image
