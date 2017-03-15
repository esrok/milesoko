##  Spot RNN
Playground for character-level recurrent neural networks in python with helpers for training on [AWS spot instances](https://aws.amazon.com/ec2/spot/). Current version uses multi-layer LSTMs from [keras library](https://github.com/fchollet/keras) (only [Theano](https://github.com/Theano/Theano) backend was tested). `AWSRunner` provides API for instance creation, management and interaction.
### char-rnn like usage
```
python runner.py --num 128 --length 20 --epoch 100 --layers 1 --out data/shakespeare -f data/shakespeare/input.txt
```
More info in `runner.py --help`
