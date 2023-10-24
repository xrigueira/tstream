#######################################################
# TX, 11/13/2021
# ConvLSTM with dropout (for MC dropout)
# ConvLSTM model outputs gain-and-losses, loss function combines dQ to compare with discharge at USGS gage
#######################################################

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch

gpudevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('The model will run on', gpudevice)
print(torch.cuda.get_device_name(0))

opt_regularization = 4  # 0: dropout only, 1: l1 on weight, 2: weight decay, 3: weight decay on weight only, 4: l2 on weight only
opt_transform = 0  # 0: only linear transformation, 1: sqrt and log at data side, 2: log in loss function
opt_spinup = 100
opt_tune_lr = 0
opt_subwatershed = 1
opt_retrain = 0  # 0: test directly, 1: tune FC weights only, with soft constraint using subwatershed masks applied on FC weights; 2: tune all parameters
opt_dropout = 0

# Load data
# x = scipy.io.loadmat('../SWIT_28x18_python.mat')
x = scipy.io.loadmat('../SWIT_28x18_python_new.mat')
x = x['X1']
#dat = np.loadtxt('../xy_LR.csv', delimiter=',')
y = scipy.io.loadmat('../Q_LR.mat')
y = y['Q1']

# load buffered masks to crop FC layers. {'FB'}    {'BC'}    {'TG'}    {'TF'}    {'WCB'}    {'RHF'}
sub_mask = scipy.io.loadmat('subwatershed_mask.mat')
sub_mask = sub_mask['ma_c_b']
sub_mask = torch.tensor(sub_mask).to(gpudevice)

# indexes of FC weights for subwatersheds. Different from masks because some FC weights are cropped (outside of boundary)
fc_id = scipy.io.loadmat('sub_fc_mask_cropped.mat')
fc_id = fc_id['sub_fc_id']
fc_id = Variable(torch.tensor(fc_id).to(gpudevice))

#site_name = ['Franklin Basin', 'Beaver Creek', 'Tony Grove', 'Ricks Spring', 'Temple Fork', 'Wood Camp Bridge', 'Right Hand Fork', 'Dewitt Springs', 'Dewitt Springs Campground']
#sub_id = [0, 1, 2, 4, 5, 6]
#t_sub_start = 12742  # 21-Aug-2015
#y2 = scipy.io.loadmat('Q_subwatershed.mat')
#y2 = y2['Q2']


site_name = ['Beaver Creak', 'LR Franklin Basin', 'Right Hand Fork', 'Temple Fork', 'LR Tony Grove', 'LR Wood Camp Bridge', 'Rick Spring', ' Dewitt Springs']
sub_id = [1, 0, 4, 3, 5, 2]
t_sub_start = 12294  # 05/31/2014 - 1
y2 = scipy.io.loadmat('..\Q_subwatershed_new.mat')
y2 = y2['Y3']
y2 = y2[:, sub_id]
y20 = y2

if opt_retrain > 0:
    num_dQ = y2.shape[1] + 1
else:
    num_dQ = 1

# Pre-processing
x_max = np.amax(x)
x = x/(x_max * 1.01)

#y = dat[:, [2]]
y0 = y


if opt_transform == 1:
    y = np.log(np.sqrt(y) + 1)
    y2 = np.log(np.sqrt(y2) + 1)
    y_min = np.amin(y, 0)
    y = y - y_min
    y_max = np.amax(y, 0)
    y = y / y_max
else:
    y_min = 0
    y_max = 4.5
    y = (y - y_min)/y_max

y2 = (y2 - y_min)/y_max

ndat = min(x.shape[0], y.shape[0])
ndat_sub = y2.shape[0]
print(ndat_sub)
y = y[0:ndat]
x = x[0:ndat, :, :, :]

train_size = 9861
id_train1 = range(0, 3000)
id_train2 = range(3000, 6000)
id_train3 = range(6000, train_size)
#train_size_sub = 365 * 3
#id_train4 = range(0, train_size_sub)
#id_train4 = range(181, 1056)
id_train4 = range(407, ndat_sub)
id_train4_x = range(t_sub_start + 407 - 1, t_sub_start + ndat_sub - 1)


test_size = ndat - train_size
id_test1 = range(train_size, ndat)
#id_test1 = range(train_size, t_sub_start + 407)  # USGS gage
# id_test2 = range(train_size_sub, ndat_sub)  # longitudinal dQ
id_test2 = range(0, 407)

dataX = Variable(torch.Tensor(np.array([x])))
dataY = Variable(torch.Tensor(np.array([y])))
dataY2 = Variable(torch.Tensor(np.array([y2])))

trainX1 = dataX[:, id_train1, :, :].to(gpudevice)
trainY1 = dataY[:, id_train1, :].to(gpudevice)

trainX2 = dataX[:, id_train2, :, :].to(gpudevice)
trainY2 = dataY[:, id_train2, :].to(gpudevice)

trainX3 = dataX[:, id_train3, :, :].to(gpudevice)
trainY3 = dataY[:, id_train3, :].to(gpudevice)

testX1 = dataX[:, id_test1, :, :].to(gpudevice)
testY1 = dataY[:, id_test1, :].to(gpudevice)

trainX4 = dataX[:, id_train4_x, :, :].to(gpudevice)
trainY4 = dataY2[:, id_train4, :].to(gpudevice)

id_tmp = range(t_sub_start - 1, t_sub_start + ndat_sub - 1)
testX2 = dataX[:, id_tmp, :, :].to(gpudevice)
testY2 = dataY2.to(gpudevice)

print(testX2.shape)
print(testY2.shape)

#######################################################
#          Define network
#######################################################
# Adapted from: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py


class ConvLSTMCell(nn.Module):

  def __init__(self, input_dim, hidden_dim, kernel_size, bias, layer_idx):
    """
    Initialize ConvLSTM cell.
    Parameters
    ----------
    input_dim: int
        Number of channels of input tensor.
    hidden_dim: int
        Number of channels of hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    """

    super(ConvLSTMCell, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.kernel_size = kernel_size
    self.padding = kernel_size[0] // 2, kernel_size[1] // 2
    self.bias = bias

    # define conv for Layer 1 and other layers separately
    self.conv_input = nn.Conv2d(in_channels=self.input_dim,
                          out_channels=4 * self.hidden_dim,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=self.bias)

    self.conv_hidden = nn.Conv2d(in_channels=self.hidden_dim,
                          out_channels=4 * self.hidden_dim,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=False)

    # would the definition for separate layers induce extra parameters?
    # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
    #                      out_channels=4 * self.hidden_dim,
    #                      kernel_size=self.kernel_size,
    #                      padding=self.padding,
    #                      bias=self.bias)

    # Pooling (only needed for Layer 1)
    self.pool = nn.AvgPool2d(2)

    # Dropout
    self.dropout_conv = nn.Dropout(p=0.01)
    self.dropout_in = nn.Dropout(p=0.01)

  def forward(self, input_tensor, cur_state, layer_idx):
    h_cur, c_cur = cur_state

    # input_tensor = self.dropout_in(input_tensor)

    input_conv = self.conv_input(input_tensor)
    hidden_conv = self.conv_hidden(h_cur)

    # Pooling (only applies for layer 1 x)
    if layer_idx == 0:
        # no need to pad for 28*18 input
        # input_conv = F.pad(input_conv, (0, 0, 0, 1))
        input_conv = self.pool(input_conv)

    # if layer_idx > 0:
    #    combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
    #    combined_conv = self.conv(combined)

    combined_conv = torch.add(input_conv, hidden_conv)

    # combined_conv = self.dropout_conv(combined_conv)

    cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.tanh(cc_g)

    c_next = f * c_cur + i * g
    h_next = o * torch.tanh(c_next)

    return h_next, c_next

  def init_hidden(self, batch_size, image_size):
    height, width = image_size
    return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_hidden.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_hidden.weight.device))


class ConvLSTM(nn.Module):
  """
  Parameters:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
      num_layers: Number of LSTM layers stacked on each other
      batch_first: Whether or not dimension 0 is the batch or not
      bias: Bias or no bias in Convolution
      return_all_layers: Return the list of computations for all layers
      Note: Will do same padding.
  Input:
      A tensor of size B, T, C, H, W or T, B, C, H, W
  Output:
      A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
          0 - layer_output_list is the list of lists of length T of each output
          1 - last_state_list is the list of last states
                  each element of the list is a tuple (h, c) for hidden state and memory
  Example:
      >> x = torch.rand((32, 10, 64, 128, 128))
      >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
      >> _, last_states = convlstm(x)
      >> h = last_states[0][0]  # 0 for layer index, 0 for h index
  """

  def __init__(self, input_dim, out_dim, hidden_dim, kernel_size, num_layers, device,
               batch_first=False, bias=True, return_all_layers=False):
    super(ConvLSTM, self).__init__()

    self._check_kernel_size_consistency(kernel_size)

    # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
    kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
    hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
    if not len(kernel_size) == len(hidden_dim) == num_layers:
      raise ValueError('Inconsistent list length.')

    self.input_dim = input_dim
    self.out_dim = out_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.bias = bias
    self.return_all_layers = return_all_layers
    self.device = device

    cell_list = []
    for i in range(0, self.num_layers):
      cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

      cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                    hidden_dim=self.hidden_dim[i],
                                    kernel_size=self.kernel_size[i],
                                    bias=self.bias,
                                    layer_idx=i))

    self.cell_list = nn.ModuleList(cell_list)

    # FC layer
    # self.linear = nn.Linear(in_features=126 * hidden_dim[0], out_features=self.out_dim)
    # remove NaN cells outside of study area
    self.linear = nn.Linear(in_features=108 * hidden_dim[0], out_features=self.out_dim)

    # Dropout
    self.dropout_fc = nn.Dropout(p=0.1)

  def forward(self, input_tensor, input_tensor2=None, hidden_state=None):
    """
    Parameters
    ----------
    input_tensor: todo
        5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
    hidden_state: todo
        None. todo implement stateful
    Returns
    -------
    last_state_list, layer_output
    """
    if not self.batch_first:
      # (t, b, c, h, w) -> (b, t, c, h, w)
      input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

    b, _, _, h, w = input_tensor.size()

    # 2*2 pooling once
    h = math.ceil(h/2)
    w = math.ceil(w/2)

    # Implement stateful ConvLSTM
    if hidden_state is not None:
        raise NotImplementedError()
    else:
        # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(batch_size=b,
                                       image_size=(h, w))

    layer_output_list = []
    last_state_list = []

    seq_len = input_tensor.size(1)
    cur_layer_input = input_tensor

    # rnn dropout parameters
    if opt_dropout > 0:
        p_rnn = 0.01
    else:
        p_rnn = 0
    multiplier_rnn = 1.0 / (1.0 - p_rnn)

    for layer_idx in range(self.num_layers):
        x_selected = torch.Tensor(cur_layer_input[0, 0, :, :, :].shape).uniform_(0, 1) > p_rnn
        h, c = hidden_state[layer_idx]
        h_selected = torch.Tensor(h.shape).uniform_(0, 1) > p_rnn
        if h.is_cuda:
            x_selected = Variable(x_selected.type(torch.cuda.FloatTensor), requires_grad=False)
            h_selected = Variable(h_selected.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            x_selected = Variable(x_selected.type(torch.FloatTensor), requires_grad=False)
            h_selected = Variable(h_selected.type(torch.FloatTensor), requires_grad=False)
        if self.training & opt_dropout > 0:
            cur_layer_input = torch.mul(x_selected, cur_layer_input) * multiplier_rnn

        output_inner = []
        for t in range(seq_len):
            h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                            cur_state=[h, c], layer_idx=layer_idx)
            if self.training & opt_dropout > 0:
                h = torch.mul(h_selected, h) * multiplier_rnn

            output_inner.append(h)

        layer_output = torch.stack(output_inner, dim=1)
        cur_layer_input = layer_output

    #    layer_output_list.append(layer_output)
    #    last_state_list.append([h, c])

    # if not self.return_all_layers:
    #    layer_output_list = layer_output_list[-1:]
    #    last_state_list = last_state_list[-1:]

    # N-to-1
    # h_last = last_state_list[0][0]
    # h_last = h_last.view(h_last.shape[0], -1)
    # out = self.linear(h_last)

    # N-to-N
    # if a subwatershed mask is supplied as the second input
    if input_tensor2 is not None:
        layer_output0 = layer_output
        num_sub = input_tensor2.shape[0]
        out = torch.zeros(b, seq_len, num_sub)
        # apply mask
        for id_sub in range(num_sub):
            fc_mask = input_tensor2[id_sub, :, :].repeat(1, seq_len, hidden_dim, 1, 1)
            layer_output = layer_output0.mul(fc_mask)
            layer_output = layer_output.view(b, seq_len, layer_output.shape[2], -1)
            id_not_selected = [0, 1, 9, 10, 18, 19, 27, 28, 36, 37, 45, 46, 54, 63, 72, 81, 124, 125]
            id_selected = np.setdiff1d(range(0, 14 * 9), id_not_selected)
            layer_output = layer_output[:, :, :, id_selected]
            h_all = layer_output.view(b, seq_len, -1)
            out[:, :, id_sub] = self.linear(h_all).flatten()

    else:
        # layer_output = self.dropout_fc(layer_output)
        layer_output = layer_output.view(b, seq_len, layer_output.shape[2], -1)
        # remove NaN cells, i.e. outside of study area boundary (not topographic delineation)
        id_not_selected = [0, 1, 9, 10, 18, 19, 27, 28, 36, 37, 45, 46, 54, 63, 72, 81, 124, 125]
        id_selected = np.setdiff1d(range(0, 14 * 9), id_not_selected)
        layer_output = layer_output[:, :, :, id_selected]
        h_all = layer_output.view(b, seq_len, -1)
        if opt_dropout > 0:
            h_all = self.dropout_fc(h_all)
        out = self.linear(h_all)

    # return layer_output_list, last_state_list
    return out

  def _init_hidden(self, batch_size, image_size):
    init_states = []
    for i in range(self.num_layers):
        init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
    return init_states

  @staticmethod
  def _check_kernel_size_consistency(kernel_size):
    if not (isinstance(kernel_size, tuple) or
            (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
      raise ValueError('`kernel_size` must be tuple or list of tuples')

  @staticmethod
  def _extend_for_multilayer(param, num_layers):
    if not isinstance(param, list):
      param = [param] * num_layers
    return param


#######################################################
#          Initialize
#######################################################
channels = 1
hidden_dim = 10
kernel_size = (3, 3)
num_layers = 3
# drop_out = 0.3

mclstm = ConvLSTM(input_dim=channels,
                 out_dim=num_dQ,
                 hidden_dim=hidden_dim,
                 kernel_size=kernel_size,
                 num_layers=num_layers,
                 device=gpudevice,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

print("Model's state_dict:")
for param_tensor in mclstm.state_dict():
    print(param_tensor, "\t", mclstm.state_dict()[param_tensor].size())

# Test forward run
#y_predict = mclstm(trainX4, sub_mask)
#print(trainX4.shape)
#print(y_predict.shape)

#######################################################
#          Load
#######################################################
if opt_retrain > 0:
    # must resize to get consistent FC weight size with the model to be loaded
    mclstm.linear = nn.Linear(in_features=108 * hidden_dim, out_features=1)

# mclstm.load_state_dict(torch.load('TrainedModel_usgs_086'))
mclstm.load_state_dict(torch.load('TrainedModel_usgs_082_2'))

if opt_retrain > 0:
    FC_weights_USGS = mclstm.state_dict()['linear.weight']

    # retrain mode 1: tune FC weights only, mode 2: tune all parameters
    if opt_retrain == 1:
        for param in mclstm.parameters():
            param.requires_grad = False

    # re-initialize the FC layer
    mclstm.linear = nn.Linear(in_features=108 * hidden_dim, out_features=num_dQ)
    # print(mclstm.state_dict()['linear.weight'].requires_grad)
    # inherit weights trained using USGS only for USGS output
    mclstm.linear.weight.data[0, :] = FC_weights_USGS

mclstm.to(gpudevice)

#######################################################
#          Train Again
#######################################################
torch.manual_seed(111111)

if opt_retrain > 0:
    num_epochs = 50
else:
    num_epochs = 0

if opt_tune_lr == 1:
    learning_rate = 1e-5
    lr_gamma = 1.07
else:
    learning_rate = 5e-4
    lr_gamma = 0.85
    step_size = num_epochs / 10

lr_track = np.zeros(num_epochs)
if opt_subwatershed == 0:
    loss_track = np.zeros((5, num_epochs))
else:
    # 0: store error excluding regularization, 1:3: train batch 1-3, 4: test, 5: batch 4 (subwatershed),
    # regularization loss for FC mask constraint
    loss_track = np.zeros((7, num_epochs))

if opt_regularization == 1:
    l1_coeff = 2e-5

if opt_regularization > 1:
    l2_coeff = 1e-5

if opt_tune_lr == 0:
    l2_FC_coeff = 40


def sseloss(outputs, targets, opt_transform, spinup):
    # output shape should be [batch, seq_length, number of dQs]

    outputs = outputs[0, spinup:, :]
    targets = targets[0, spinup:, :]

    #outputs = outputs.flatten()
    #targets = targets.flatten()

    #id_nan = torch.isnan(targets)
    #id_not_nan = torch.any(id_nan)
    #targets = torch.nan_to_num(targets)
    #outputs[id_not_nan] = 0

    if opt_transform == 2:
        # outputs = np.log(np.sqrt(outputs) + 1)
        # targets = np.log(np.sqrt(targets) + 1)
        outputs = torch.log2(outputs + 0.6)
        targets = torch.log2(targets + 0.6)

    err = targets - outputs
    sse = (err ** 2).sum()

    return sse


# separate weight from bias

weight_p, bias_p, weight_fc = [], [], []
for name, p in mclstm.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
        if 'linear' in name:
            weight_fc = p
            # print(p.shape)
            # print(p.requires_grad)


def l1_reg(parameters):
    l1_loss = 0
    for parameter in parameters:
        l1_loss += torch.norm(parameter, 1)

    return l1_loss


def l2_reg(parameters):
    l2_loss = 0
    for parameter in parameters:
        l2_loss += torch.norm(parameter, 2) ** 2
        #print(parameter.requires_grad)
    return l2_loss


def l2_reg_sub_FC(parameters, fc_id):
    # soft constrain to penalize deviation from a masked version of USGS FC weights for each subwatershed
    l2_loss = 0
    # print(parameters.requires_grad)
    for i in range(fc_id.shape[0]):
        tmp = fc_id[i, :].repeat(hidden_dim, 1).flatten()
        err = parameters[0, :].mul(tmp) - parameters[i+1, :]
        l2_loss += torch.norm(err, 2) ** 2
    # print(l2_loss.requires_grad)
    return l2_loss


if opt_regularization == 1 or opt_regularization == 4:
    optimizer = torch.optim.Adam(mclstm.parameters(), lr=learning_rate)

if opt_regularization == 2:
    # criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(mclstm.parameters(), lr=learning_rate, weight_decay=l2_coeff)
    # optimizer = torch.optim.SGD(mclstm.parameters(), lr=learning_rate, weight_decay=l2_coeff, momentum=0.5)

if opt_regularization == 3:
    optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': l2_coeff},
                                  {'params': bias_p, 'weight_decay': 0}], lr=learning_rate)

if opt_tune_lr == 1:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_gamma, step_size=step_size)

# Train the model
for epoch in range(num_epochs):

    # calculate learning rate
    #learning_rate = min_lr + (max_lr - min_lr) * epoch / num_epochs

    outputs = mclstm(trainX1)
    optimizer.zero_grad()
    # obtain the loss function
    # loss = criterion(outputs, trainY1)
    loss = sseloss(outputs[:, :, [0]], trainY1, opt_transform, opt_spinup)

    loss_track[0, epoch] = loss

    if epoch % 5 == 0:
        print("Epoch: %d, training loss: %1.5f" % (epoch, loss.item(), ))

    if opt_regularization == 1:
        loss_reg = l1_reg(weight_p)
        loss += l1_coeff * loss_reg

    if opt_regularization == 4:
        loss_reg = l2_reg(weight_p)  # use weight decay
        loss += l2_coeff * loss_reg

    if opt_retrain == 1:
        params = mclstm.state_dict()
        loss_reg = l2_reg_sub_FC(weight_fc, fc_id)
        loss_track[6, epoch] = l2_FC_coeff * loss_reg
        loss += l2_FC_coeff * loss_reg

    loss_track[1, epoch] = loss

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print("Epoch: %d, training loss + penalty: %1.5f" % (epoch, loss.item()))

    outputs = mclstm(trainX2)
    optimizer.zero_grad()
    # loss = criterion(outputs, trainY2)
    loss = sseloss(outputs[:, :, [0]], trainY2, opt_transform, opt_spinup)

    if epoch % 5 == 0:
        print("Epoch: %d, training loss: %1.5f" % (epoch, loss.item()))

    if opt_regularization == 1:
        loss_reg = l1_reg(weight_p)
        loss += l1_coeff * loss_reg

    if opt_regularization == 4:
        loss_reg = l2_reg(weight_p)  # use weight decay
        loss += l2_coeff * loss_reg

    loss_track[2, epoch] = loss

    if opt_retrain == 1:
        params = mclstm.state_dict()
        loss_reg = l2_reg_sub_FC(weight_fc, fc_id)
        loss_track[6, epoch] = l2_FC_coeff * loss_reg
        loss += l2_FC_coeff * loss_reg

    loss.backward()
    optimizer.step()

    outputs = mclstm(trainX3)
    optimizer.zero_grad()
    # loss = criterion(outputs, trainY3)
    loss = sseloss(outputs[:, :, [0]], trainY3, opt_transform, opt_spinup)
    loss_track[3, epoch] = loss

    if epoch % 5 == 0:
        print("Epoch: %d, training loss: %1.5f" % (epoch, loss.item()))

    if opt_regularization == 1:
        loss_reg = l1_reg(weight_p)
        loss += l1_coeff * loss_reg

    if opt_regularization == 4:
        loss_reg = l2_reg(weight_p)  # use weight decay
        loss += l2_coeff * loss_reg

    if opt_retrain == 1:
        params = mclstm.state_dict()
        loss_reg = l2_reg_sub_FC(weight_fc, fc_id)
        loss_track[6, epoch] = l2_FC_coeff * loss_reg
        loss += l2_FC_coeff * loss_reg

    loss.backward()
    optimizer.step()

    # add subwatershed
    if opt_subwatershed > 0:
        outputs = mclstm(trainX4)
        optimizer.zero_grad()
        # loss = criterion(outputs, trainY4)
        loss = sseloss(outputs[:, :, 1:], trainY4, opt_transform, opt_spinup)
        loss_track[5, epoch] = loss

        if opt_regularization == 1:
            loss_reg = l1_reg(weight_p)
            loss += l1_coeff * loss_reg

        if opt_regularization == 4:
            loss_reg = l2_reg(weight_p)  # use weight decay
            loss += l2_coeff * loss_reg

        if opt_retrain == 1:
            params = mclstm.state_dict()
            loss_reg = l2_reg_sub_FC(weight_fc, fc_id)
            loss_track[6, epoch] = l2_FC_coeff * loss_reg
            loss += l2_FC_coeff * loss_reg

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print("Epoch: %d, training loss: %1.5f, FC constraint loss: %1.5f" % (epoch, loss_track[5, epoch], loss_reg.item()))

    scheduler.step()
    lr_track[epoch] = scheduler.get_last_lr()[0]

    with torch.no_grad():
        y_test = mclstm(testX1)

    # loss_te = criterion(y_test, testY)
    loss_te = sseloss(y_test[:, :, [0]], testY1, opt_transform, opt_spinup)
    loss_track[4, epoch] = loss_te
    loss_te = loss_te/(y_test.shape[1])
    if epoch % 5 == 0:
        print("Epoch: %d, test loss, USGS: %1.5f" % (epoch, loss_te.item()))
        if loss_te.item() < 0.025:
            torch.save(mclstm.state_dict(), 'TrainedModel_epoch_'+str(epoch))

if num_epochs > 0:
    loss_track[0, :] = loss_track[1, :] - loss_track[0, :]
    loss_track[1, :] -= loss_track[0, :]
    # performance track
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(5)
    if opt_tune_lr == 1:
        plt.plot(lr_track, loss_track[0, :], label='L1/L2 penalty')
        for i in range(1, 4):
            plt.plot(lr_track, loss_track[i, :], color='blue', label='USGS train')
        plt.plot(lr_track, loss_track[4, :], ls='dashdot', label='USGS test')
        if opt_subwatershed > 0:
            plt.plot(lr_track, loss_track[5, :], label='Subwatershed train')
            plt.plot(lr_track, loss_track[6, :], label='FC mask constraint')
        plt.xscale('log')
    else:
        plt.plot(loss_track[0, :], label='L1/L2 penalty')
        for i in range(1, 4):
            plt.plot(loss_track[i, :], color='blue', label='USGS train')
        plt.plot(loss_track[4, :], ls='dashdot', label='USGS test')
        if opt_subwatershed > 0:
            plt.plot(loss_track[5, :], label='Subwatershed train')
            plt.plot(loss_track[6, :], label='FC mask constraint')
            plt.plot(loss_track[5, :] + loss_track[6, :], label='Subwatershed train + FC')

    legend = plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.plot()
    plt.show()

#######################################################
#          Test
#######################################################


def predict_reg(model, X, T, X2=None):
    model = model.eval()
    y_pred = model(X, X2).data.cpu().numpy()

    n_dQ = y_pred.shape[2]
    y_pred = y_pred[0, :, :]

    model = model.train()
    y_hat = np.zeros((X.shape[1], n_dQ, T))
    with torch.no_grad():
        for j in range(T):
            tmpy = model(X, X2).data.cpu().numpy()
            y_hat[:, :, j] = tmpy[0, :, :]

    return y_pred, y_hat


y_full, y_mc = predict_reg(mclstm, testX1, 2)
if opt_retrain == 0:
    y_full_sub, y_mc_sub = predict_reg(mclstm, testX1, 2, sub_mask)

y_full = y_full * y_max + y_min
y_mc = y_mc * y_max + y_min

if opt_transform == 1:
    y_full = np.square(np.exp(y_full) - 1)
    y_mc = np.square(np.exp(y_mc) - 1)

if opt_retrain == 0:
    y_full_sub = y_full_sub * y_max + y_min
    y_mc_sub = y_mc_sub * y_max + y_min
    if opt_transform == 1:
        y_full_sub = np.square(np.exp(y_full_sub) - 1)
        y_mc_sub = np.square(np.exp(y_mc_sub) - 1)
else:
    y_full_sub = y_full[:, 1:]
    y_mc_sub = y_mc[:, 1:, :]

# calculate performance
y_mc_mean = y_mc.mean(axis=2)
# y_mc_std = y_hat.std(axis=2)
y_mc_usgs = y_mc_mean[:, [0]]
y_full_usgs = y_full[:, [0]]   # np.sum(y_full, 1)

tmp = y0[id_test1[opt_spinup:]]
err = tmp - y_mc_usgs[opt_spinup:]
nse = 1 - np.sum(np.power(err, 2))/np.sum(np.power(tmp - np.sum(tmp)/err.shape[0], 2))
print("MC dropout NSE: %f" % nse)

err = tmp - y_full_usgs[opt_spinup:]
nse = 1 - np.nansum(np.power(err, 2))/np.nansum(np.power(tmp - np.nansum(tmp)/err.shape[0], 2))
print("No dropout NSE: %f" % nse)

id_tmp = range(t_sub_start - 1 - train_size, t_sub_start + ndat_sub - 1 - train_size)
y_mc_sub = y_mc_sub.mean(axis=2)[id_tmp, :]
y_full_sub = y_full_sub[id_tmp, :]

if opt_retrain == 0:
    err = y20 - y_mc_sub
    nse = 1 - np.nansum(np.power(err, 2))/np.nansum(np.power(y20 - np.nansum(y20)/err.shape[0], 2))
    print("Subwatershed MC dropout NSE: %f" % nse)
else:
    tmp = y20[id_test2, :]
    err = tmp - y_mc_sub[id_test2, :]
    nse = 1 - np.nansum(np.power(err, 2))/np.nansum(np.power(tmp - np.nansum(tmp)/err.shape[0], 2))
    print("Subwatershed MC dropout NSE: %f" % nse)

# USGS prediction
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(5)
plt.plot(y0[id_test1], label='Q @ USGS measurements')
plt.plot(y_mc_usgs, label='MC dropout mean')
plt.plot(y_full_usgs, label='No drop out during test')
legend = plt.legend()
plt.plot()
plt.show()

f = plt.figure()
f.set_figwidth(18)
f.set_figheight(9)
for idx in range(y20.shape[1]):
    # Add a subplot for the image
    ax = f.add_subplot(math.ceil(y20.shape[1] / 2), 2, idx + 1)
    ax.plot(y20[:, idx], label='dQ measurements')
    ax.plot(y_mc_sub[:, idx], label='MC dropout mean')
    ax.plot(y_full_sub[:, idx], label='No drop out during test')
    if opt_retrain > 0:
        ax.plot([407, 407], [0, 5])
    legend = ax.legend()
    plt.title(site_name[sub_id[idx]])

plt.tight_layout()
plt.show()

#######################################################
#          Plot FC weights
#######################################################
weights = mclstm.state_dict()['linear.weight'].cpu()
weights = weights.view(num_dQ, hidden_dim, -1)

w_max = torch.max(torch.abs(weights))
w_min = torch.neg(w_max)

id_not_selected = [0, 1, 9, 10, 18, 19, 27, 28, 36, 37, 45, 46, 54, 63, 72, 81, 124, 125]
id_selected = np.setdiff1d(range(0, 14 * 9), id_not_selected)

fig = plt.figure(figsize=(16, 8))
for idy in range(num_dQ):
    for idx in range(hidden_dim):
        ax = fig.add_subplot(num_dQ, hidden_dim, hidden_dim * idy + idx + 1, xticks=[], yticks=[])
        tmp = weights[idy, idx, :]
        tmp2 = torch.zeros(126)
        tmp2[id_selected] = tmp
        tmp2 = tmp2.view(14, 9)
        subplotobj = ax.imshow(tmp2, 'RdBu_r', vmin=w_min, vmax=w_max)
        if idx == hidden_dim - 1:
            cbar = fig.colorbar(subplotobj, ax=ax, extend='both')

plt.tight_layout()
plt.show()

#######################################################
#          Save
#######################################################
scipy.io.savemat('y_clstm_subwatershed_FCmask.mat', {"y_obs": y0[id_test1], "y_full": y_full, "y_mc": y_mc, "y_obs_sub": y20, "y_full_sub": y_full_sub, "y_mc_sub": y_mc_sub})
torch.save(mclstm.state_dict(), 'TrainedModel_sub')
torch.cuda.empty_cache()
