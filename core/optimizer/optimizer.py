from torch import optim


class SGD(object):

    def __init__(self,
                 lr=0.1,
                 momentum=0.9,
                 weight_decay=0,
                 enable_bais_decay=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.enable_bais_decay = enable_bais_decay

    def __call__(self, parameters):
        if not self.enable_bais_decay:
            bias_params = []
            weight_params = []
            for name, param in parameters:
                if 'bias' in name:
                    bias_params.append(param)
                else:
                    weight_params.append(param)
            opt = optim.SGD([{
                'params': bias_params,
                'weight_decay': 0.
            }, {
                'params': weight_params,
                'weight_decay': self.weight_decay
            }],
                            lr=self.lr,
                            momentum=self.momentum)
        else:
            parameters = [param for name, param in parameters]
            opt = optim.SGD(parameters,
                            lr=self.lr,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay)
        return opt


class Adam(object):

    def __init__(self,
                 lr=1e-3,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 enable_bais_decay=False,
                 weight_decay=0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.enable_bais_decay = enable_bais_decay

    def __call__(self, parameters):
        if not self.enable_bais_decay:
            bias_params = []
            weight_params = []
            for name, param in parameters:
                if 'bias' in name:
                    bias_params.append(param)
                else:
                    weight_params.append(param)
            opt = optim.Adam([{
                'params': bias_params,
                'weight_decay': 0.
            }, {
                'params': weight_params,
                'weight_decay': self.weight_decay
            }],
                             lr=self.lr,
                             eps=self.epsilon,
                             betas=(self.beta1, self.beta2))
        else:
            parameters = [param for name, param in parameters]
            opt = optim.Adam(parameters,
                             lr=self.lr,
                             betas=(self.beta1, self.beta2),
                             eps=self.epsilon,
                             weight_decay=self.weight_decay)
        return opt
