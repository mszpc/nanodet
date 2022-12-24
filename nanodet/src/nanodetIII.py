import numpy as np

import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.common.initializer import initializer, XavierUniform
from mindspore import Parameter
from src.model_utils.config import config

class DepthwiseConvModule(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
    ):
        super(DepthwiseConvModule, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            group=in_channels,
            has_bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, has_bias=False)
        self.dwnorm = nn.BatchNorm2d(in_channels)
        self.pwnorm = nn.BatchNorm2d(out_channels)
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU(alpha=0.1)
        # self.init_weights()

    def construct(self, x):
        x = self.depthwise(x)
        x = self.dwnorm(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pwnorm(x)
        x = self.act(x)
        return x


class ShuffleV2Block(nn.Cell):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(in_channels=inp, out_channels=mid_channels, kernel_size=1, stride=1,
                      pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=mid_channels, momentum=0.9),
            nn.LeakyReLU(alpha=0.1),
            # dw
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=ksize, stride=stride,
                      pad_mode='pad', padding=pad, group=mid_channels, has_bias=False),

            nn.BatchNorm2d(num_features=mid_channels, momentum=0.9),
            # pw-linear
            nn.Conv2d(in_channels=mid_channels, out_channels=outputs, kernel_size=1, stride=1,
                      pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=outputs, momentum=0.9),
            nn.LeakyReLU(alpha=0.1),
        ]
        self.branch_main = nn.SequentialCell(branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=ksize, stride=stride,
                          pad_mode='pad', padding=pad, group=inp, has_bias=False),
                nn.BatchNorm2d(num_features=inp, momentum=0.9),
                # pw-linear
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1,
                          pad_mode='pad', padding=0, has_bias=False),
                nn.BatchNorm2d(num_features=inp, momentum=0.9),
                nn.LeakyReLU(alpha=0.1),
            ]
            self.branch_proj = nn.SequentialCell(branch_proj)
        else:
            self.branch_proj = None
        self.squeeze = P.Squeeze(axis=0)

    def construct(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            x_proj = self.squeeze(x_proj)
            x = self.squeeze(x)
            return P.Concat(1)((x_proj, self.branch_main(x)))
        if self.stride == 2:
            x_proj = old_x
            x = old_x
            return P.Concat(1)((self.branch_proj(x_proj), self.branch_main(x)))
        return None

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = P.Shape()(x)
        x = P.Reshape()(x, (batchsize * num_channels // 2, 2, height * width,))
        x = P.Transpose()(x, (1, 0, 2,))
        x = P.Reshape()(x, (2, -1, num_channels // 2, height, width,))
        return x[0:1, :, :, :, :], x[-1:, :, :, :, :]

class ShuffleNetV2(nn.Cell):
    def __init__(self, input_size=224, model_size='1.0x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(num_features=input_channel, momentum=0.9),
            nn.LeakyReLU(alpha=0.1),
        ])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            feature = []
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    feature.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    feature.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel
            self.features.append(feature)

        # self.features = nn.SequentialCell([*self.features])
        self.stage2 = nn.SequentialCell([*self.features[0]])
        self.stage3 = nn.SequentialCell([*self.features[1]])
        self.stage4 = nn.SequentialCell([*self.features[2]])

        self._initialize_weights()

    def construct(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        # x = self.features(x)
        C2 = self.stage2(x)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        return C2, C3, C4

    def _initialize_weights(self):
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    m.weight.set_data(Tensor(np.random.normal(0, 0.01,
                                                              m.weight.data.shape).astype("float32")))
                else:
                    m.weight.set_data(Tensor(np.random.normal(0, 1.0/m.weight.data.shape[1],
                                                              m.weight.data.shape).astype("float32")))

            if isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))

def shuffleNet(model_size='1.0x'):
    return ShuffleNetV2(input_size=320, model_size=model_size)

class NanoDet(nn.Cell):
    def __init__(self, backbone, config, is_training=True):
        super(NanoDet, self).__init__()
        self.backbone = backbone
        feature_size = config.feature_size
        self.strides = [8, 16, 32]
        self.ConvModule = DepthwiseConvModule
        self.lateral_convs = nn.CellList()
        self.lateral_convs.append(nn.Conv2d(116, 96, kernel_size=1, stride=1, pad_mode='same', has_bias=True))
        self.lateral_convs.append(nn.Conv2d(232, 96, kernel_size=1, stride=1, pad_mode='same', has_bias=True))
        self.lateral_convs.append(nn.Conv2d(464, 96, kernel_size=1, stride=1, pad_mode='same', has_bias=True))
        # self.P_upSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        # self.P_upSample2 = P.ResizeBilinear((feature_size[0], feature_size[0]), half_pixel_centers=True)
        # self.P_downSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        # self.P_downSample2 = P.ResizeBilinear((feature_size[2], feature_size[2]), half_pixel_centers=True)
        self.P_upSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        self.P_upSample2 = P.ResizeBilinear((feature_size[0], feature_size[0]))
        self.P_downSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        self.P_downSample2 = P.ResizeBilinear((feature_size[2], feature_size[2]))
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=2)
        self.transpose = P.Transpose()
        self.slice = P.Slice()
        self._make_layer()
        # self.tensor_summary = P.TensorSummary()

    def _build_shared_head(self):
        cls_convs = nn.SequentialCell()
        # reg_convs = nn.CellList()
        for i in range(2):
            cls_convs.append(
                self.ConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        return cls_convs

    def _make_layer(self):
        self.cls_convs = nn.CellList()
        # self.reg_convs = nn.CellList()
        for _ in self.strides:
            # cls_convs, reg_convs = self._build_shared_head()
            cls_convs = self._build_shared_head()
            self.cls_convs.append(cls_convs)
            # self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.CellList()
        for _ in self.strides:
            self.gfl_cls.append(
                nn.Conv2d(
                    in_channels=96,
                    out_channels=112,
                    kernel_size=1,
                    padding=0,
                    has_bias=True,
                )
            )
        # self.init_weights()

    def construct(self, inputs):
        C2, C3, C4 = self.backbone(inputs)
        # self.tensor_summary("C3", C3)
        # 对齐通道
        P4 = self.lateral_convs[2](C4)
        P3 = self.lateral_convs[1](C3)
        P2 = self.lateral_convs[0](C2)
        # top -> down
        P3 = self.P_upSample1(P4) + P3
        P2 = self.P_upSample2(P3) + P2
        # down -> top
        P3 = self.P_downSample1(P2) + P3
        P4 = self.P_downSample2(P3) + P4
        # P2, P3, P4 = inputs
        P2 = self.cls_convs[0](P2)
        P3 = self.cls_convs[1](P3)
        P4 = self.cls_convs[2](P4)

        P4 = self.gfl_cls[2](P4)
        P3 = self.gfl_cls[1](P3)
        P2 = self.gfl_cls[0](P2)
        # for k,v in self.gfl_cls[0].parameters_and_names():
        #     print("weight",Tensor(v))
        # self.tensor_summary("P3", P3)

        P4 = self.reshape(P4, (-1, 112, 100))
        P3 = self.reshape(P3, (-1, 112, 400))
        P2 = self.reshape(P2, (-1, 112, 1600))
        preds = self.concat((P2, P3, P4))
        preds = self.transpose(preds, (0, 2, 1))

        cls_scores = self.slice(preds, (0, 0, 0), (-1, -1, 80))
        bbox_preds = self.slice(preds, (0, 0, 80), (-1, -1, -1))
        return cls_scores, bbox_preds

    def init_weights(self):
        pass

class SigmoidFocalClassificationLoss(nn.Cell):
    """"
    Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples. Default: 2.0
        alpha (float): Hyper-parameter to balance the positive and negative example. Default: 0.25

    Returns:
        Tensor, the focal loss.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.sigmoid = P.Sigmoid()
        self.pow = P.Pow()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = F.cast(label, mstype.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class LossWithFocalLoss(nn.Cell):
    def __init__(self, network):
        super(LossWithFocalLoss, self).__init__()
        self.network = network
        self.class_loss = SigmoidFocalClassificationLoss(2, 0.75)
        self.reduce_sum = P.ReduceSum()
        self.max = P.Maximum()

    def construct(self, x, res_boxes, res_labels, res_center_priors, nums_match):
        cls_scores, bbox_preds = self.network(x)
        cls_scores = self.cast(cls_scores, mstype.float32)
        bbox_preds = self.cast(bbox_preds, mstype.float32)

        nums_match = self.max(nums_match, 1)
        loss_cls = self.class_loss(cls_scores, res_labels)
        loss_cls = self.reduce_sum(loss_cls) / nums_match

        return loss_cls

class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__()
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")

        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True

        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss

if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    x = Tensor(np.random.rand(1,3,320,320),ms.float32)
    backbone = shuffleNet()
    net = NanoDet(backbone, config)
    out = net(x)

    print("!")