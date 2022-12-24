import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.common.initializer import initializer, XavierUniform
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
            weight_init="normal"
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=False,
            weight_init="normal")
        self.dwnorm = nn.BatchNorm2d(in_channels)
        self.pwnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(alpha=0.1)

    def construct(self, x):
        x = self.depthwise(x)
        x = self.dwnorm(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pwnorm(x)
        x = self.act(x)
        return x


class Integral(nn.Cell):
    def __init__(self):
        super(Integral, self).__init__()
        self.softmax = P.Softmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.linspace = Tensor([[0, 1, 2, 3, 4, 5, 6, 7]], mstype.float32)
        self.matmul = P.MatMul(transpose_b=True)

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(x, (-1, 8))
        x = self.softmax(x)
        x = self.matmul(x, self.linspace)
        out_shape = x_shape[:-1] + (4,)
        x = self.reshape(x, out_shape)
        return x

class Distance2bbox(nn.Cell):
    def __init__(self):
        super(Distance2bbox, self).__init__()
        self.stack = P.Stack(-1)

    def construct(self, points, distance):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        return self.stack([x1, y1, x2, y2])


class BBox2Distance(nn.Cell):
    def __init__(self):
        super(BBox2Distance, self).__init__()
        self.stack = P.Stack(-1)

    def construct(self, points, bbox):
        left = points[..., 0] - bbox[..., 0]
        top = points[..., 1] - bbox[..., 1]
        right = bbox[..., 2] - points[..., 0]
        bottom = bbox[..., 3] - points[..., 1]
        left = C.clip_by_value(left, Tensor(0.0), Tensor(6.9))
        top = C.clip_by_value(top, Tensor(0.0), Tensor(6.9))
        right = C.clip_by_value(right, Tensor(0.0), Tensor(6.9))
        bottom = C.clip_by_value(bottom, Tensor(0.0), Tensor(6.9))
        return self.stack((left, top, right, bottom))


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
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.full(m.gamma.data.shape, 1).astype("float32")))
                if m.beta is not None:
                    m.beta.set_data(Tensor(np.full(m.beta.data.shape, 0.0001).astype("float32")))
                m.moving_mean.set_data(Tensor(np.full(m.moving_mean.data.shape, 0).astype("float32")))


def shuffleNet(input_size=320, model_size='1.0x'):
    return ShuffleNetV2(input_size=input_size, model_size=model_size)


class NanoDet(nn.Cell):
    def __init__(self, backbone, config, is_training=True):
        super(NanoDet, self).__init__()
        self.backbone = backbone
        feature_size = config.feature_size
        self.strides = [8, 16, 32]
        self.ConvModule = DepthwiseConvModule
        self.lateral_convs_C2 = nn.Conv2d(
            in_channels=116,
            out_channels=96,
            kernel_size=1,
            stride=1,
            pad_mode='same',
            has_bias=True,
            weight_init="xavier_uniform",
            bias_init="zeros"
            )
        self.lateral_convs_C3 = nn.Conv2d(
            in_channels=232,
            out_channels=96,
            kernel_size=1,
            stride=1,
            pad_mode='same',
            has_bias=True,
            weight_init="xavier_uniform",
            bias_init="zeros"
        )
        self.lateral_convs_C4 = nn.Conv2d(
            in_channels=464,
            out_channels=96,
            kernel_size=1,
            stride=1,
            pad_mode='same',
            has_bias=True,
            weight_init="xavier_uniform",
            bias_init="zeros"
        )
        # self.P_upSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        # self.P_upSample2 = P.ResizeBilinear((feature_size[0], feature_size[0]), half_pixel_centers=True)
        # self.P_downSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        # self.P_downSample2 = P.ResizeBilinear((feature_size[2], feature_size[2]), half_pixel_centers=True)
        self.P_upSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        self.P_upSample2 = P.ResizeBilinear((feature_size[0], feature_size[0]))
        self.P_downSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        self.P_downSample2 = P.ResizeBilinear((feature_size[2], feature_size[2]))

        self.cls_convs_P2 = nn.SequentialCell([
            DepthwiseConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
            ),
            DepthwiseConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
            ),
        ])
        self.cls_convs_P3 = nn.SequentialCell([
            DepthwiseConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
            ),
            DepthwiseConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
            ),
        ])
        self.cls_convs_P4 = nn.SequentialCell([
            DepthwiseConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
            ),
            DepthwiseConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
            ),
        ])

        self.gfl_cls_P2 = nn.Conv2d(
            in_channels=96,
            out_channels=113,
            kernel_size=1,
            padding=0,
            has_bias=True,
            weight_init="normal",
            bias_init=-4.595,
        )
        self.gfl_cls_P3 = nn.Conv2d(
            in_channels=96,
            out_channels=113,
            kernel_size=1,
            padding=0,
            has_bias=True,
            weight_init="normal",
            bias_init=-4.595,
        )
        self.gfl_cls_P4 = nn.Conv2d(
            in_channels=96,
            out_channels=113,
            kernel_size=1,
            padding=0,
            has_bias=True,
            weight_init="normal",
            bias_init=-4.595,
        )
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=2)
        self.transpose = P.Transpose()
        self.slice = P.Slice()

    def construct(self, inputs):
        C2, C3, C4 = self.backbone(inputs)
        # neck
        # 对齐通道
        P4 = self.lateral_convs_C4(C4)
        P3 = self.lateral_convs_C3(C3)
        P2 = self.lateral_convs_C2(C2)
        # top -> down
        P3 = self.P_upSample1(P4) + P3
        P2 = self.P_upSample2(P3) + P2
        # down -> top
        P3 = self.P_downSample1(P2) + P3
        P4 = self.P_downSample2(P3) + P4
        # head
        # 每个feature_map 经过两层卷积
        P2 = self.cls_convs_P2(P2)
        P3 = self.cls_convs_P3(P3)
        P4 = self.cls_convs_P4(P4)
        # 对齐分类+回归框, out_channel = 112 (80 + 32)
        P4 = self.gfl_cls_P4(P4)
        P3 = self.gfl_cls_P3(P3)
        P2 = self.gfl_cls_P2(P2)

        P4 = self.reshape(P4, (-1, 112, 100))
        P3 = self.reshape(P3, (-1, 112, 400))
        P2 = self.reshape(P2, (-1, 112, 1600))
        preds = self.concat((P2, P3, P4))
        preds = self.transpose(preds, (0, 2, 1))

        cls_scores = self.slice(preds, (0, 0, 0), (-1, -1, 81))
        bbox_preds = self.slice(preds, (0, 0, 81), (-1, -1, -1))
        return cls_scores, bbox_preds

class QualityFocalLoss(nn.Cell):
    def __init__(self, beta=2.0):
        super(QualityFocalLoss, self).__init__()
        self.sigmoid = P.Sigmoid()
        self.sigmiod_cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.pow = P.Pow()
        self.abs = P.Abs()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.be = beta

    def construct(self, logits, label, score):
        # print(logits[0])
        logits_sigmoid = self.sigmoid(logits)
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        # N, 2100 ,80
        # N, 2100 -> N, 2100 ,1 ->N, 2100, 80
        # N, 2100, 80 (ont hot =1 = iou 0 0)
        score = self.tile(self.expand_dims(score, -1), (1, 1, F.shape(logits)[-1]))
        label = label * score
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        # N, 2100 ,80     N, 2100 ,80
        modulating_factor = self.pow(self.abs(label - logits_sigmoid), self.be)
        qfl_loss = sigmiod_cross_entropy * modulating_factor
        return qfl_loss

class DistributionFocalLoss(nn.Cell):
    def __init__(self):
        super(DistributionFocalLoss, self).__init__()
        # self.loss_weight = loss_weight
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()

    def construct(self, pred, label):
        dis_left = self.cast(label, mstype.int32)
        dis_right = dis_left + 1
        weight_left = self.cast(dis_right, mstype.float32) - label
        weight_right = label - self.cast(dis_left, mstype.float32)

        dfl_loss = (
                self.cross_entropy(pred, dis_left) * weight_left
                + self.cross_entropy(pred, dis_right) * weight_right)
        # dfl_loss = dfl_loss * self.loss_weight
        return dfl_loss

class GIou(nn.Cell):
    """Calculating giou"""

    def __init__(self):
        super(GIou, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        xc_1 = self.min(box_p[..., 0:1], box_gt[..., 0:1])
        xc_2 = self.max(box_p[..., 2:3], box_gt[..., 2:3])
        yc_1 = self.min(box_p[..., 1:2], box_gt[..., 1:2])
        yc_2 = self.max(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        res_mid0 = c_area - union
        res_mid1 = self.div(self.cast(res_mid0, ms.float32), self.cast(c_area, ms.float32))
        giou = iou - res_mid1
        giou = 1 - giou
        giou = C.clip_by_value(giou, -1.0, 1.0)
        giou = giou.squeeze(-1)
        return giou


class Iou(nn.Cell):
    def __init__(self):
        super(Iou, self).__init__()
        self.cast = P.Cast()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        iou = iou.squeeze(-1)
        return iou


class NanoDetWithLossCell(nn.Cell):
    def __init__(self, network):
        super(NanoDetWithLossCell, self).__init__()
        self.network = network
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.less = P.Less()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.zeros = P.Zeros()
        self.reshape = P.Reshape()
        self.sigmoid = P.Sigmoid()
        self.ones = P.Ones()
        self.iou = Iou()
        self.loss_bbox = GIou()
        self.loss_qfl = QualityFocalLoss()
        self.loss_dfl = DistributionFocalLoss()
        self.integral = Integral()
        self.distance2bbox = Distance2bbox()
        self.bbox2distance = BBox2Distance()
        self.sigmoid = P.Sigmoid()
        self.argmax = P.ArgMaxWithValue(axis=-1)
        self.max = P.Maximum()

    def construct(self, x, res_boxes, res_labels, res_center_priors, nums_match):
        cls_scores, bbox_preds = self.network(x)
        cls_scores = self.cast(cls_scores, mstype.float32)
        bbox_preds = self.cast(bbox_preds, mstype.float32)
        mask = self.cast(self.less(-1, res_labels), mstype.float32)
        nums_match = self.reduce_sum(self.cast(nums_match, mstype.float32))
        score = self.ones(F.shape(res_labels), mstype.float32)
        score = score * mask
        pos_nums = C.count_nonzero(mask)
        nums_match = self.max(nums_match, 1)
        if pos_nums > 0:
            weight_targets = self.argmax(self.sigmoid(cls_scores))[1] * mask
            mask_bbox = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
            mask_bbox_preds = self.tile(self.expand_dims(mask, -1), (1, 1, 32))
            bbox_pred_corners = self.integral(bbox_preds)
            decode_grid_cell_centers = (res_center_priors / self.tile(
                self.expand_dims(res_center_priors[..., 2], -1), (1, 1, 4)))
            decode_bbox = res_boxes / self.tile(
                self.expand_dims(res_center_priors[..., 2], -1), (1, 1, 4))
            decode_bbox_pred = self.distance2bbox(decode_grid_cell_centers, bbox_pred_corners)
            # loss_bbox
            loss_bbox = self.loss_bbox(decode_bbox_pred, decode_bbox) * mask * weight_targets
            loss_bbox = self.reduce_sum(loss_bbox) * 2.0
            # loss_dfl
            pred_corners = self.reshape(bbox_preds, (-1, 8))
            target_corners = self.reshape(self.bbox2distance(decode_grid_cell_centers, decode_bbox), (-1,))
            loss_dfl = self.loss_dfl(pred_corners, target_corners) * mask_bbox.reshape(-1) \
                       * self.tile(self.expand_dims(weight_targets, -1), (1, 1, 4)).reshape(-1)
            loss_dfl = self.reduce_sum(loss_dfl) / 4.0 * 0.25
            score = score * self.iou(decode_bbox_pred, decode_bbox) * mask
        else:
            loss_bbox = self.reduce_sum(bbox_preds) * 0
            loss_dfl = self.reduce_sum(bbox_preds) * 0
            weight_targets = self.zeros((1,), mstype.float32)
        # qfl
        # 2100 -> -1 , (0-79), -1, -1,(0-79)
        loss_qfl = self.loss_qfl(cls_scores, res_labels, score)
        loss_qfl = self.reduce_sum(loss_qfl) / nums_match

        avg_factor = self.reduce_sum(weight_targets)

        if avg_factor <= 0:
            loss_qfl = self.zeros((1,), mstype.float32)
            loss_bbox = self.zeros((1,), mstype.float32)
            loss_dfl = self.zeros((1,), mstype.float32)
        else:
            loss_bbox = loss_bbox / avg_factor
            loss_dfl = loss_dfl / avg_factor
            loss_qfl = self.reduce_sum(loss_qfl)
            loss_bbox = self.reduce_sum(loss_bbox)
            loss_dfl = self.reduce_sum(loss_dfl)

        loss = loss_qfl + loss_bbox + loss_dfl

        return loss

class NanodetWithQFL(nn.Cell):
    def __init__(self, network):
        super(NanodetWithQFL, self).__init__()
        self.network = network
        self.cast = P.Cast()
        self.less = P.Less()
        self.tile = P.Tile()
        self.max = P.Maximum()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.one = Tensor(1, mstype.float32)
        self.class_loss = SigmoidFocalClassificationLoss(2.0, 0.75)

    def construct(self, x, res_boxes, res_labels, res_center_priors, nums_match):
        cls_scores, bbox_preds = self.network(x)
        cls_scores = self.cast(cls_scores, mstype.float32)
        bbox_preds = self.cast(bbox_preds, mstype.float32)
        mask = self.cast(self.less(0, res_labels), mstype.float32)
        nums_match = self.reduce_sum(self.cast(nums_match, mstype.float32))
        loss_cls = self.class_loss(cls_scores, res_labels)
        loss_cls = self.reduce_sum(loss_cls, (1,2))
        nums_match = self.max(nums_match, self.one)

        return loss_cls / nums_match


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
        # 1,2100 [4,9,32...]
        # label 1,2100,80
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        # label 1,2100,80  cls_preds 1,2100,80
        #  -> 2100,80 softmax
        # 80 二分类的交叉熵 [0, 0, 0, 1, 0, 0......0,0]
        #                 []
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)

        sigmoid = self.sigmoid(logits)
        label = F.cast(label, mstype.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss

# 使用RetinaNet分类损失
class LossWithFocalLoss(nn.Cell):
    def __init__(self, network):
        super(LossWithFocalLoss, self).__init__()
        self.network = network
        self.class_loss = SigmoidFocalClassificationLoss(2, 0.75)
        self.reduce_sum = P.ReduceSum()
        self.max = P.Maximum()

    def construct(self, x, res_boxes, res_labels, res_center_priors, nums_match):
        # N, 67995,
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

class NanodetInferWithDecoder(nn.Cell):
    def __init__(self, network, center_priors, config):
        super(NanodetInferWithDecoder, self).__init__()
        self.network = network
        # self.distance2bbox = Distance2bbox(config.img_shape)
        self.distribution_project = Integral()
        self.center_priors = center_priors
        self.sigmoid = P.Sigmoid()
        self.expandDim = P.ExpandDims()
        self.tile = P.Tile()
        self.shape = P.Shape()
        self.stack = P.Stack(-1)

    def construct(self, x, max_shape):
        x_shape = self.shape(x)
        default_priors = self.expandDim(self.center_priors, 0)
        cls_preds, reg_preds = self.network(x)
        dis_preds = self.distribution_project(reg_preds) * self.tile(self.expandDim(default_priors[..., 2], -1),
                                                                     (1, 1, 4))
        bboxes = self.distance2bbox(default_priors[..., :2], dis_preds, max_shape)
        scores = self.sigmoid(cls_preds)
        # bboxes = self.tile(self.expandDim(bboxes, -2), (1, 1, 80, 1))
        return bboxes, scores

    def distance2bbox(self, points, distance, max_shape=None):
        y1 = points[..., 0] - distance[..., 1]
        x1 = points[..., 1] - distance[..., 0]
        y2 = points[..., 0] + distance[..., 3]
        x2 = points[..., 1] + distance[..., 2]
        if self.max_shape is not None:
            y1 = C.clip_by_value(y1, Tensor(0), Tensor(self.max_shape[0]))
            x1 = C.clip_by_value(x1, Tensor(0), Tensor(self.max_shape[1]))
            y2 = C.clip_by_value(y2, Tensor(0), Tensor(self.max_shape[0]))
            x2 = C.clip_by_value(x2, Tensor(0), Tensor(self.max_shape[1]))
        return self.stack([y1, x1, y2, x2])

if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    backbone = shuffleNet()
    net = NanoDet(backbone, config)
    x = Tensor(np.random.rand(1,3,320,320),mstype.float32)
    out = net(x)
    print("!")