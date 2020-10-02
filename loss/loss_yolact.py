import tensorflow as tf
import time
from utils import utils


class YOLACTLoss(object):

    def __init__(self, loss_weight_cls=1,
                 loss_weight_box=1.5,
                 loss_weight_mask=6.125,
                 loss_seg=1,
                 neg_pos_ratio=3,
                 max_masks_for_train=100):
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._loss_weight_seg = loss_seg
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train

    def __call__(self, pred, label, num_classes):
        """
        :param num_classes:
        :param anchors:
        :param label: labels dict from dataset
        :param pred:
        :return:
        """
        # all prediction component
        pred_cls = pred['pred_cls']
        pred_offset = pred['pred_offset']
        pred_mask_coef = pred['pred_mask_coef']
        proto_out = pred['proto_out']
        seg = pred['seg']

        # all label component
        # all_offsets: the transformed box coordinate offsets of each pair of prior and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # prior_max_box: the corresponding max IoU gt box for each prior
        # prior_max_index: the index of the corresponding max IoU gt box for each prior
        gt_offset = label['all_offsets']
        conf_gt = label['conf_gt']
        prior_max_box = label['prior_max_box']
        prior_max_index = label['prior_max_index']

        bbox_norm = label['bbox_for_norm']
        masks = label['mask_target']
        classes = label['classes']
        num_obj = label['num_obj']

        # calculate num_pos
        loc_loss = self._loss_location(pred_offset, gt_offset, conf_gt) * self._loss_weight_box
        conf_loss = self._loss_class(pred_cls, num_classes, conf_gt) * self._loss_weight_cls
        mask_loss = self._loss_mask(prior_max_index, pred_mask_coef, proto_out, masks, prior_max_box, conf_gt) * self._loss_weight_mask
        seg_loss = self._loss_semantic_segmentation(seg, masks, classes, num_obj) * self._loss_weight_seg
        total_loss = loc_loss + conf_loss + mask_loss + seg_loss
        return loc_loss, conf_loss, mask_loss, seg_loss, total_loss

    def _loss_location(self, pred_offset, gt_offset, conf_gt):
        # only compute losses from positive samples
        # get postive indices
        pos_indices = tf.where(conf_gt > 0 )
        pred_offset = tf.gather_nd(pred_offset, pos_indices)
        gt_offset = tf.gather_nd(gt_offset, pos_indices)

        # calculate the smoothL1(positive_pred, positive_gt) and return
        num_pos = tf.shape(gt_offset)[0]
        smoothl1loss = tf.keras.losses.Huber(delta=1., reduction=tf.losses.Reduction.NONE)
        loss_loc = tf.reduce_sum(smoothl1loss(gt_offset, pred_offset)) / tf.cast(num_pos, tf.float32)

        return loss_loc

    def _loss_class(self, pred_cls, num_cls, conf_gt):
        batch_conf = tf.reshape(pred_cls, [-1, num_cls])
        batch_conf_max = tf.math.reduce_max(pred_cls)

        # Hard Negative Mining

        # This will be used to determine unaveraged confidence loss across all examples in a batch.
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/modules/multibox_loss.py#L251
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/box_utils.py#L316
        mark = tf.math.log(tf.math.reduce_sum(tf.math.exp(batch_conf-batch_conf_max), 1)) + batch_conf_max - batch_conf[:,0]

        mark = tf.reshape(mark, (tf.shape(pred_cls)[0], -1))  # (n, 27429)
        pos_indices = tf.where(conf_gt > 0 )
        mark = tf.tensor_scatter_nd_update(mark, pos_indices, tf.zeros(tf.shape(pos_indices)[0])) # filter out pos boxes
        num_pos = tf.math.count_nonzero(tf.greater(conf_gt,0), axis=1, keepdims=True)
        num_neg = num_pos * self._neg_pos_ratio

        neutrals_indices = tf.where(conf_gt < 0 )
        mark = tf.tensor_scatter_nd_update(mark, neutrals_indices, tf.zeros(tf.shape(neutrals_indices)[0])) # filter out neutrals (conf_gt = -1)

        idx = tf.argsort(mark, axis=1, direction='DESCENDING')
        idx_rank = tf.argsort(idx, axis=1)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        # Filter out neutrals and positive
        neg_indices = tf.where((tf.cast(idx_rank, dtype=tf.int64) < num_neg) & (conf_gt == 0))

        neg_pred_cls_for_loss = tf.gather_nd(pred_cls, neg_indices)
        neg_gt_for_loss = tf.gather_nd(conf_gt, neg_indices)
        pos_pred_cls_for_loss = tf.gather_nd(pred_cls, pos_indices)
        pos_gt_for_loss = tf.gather_nd(conf_gt, pos_indices)

        target_logits = tf.concat([pos_pred_cls_for_loss, neg_pred_cls_for_loss], axis=0)
        target_labels = tf.concat([pos_gt_for_loss, neg_gt_for_loss], axis=0)
        target_labels = tf.one_hot(tf.squeeze(target_labels), depth=num_cls)

        loss_conf = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(target_labels, target_logits)) / tf.cast(num_pos, tf.float32)

        return loss_conf

    def _loss_mask(self, prior_max_index, coef_p, proto_p, mask_gt, prior_max_box, conf_gt):
        import pdb 
        pdb.set_trace()

        shape_proto = tf.shape(proto_p)
        proto_h = shape_proto[1]
        proto_w = shape_proto[2]
        num_batch = shape_proto[0]
        loss_m = 0.0

        mask_gt = tf.transpose(mask_gt, (0,2,3,1)) #[batch, height, width, num_object]

        for i in tf.range(num_batch):
            pos_indices = tf.where(conf_gt[i] > 0 )
            _pos_prior_index = tf.gather_nd(prior_max_index[i], pos_indices) #shape: [num_positives]
            _pos_prior_box = tf.gather_nd(prior_max_box[i], pos_indices) #shape: [num_positives]
            _pos_coef = tf.gather_nd(coef_p[i], pos_indices) #shape: [num_positives]
            _mask_gt = mask_gt[i]

            if pos_prior_index.shape[1] == 0: # num_positives are zero
                continue
            
            # If exceeds the number of masks for training, select a random subset
            old_num_pos = _pos_coef.shape[0]
            
            if old_num_pos > self._max_masks_for_train:
                perm = tf.random.shuffle(tf.range(_pos_coef.shape[0]))
                select = perm[:self._max_masks_for_train]
                _pos_coef = tf.gather(_pos_coef, select)
                _pos_prior_index = tf.gather(_pos_prior_index, select)
                _pos_prior_box = tf.gather(_pos_prior_box, select)
                
            num_pos = _pos_coef.shape[0]
            _pos_mask_gt = tf.gather(_mask_gt, _pos_prior_index, axis=-1)  
            
            # mask assembly by linear combination
            mask_p = tf.linalg.matmul(proto_p[i], _pos_coef, transpose_a=False, transpose_b=True) # [proto_height, proto_width, num_pos]
            mask_p = tf.math.sigmoid(mask_p)
            
            mask_p = utils.crop(mask_p, _pos_prior_box)  # _pos_prior_box.shape: (num_pos, 4)
            
            mask_loss = tf.keras.losses.binary_crossentropy(_pos_mask_gt, mask_p)
            # Normalize the mask loss to emulate roi pooling's effect on loss.
            pos_get_csize = utils.map_to_center_form(_pos_prior_box)
            mask_loss = tf.reduce_sum(mask_loss, [0, 1]) / pos_get_csize[:, 2] / pos_get_csize[:, 3]
            
            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += tf.reduce_sum(mask_loss)
            
        loss_m /= tf.cast(proto_h, tf.float32) / tf.cast(proto_w, tf.float32)

        return loss_m

    def _loss_semantic_segmentation(self, pred_seg, mask_gt, classes, num_obj):

        shape_mask = tf.shape(mask_gt)
        num_batch = shape_mask[0]
        seg_shape = tf.shape(pred_seg)[1:3]
        loss_seg = 0.

        for idx in tf.range(num_batch):
            seg = pred_seg[idx]
            masks = mask_gt[idx]
            cls = classes[idx]
            objects = num_obj[idx]

            # seg shape (p3 height, p3 width, num_cls)
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.image.resize(masks, [seg_shape[0], seg_shape[1]], method=tf.image.ResizeMethod.BILINEAR)
            masks = tf.cast(masks + 0.5, tf.int64)
            masks = tf.squeeze(tf.cast(masks, tf.float32))

            # obj_mask shape (objects, p3 height, p3 width)
            obj_mask = masks[:objects]
            obj_cls = tf.expand_dims(cls[:objects], axis=-1)

            # create empty ground truth (138, 138, num_cls)
            seg_gt = tf.zeros_like(seg)
            seg_gt = tf.transpose(seg_gt, perm=(2, 0, 1))
            seg_gt = tf.tensor_scatter_nd_update(seg_gt, indices=obj_cls, updates=obj_mask)
            seg_gt = tf.transpose(seg_gt, perm=(1, 2, 0))
            loss_seg += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(seg_gt, seg))
        loss_seg = loss_seg / tf.cast(seg_shape, tf.float32) ** 2 / tf.cast(num_batch, tf.float32)

        return loss_seg
