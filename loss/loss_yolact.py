import tensorflow as tf
import tensorflow_addons as tfa
import time
from utils import utils

class YOLACTLoss(object):
    def __init__(self, 
                 img_h,
                 img_w, 
                 loss_weight_cls=1.0,
                 loss_weight_box=1.5,
                 loss_weight_mask=6.125,
                 loss_weight_mask_iou=25.0,
                 loss_seg=1.0,
                 neg_pos_ratio=3,
                 max_masks_for_train=100, 
                 use_mask_iou=False):
        self.img_h = img_h
        self.img_w = img_w
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._loss_weight_mask_iou = loss_weight_mask_iou
        self._loss_weight_seg = loss_seg
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train
        self.use_mask_iou = use_mask_iou

    def __call__(self, model, pred, label, num_classes, image = None):
        """
        :param num_classes:
        :param anchors:
        :param label: labels dict from dataset
            all_offsets: the transformed box coordinate offsets of each pair of 
                      prior and gt box
            conf_gt: the foreground and background labels according to the 
                     'pos_thre' and 'neg_thre',
                     '0' means background, '>0' means foreground.
            prior_max_box: the corresponding max IoU gt box for each prior
            prior_max_index: the index of the corresponding max IoU gt box for 
                      each prior
        :param pred:
        :return:
        """
        self.image = image
        # all prediction component
        self.pred_cls = pred['pred_cls']
        self.pred_offset = pred['pred_offset']
        self.pred_mask_coef = pred['pred_mask_coef']
        self.proto_out = pred['proto_out']
        self.seg = pred['seg']

        # all label component
        self.gt_offset = label['all_offsets']
        self.conf_gt = label['conf_gt']
        self.prior_max_box = label['prior_max_box']
        self.prior_max_index = label['prior_max_index']

        self.masks = label['mask_target']
        self.classes = label['classes']
        self.num_classes = num_classes
        self.model = model

        # pos_boxes = []

        # for i in range(label['all_offsets'].shape[0]):
        #     boxes_decoded = model.detect._decode(label['all_offsets'][i], model.priors)
        #     pos_indices = tf.where(self.conf_gt[i] > 0 )
        #     pos_boxes_decoded = tf.gather_nd(boxes_decoded, pos_indices)
        #     pos_boxes.append(pos_boxes_decoded.numpy())

        loc_loss = self._loss_location() 

        conf_loss = self._loss_class_ohem() 

        mask_loss, mask_iou_loss = self._loss_mask() 
        mask_iou_loss *= self._loss_weight_mask_iou

        seg_loss = self._loss_semantic_segmentation() 

        total_loss = loc_loss + conf_loss + mask_loss + seg_loss + mask_iou_loss
        
        return loc_loss, conf_loss, mask_loss, mask_iou_loss, seg_loss, \
                total_loss

    def _loss_location(self):
        # only compute losses from positive samples
        # get postive indices
        pos_indices = tf.where(self.conf_gt > 0 )
        pred_offset = tf.gather_nd(self.pred_offset, pos_indices)
        gt_offset = tf.gather_nd(self.gt_offset, pos_indices)

        # calculate the smoothL1(positive_pred, positive_gt) and return
        num_pos = tf.shape(gt_offset)[0]
        smoothl1loss = tf.keras.losses.Huber(delta=1.)
        if tf.reduce_sum(tf.cast(num_pos, tf.float32)) > 0.0:
            loss_loc = smoothl1loss(gt_offset, pred_offset)
        else:
            loss_loc = 0.0

        tf.debugging.assert_all_finite(loss_loc, "Loss Location NaN/Inf")

        return loss_loc*self._loss_weight_box

    def _focal_conf_sigmoid_loss(self, focal_loss_alpha=0.75, focal_loss_gamma=2):
        """
        Focal loss but using sigmoid like the original paper.
        """
        labels = tf.one_hot(self.conf_gt, depth=num_cls)
        # filter out "neutral" anchors
        indices = tf.where(self.conf_gt >= 0)
        labels = tf.gather_nd(labels, indices)
        pred_cls = tf.gather_nd(self.pred_cls, indices)

        fl = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, 
            reduction=tf.keras.losses.Reduction.SUM)
        loss = fl(y_true=labels, y_pred=pred_cls)

        pos_indices = tf.where(self.conf_gt > 0 )
        num_pos = tf.shape(pos_indices)[0]
        return loss #tf.math.divide_no_nan(loss, tf.cast(num_pos, tf.float32))

    def _loss_class(self):
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        loss_conf = scce(tf.cast(self.conf_gt, dtype=tf.int32), self.pred_cls, 
                            self._loss_weight_cls)

        return loss_conf

    def _loss_class_ohem(self):
        # num_cls includes background
        batch_conf = tf.reshape(self.pred_cls, [-1, self.num_classes])

        # Hard Negative Mining
        # Using tf.nn.softmax or tf.math.log(tf.math.reduce_sum(tf.math.exp(batch_conf), 1)) to calculate log_sum_exp
        # might cause NaN problem. This is a known problem https://github.com/tensorflow/tensorflow/issues/10142
        # To get around this using tf.math.reduce_logsumexp and softmax_cross_entropy_with_logit

        # This will be used to determine unaveraged confidence loss across all examples in a batch.
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/modules/multibox_loss.py#L251
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/box_utils.py#L316
        # log_sum_exp = tf.math.log(tf.math.reduce_sum(tf.math.exp(batch_conf), 1))

        # Using inbuild reduce_logsumexp to avoid NaN
        # This function is more numerically stable than log(sum(exp(input))). It avoids overflows caused by taking the exp of large inputs and underflows caused by taking the log of small inputs.
        log_sum_exp = tf.math.reduce_logsumexp(batch_conf, 1)
        # tf.print(log_sum_exp)
        loss_c = log_sum_exp - batch_conf[:,0]

        loss_c = tf.reshape(loss_c, (tf.shape(self.pred_cls)[0], -1))  # (batch_size, 27429)
        pos_indices = tf.where(self.conf_gt > 0 )
        loss_c = tf.tensor_scatter_nd_update(loss_c, pos_indices, tf.zeros(tf.shape(pos_indices)[0])) # filter out pos boxes
        num_pos = tf.math.count_nonzero(tf.greater(self.conf_gt,0), axis=1, keepdims=True)
        num_neg = tf.clip_by_value(num_pos * self._neg_pos_ratio, clip_value_min=tf.constant(self._neg_pos_ratio, dtype=tf.int64), clip_value_max=tf.cast(tf.shape(self.conf_gt)[1]-1, tf.int64))

        neutrals_indices = tf.where(self.conf_gt < 0 )
        loss_c = tf.tensor_scatter_nd_update(loss_c, neutrals_indices, tf.zeros(tf.shape(neutrals_indices)[0])) # filter out neutrals (conf_gt = -1)

        idx = tf.argsort(loss_c, axis=1, direction='DESCENDING')
        idx_rank = tf.argsort(idx, axis=1)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        # Filter out neutrals and positive
        neg_indices = tf.where((tf.cast(idx_rank, dtype=tf.int64) < num_neg) & (self.conf_gt == 0))

        # neg_indices shape is (batch_size, no_prior)
        # pred_cls shape is (batch_size, no_prior, no_class)
        neg_pred_cls_for_loss = tf.gather_nd(self.pred_cls, neg_indices)
        neg_gt_for_loss = tf.gather_nd(self.conf_gt, neg_indices)
        pos_pred_cls_for_loss = tf.gather_nd(self.pred_cls, pos_indices)
        pos_gt_for_loss = tf.gather_nd(self.conf_gt, pos_indices)

        target_logits = tf.concat([pos_pred_cls_for_loss, neg_pred_cls_for_loss], axis=0)
        target_labels = tf.concat([pos_gt_for_loss, neg_gt_for_loss], axis=0)
        target_labels = tf.one_hot(tf.squeeze(target_labels), depth=self.num_classes)

        if tf.reduce_sum(tf.cast(num_pos, tf.float32)+tf.cast(num_neg, tf.float32)) > 0.0:
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                reduction=tf.keras.losses.Reduction.SUM)
            loss_conf = cce(target_labels, target_logits) / tf.reduce_sum(tf.cast(num_pos, tf.float32)+tf.cast(num_neg, tf.float32))
        else:
            loss_conf = 0.0
        return loss_conf*self._loss_weight_cls

    def _loss_mask(self, use_cropped_mask=True):

        shape_proto = tf.shape(self.proto_out)
        proto_h = shape_proto[1]
        proto_w = shape_proto[2]
        num_batch = shape_proto[0]
        loss_m = 0.0
        loss_iou = 0.0

        #[batch, height, width, num_object]
        mask_gt = tf.transpose(self.masks, (0,2,3,1)) 

        maskiou_t_list = []
        maskiou_net_input_list = []
        class_t_list = []
        total_pos = 0

        for i in tf.range(num_batch):
            pos_indices = tf.where(self.conf_gt[i] > 0 )

            #shape: [num_positives]
            _pos_prior_index = tf.gather_nd(self.prior_max_index[i], pos_indices) 

            #shape: [num_positives]
            _pos_prior_box = tf.gather_nd(self.prior_max_box[i], pos_indices) 

            #shape: [num_positives]
            _pos_coef = tf.gather_nd(self.pred_mask_coef[i], pos_indices)

            _mask_gt = mask_gt[i]
            cur_class_gt = self.classes[i]

            if tf.shape(_pos_prior_index)[0] == 0: # num_positives are zero
                continue
            
            # If exceeds the number of masks for training, 
            # select a random subset
            old_num_pos = tf.shape(_pos_coef)[0]
            
            if old_num_pos > self._max_masks_for_train:
                perm = tf.random.shuffle(tf.range(tf.shape(_pos_coef)[0]))
                select = perm[:self._max_masks_for_train]
                _pos_coef = tf.gather(_pos_coef, select)
                _pos_prior_index = tf.gather(_pos_prior_index, select)
                _pos_prior_box = tf.gather(_pos_prior_box, select)

            num_pos = tf.shape(_pos_coef)[0]
            total_pos += num_pos
            pos_mask_gt = tf.gather(_mask_gt, _pos_prior_index, axis=-1) 
            pos_class_gt = tf.gather(cur_class_gt, _pos_prior_index, axis=-1)   
            
            # mask assembly by linear combination
            mask_p = tf.linalg.matmul(self.proto_out[i], _pos_coef, transpose_a=False, 
                transpose_b=True) # [proto_height, proto_width, num_pos]
            mask_p = tf.sigmoid(mask_p)

            # crop the pred (not real crop, zero out the area outside the 
            # gt box)
            if use_cropped_mask:
                # _pos_prior_box.shape: (num_pos, 4)
                # bboxes_for_cropping = tf.stack([
                #     _pos_prior_box[:, 0]/self.img_h, 
                #     _pos_prior_box[:, 1]/self.img_w,
                #     _pos_prior_box[:, 2]/self.img_h,
                #     _pos_prior_box[:, 3]/self.img_w
                #     ], axis=-1)
                # mask_p = utils.crop(mask_p, bboxes_for_cropping)
                mask_p = utils.crop(mask_p, _pos_prior_box)  
                # pos_mask_gt = utils.crop(pos_mask_gt, _pos_prior_box)

            # mask_p = tf.clip_by_value(mask_p, clip_value_min=0.0, 
            #     clip_value_max=1.0)

            # Divide the loss by normalized boxes width and height to get 
            # ROIAlign affect. 

            # Getting normalized boxes widths and height
            # boxes_w = (_pos_prior_box[:, 3] - _pos_prior_box[:, 1])/self.img_w
            # boxes_h = (_pos_prior_box[:, 2] - _pos_prior_box[:, 0])/self.img_h
            boxes_w = (_pos_prior_box[:, 3] - _pos_prior_box[:, 1])
            boxes_h = (_pos_prior_box[:, 2] - _pos_prior_box[:, 0])

            # Adding extra dimension as i/p and o/p shapes are different with 
            # "reduction" is set to None.
            # https://github.com/tensorflow/tensorflow/issues/27190
            _pos_mask_gt = tf.transpose(pos_mask_gt, (2,0,1))
            _mask_p = tf.transpose(mask_p, (2,0,1))
            _pos_mask_gt = tf.expand_dims(_pos_mask_gt, axis=-1)
            _mask_p = tf.expand_dims(_mask_p, axis=-1)
                       
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, 
                reduction=tf.losses.Reduction.NONE)
            mask_loss = bce(_pos_mask_gt, _mask_p)

            mask_loss = tf.reduce_mean(mask_loss, 
                                        axis=(1,2)) 

            tf.debugging.assert_all_finite(mask_loss, "Mask Loss NaN/Inf")

            if use_cropped_mask:
                mask_loss = tf.math.divide_no_nan(mask_loss, boxes_w * boxes_h)
            
            mask_loss = tf.reduce_sum(mask_loss)
            
            if old_num_pos > num_pos:
                mask_loss *= tf.cast(old_num_pos / num_pos, tf.float32)

            loss_m += mask_loss

            # Mask IOU loss
            if self.use_mask_iou:
                pos_mask_gt_area = tf.reduce_sum(pos_mask_gt, axis=(0,1))

                # Area threshold of 25 pixels
                select_indices = tf.where(pos_mask_gt_area > 25 ) 

                if tf.shape(select_indices)[0] == 0: # num_positives are zero
                    continue

                _pos_prior_box = tf.gather_nd(_pos_prior_box, select_indices)
                mask_p = tf.gather(mask_p, tf.squeeze(select_indices), axis=-1)
                pos_mask_gt = tf.gather(pos_mask_gt, tf.squeeze(select_indices), 
                    axis=-1)
                pos_class_gt = tf.gather_nd(pos_class_gt, select_indices)

                mask_p = tf.cast(mask_p + 0.5, tf.uint8)
                mask_p = tf.cast(mask_p, tf.float32)
                maskiou_t = self._mask_iou(mask_p, pos_mask_gt)

                if tf.size(maskiou_t) == 1:
                    maskiou_t = tf.expand_dims(maskiou_t, axis=0)
                    mask_p = tf.expand_dims(mask_p, axis=-1)

                maskiou_net_input_list.append(mask_p)
                maskiou_t_list.append(maskiou_t)
                class_t_list.append(pos_class_gt)

        loss_m = tf.math.divide_no_nan(loss_m, tf.cast(total_pos, tf.float32))

        if len(maskiou_t_list) == 0:
            return loss_m , loss_iou
        else:
            maskiou_t = tf.concat(maskiou_t_list, axis=0)
            class_t = tf.concat(class_t_list, axis=0)
            maskiou_net_input = tf.concat(maskiou_net_input_list, axis=-1)

            maskiou_net_input = tf.transpose(maskiou_net_input, (2,0,1))
            maskiou_net_input = tf.expand_dims(maskiou_net_input, axis=-1)
            num_samples = tf.shape(maskiou_t)[0]
            # TODO: train random sample (maskious_to_train)

            maskiou_p = self.model.fastMaskIoUNet(maskiou_net_input)

            # Using index zero for class label.
            # Indices are K-dimensional. 
            # [number_of_selections, [1st_dim_selection, 2nd_dim_selection, ..., 
            #  kth_dim_selection]]
            indices = tf.concat(
                (
                    tf.expand_dims(tf.range((num_samples), 
                        dtype=tf.int64), axis=-1), 
                    tf.expand_dims(class_t-1, axis=-1)
                ), axis=-1)
            maskiou_p = tf.gather_nd(maskiou_p, indices)

            smoothl1loss = tf.keras.losses.Huber(delta=1.)
            loss_i = smoothl1loss(maskiou_t, maskiou_p)

            loss_iou += loss_i

        return loss_m*self._loss_weight_mask , loss_iou

    def _mask_iou(self, mask1, mask2):
        intersection = tf.reduce_sum(mask1*mask2, axis=(0, 1))
        area1 = tf.reduce_sum(mask1, axis=(0, 1))
        area2 = tf.reduce_sum(mask2, axis=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def _loss_semantic_segmentation(self):
        # Note num_classes here is without the background class so 
        # cfg.num_classes-1
        batch_size = tf.shape(self.seg)[0]
        mask_h = tf.shape(self.seg)[1]
        mask_w = tf.shape(self.seg)[2]
        num_classes = tf.shape(self.seg)[3]
        loss_s = 0.0

        for i in range(batch_size):
            cur_segment = self.seg[i]
            cur_class_gt = self.classes[i]
            masks = self.masks[i]

            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.image.resize(masks, [mask_h, mask_w], 
                method=tf.image.ResizeMethod.BILINEAR)
            masks = tf.cast(masks + 0.5, tf.int64)
            masks = tf.squeeze(tf.cast(masks, tf.float32))

            # [height, width, num_cls]; num_cls including background
            segment_gt = tf.zeros((mask_h, mask_w, num_classes+1)) 
            segment_gt = tf.transpose(segment_gt, perm=(2, 0, 1))

            obj_cls = tf.expand_dims(cur_class_gt, axis=-1)
            segment_gt = tf.tensor_scatter_nd_max(segment_gt, indices=obj_cls, 
                updates=masks)
            segment_gt = tf.transpose(segment_gt, perm=(1, 2, 0))

            segment_gt = tf.expand_dims(segment_gt, axis=-1)
            cur_segment = tf.sigmoid(cur_segment)
            cur_segment = tf.expand_dims(cur_segment, axis=-1)
            cce = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE)
            loss = cce(segment_gt[:,:,1:,:], cur_segment)
            loss = tf.reduce_mean(loss)
            loss_s += loss            

        loss_s /= tf.cast(batch_size, dtype=tf.float32)
        return loss_s*self._loss_weight_seg
