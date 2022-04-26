from logger.default import Logger
import torch
from metrics.map import mean_average_precision
import torchvision

class Trainer:
    def __init__(self, config, device, model, trainval_dataloaders, optimizer, lr_scheduler, logger):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.logger = logger
        self.train_dataloader = trainval_dataloaders['train']
        self.val_dataloader = trainval_dataloaders['val']
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.iter_batch = 0
        self.iter = 1
        self.thresholds = [0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        self.num_of_epochs = self.config['NUM_OF_EPOCHS']
        self.score_threshold = self.config['SCORE_THRESHOLD']
        self.nms_threshold = self.config['NMS_THRESHOLD']

    def transform_input_target(self, inputs, targets):
        inputs = [input_.to(self.device) for input_ in inputs]

        labels = []
        for i in range(len(inputs)):
            d = {}
            d['boxes'] = torch.LongTensor(targets[i]).to(self.device)
            d['labels'] = torch.ones(d['boxes'].shape[0], dtype = torch.int64).to(self.device)
            labels.append(d)

        return inputs, labels

    def train_epoch(self):
        self.model.train()
        
        loss = None
        
        for inputs, targets in self.train_dataloader:
            if len(inputs) == 0 or len(targets[0]) == 0:
                continue

            inputs, targets = self.transform_input_target(inputs, targets)
            loss_dict = self.model(inputs, targets)
            loss = sum(loss for loss in loss_dict.values()).to("cpu")

            loss.backward(retain_graph = True)
            
            self.logger.add_scalar('train_loss', loss.item(), global_step = self.iter)
            self.logger.log_info('Iter {} train loss: {}'.format(self.iter, loss.item()))
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss = None
            
            self.iter += 1

    def valid_epoch(self):
        self.model.eval()

        gt_boxes = []
        pred_boxes = []
        train_idx = 0

        with torch.no_grad():
            for inputs, targets in self.train_dataloader:
                if len(inputs) == 0 or len(targets) == 0:
                    continue

                inputs, targets = self.transform_input_target(inputs, targets)

                outputs = self.model(inputs)

                # Filter predictions that pass the score threshold
                if len(outputs[0]['scores']) > 0:
                    pred_boxes += [
                        [train_idx, outputs[0]['labels'][index], outputs[0]['scores'][index]] + outputs[0]['boxes'][index].tolist()
                        for index in torchvision.ops.nms(outputs[0]['boxes'], outputs[0]['scores'], self.nms_threshold) 
                        if outputs[0]['scores'][index] > self.score_threshold
                    ]
                
                for target in targets:
                    gt_boxes += [
                        [train_idx, 1, 1] + box.tolist()
                        for box in target['boxes']
                    ]

                train_idx += 1    
                 
        scores = [mean_average_precision(pred_boxes, gt_boxes, iou_threshold = threshold, num_classes = 2) for threshold in self.thresholds]
        mAP = sum(scores) / len(scores)
        
        return mAP

    def train(self):
        for epoch in range(self.num_of_epochs):
            self.logger.log_info('Starting training of epoch {}.'.format(epoch))
            self.train_epoch()
            self.logger.log_info('Entering evaluation for epoch {}'.format(epoch))
            mAP = self.valid_epoch()
            self.logger.log_info('Epoch {}: mAP @ 0.5:0.95:0.05: {}'.format(epoch, mAP))
            torch.save(self.model.state_dict(), 'out/model{}.pt'.format(epoch + 1))
