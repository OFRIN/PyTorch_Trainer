import sys
import torch

from torch.utils.tensorboard import SummaryWriter

from tools.general import io_utils, time_utils, json_utils

class Trainer:
    def __init__(self, 
        model, optimizer, scheduler, loader, losses,
        amp, tensorboard_dir, log_names, max_epochs,
        data_path):

        self.model = model
        self.loader = loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.losses = losses

        self.amp = amp
        self.log_names = log_names

        self.train_timer = time_utils.Timer()
        self.train_meter = io_utils.Average_Meter(self.log_names)
        
        self.writer = SummaryWriter(tensorboard_dir)

        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.num_iterations = len(self.loader)
        self.use_cuda = torch.cuda.is_available()

        self.epoch = 1
        self.max_epochs = max_epochs

        self.data = {
            'train':[],
            'validation':[]
        }
        self.data_path = data_path

    # Have to override
    def calculate_losses(self, logits, labels):
        class_loss_fn = self.losses[0]

        loss = class_loss_fn(logits, labels)
        return loss

    def update_data(self, domain, data):
        self.data[domain].append(data)
        json_utils.write_json(self.data_path, self.data)

    def update_tensorboard(self, tag, value, epoch):
        self.writer.add_scalar(tag, value, epoch)

    def step(self):
        self.train_timer.tik()

        ep_digits = io_utils.get_digits_in_number(self.max_epochs)
        ni_digits = io_utils.get_digits_in_number(self.num_iterations)

        for i, (images, labels) in enumerate(self.loader):
            i += 1
            progress_format = '\r# Epoch = %0{}d, [%0{}d/%0{}d] = %02.2f%%'.format(ep_digits, ni_digits, ni_digits)

            sys.stdout.write(progress_format%(self.epoch, i, self.num_iterations, i / self.num_iterations * 100))
            sys.stdout.flush()

            if self.use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(images)

            losses = self.calculate_losses(logits, labels)
            if not isinstance(losses, list):
                losses = [losses]

            self.train_meter.add({name:loss.item() for name, loss in zip(self.log_names, losses)})
            
            self.optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(losses[0]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses[0].backward()
                self.optimizer.step()
        print('\r', end='')

        self.epoch += 1
        self.scheduler.step()
        
        return self.train_meter.get(clear=True), self.train_timer.tok(clear=True)