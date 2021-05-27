import os

import numpy as np
import torch
import sys
sys.path.insert(1, '/Users/nehamotlani/Desktop/College_Courses/Research/Repo/LayoutAndElementExtraction/basecode/chargrid2d-icdar/')

from chargrid2d.dataloader import ChargridDataloader
from chargrid2d.loss import ChargridLoss
from chargrid2d.metrics import IoU

from chargrid2d.model import Chargrid2D

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(weights_folder='/Users/nehamotlani/Desktop/College_Courses/Research/Repo/LayoutAndElementExtraction/basecode/chargrid2d-icdar/data/weights'):
    N_EPOCHS = 100
    best_loss = np.infty

    dataloader = ChargridDataloader(root='/Users/nehamotlani/Desktop/College_Courses/Research/Repo/LayoutAndElementExtraction/basecode/chargrid2d-icdar/data', list_file_name_path='train_files.txt',
                                    image_size=512, batch_size=1, validation_split=0.1)
    val_dataloader = dataloader.split_validation()

    loss_fn = ChargridLoss()
    model = Chargrid2D(input_channels=len(dataloader.dataset.corpus) + 1, n_classes=len(dataloader.dataset.target))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(N_EPOCHS):
        print('Epoch {}/{}'.format(epoch, N_EPOCHS - 1))
        print('-' * 10)

        # metrics
        train_metrics = IoU(len(dataloader.dataset.target), ignore_index=0)
        val_metrics = IoU(len(dataloader.dataset.target), ignore_index=0)

        # -------- TRAIN -------
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            # we need to get gt_seg, gt_boxmask, gt_boxcoord
            img, mask, boxes, lbl_boxes = batch
            print(img,mask,boxes,lbl_boxes)
            img, mask = img.to(device), mask.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            pred_seg, pred_boxmask, pred_boxcoord = model(img)
            # default ground truth boxmask
            gt_boxmask = torch.ones([pred_boxmask.size()[0], pred_boxmask.size()[2], pred_boxmask.size()[3]]
                                    , dtype=torch.int64)  # TODO: wrong

            mask = mask.type(torch.int64)
            loss = loss_fn(pred_seg, pred_boxmask, pred_boxcoord, mask, gt_boxmask, boxes)
            train_metrics.add(pred_seg, mask)
            epoch_loss += loss.item()  # loss is mean loss of batch
            print("Step", i, 'loss =', loss.item(), '\t cumulative iou =', train_metrics.value()[1])
            # backward
            loss.backward()
            optimizer.step()

        print("Epoch {} Training loss: {}".format(epoch, epoch_loss / len(dataloader)))

        # -------- EVALUATION -------
        model.eval()
        # TODO: update evaluation
        print("START EVALUATION")
        epoch_loss = 0
        for i, batch in enumerate(val_dataloader):
            # we need to get gt_seg, gt_boxmask, gt_boxcoord
            img, mask, boxes, lbl_boxes = batch
            img, mask = img.to(device), mask.to(device)

            # forward
            pred_seg, pred_boxmask, pred_boxcoord = model(img)
            mask = mask.type(torch.int64)

            # default ground truth boxmask
            gt_boxmask = torch.ones([pred_boxmask.size()[0], pred_boxmask.size()[2], pred_boxmask.size()[3]]
                                    , dtype=torch.int64)  # TODO: wrong

            loss = loss_fn(pred_seg, pred_boxmask, pred_boxcoord, mask, gt_boxmask, boxes)
            val_metrics.add(pred_seg, mask)
            epoch_loss += loss.item()  # loss is mean loss of batch
            print("Step", i, 'loss =', loss.item(), '\t cumulative iou =', val_metrics.value()[1])
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(weights_folder, 'model_epoch_' + str(epoch) + '.pth'))

        print("Epoch {} validation loss: {}".format(epoch, epoch_loss / len(val_dataloader)))


if __name__ == '__main__':
    train()

# 87533049
# 0001118259
# 0060207528
# 87672097
# 0001123541
# 0060214859
# 87682908
# 0001129658
# 0060255888
# 88057519
# 0001209043
# 0060262650
# 88547278_88547279
# 0001239897
# 0060270727
# 89368010
# 0001438955
# 0060302201
# 89386032
# 0001456787
# 0060308251
# 89817999_8002
# 0001463282
# 0060308461
# 89867723
# 0001463448
# 0071032790
# 91104867
# 0001476912
# 0071032807
# 91161344_91161347
# 0001477983
# 00920222
# 91315069_91315070
# 0001485288
# 00920294
# 91355841
# 0011505151
# 00922237
# 91356315
# 0011838621
# 01073843
# 91361993
# 0011845203
# 01122115
# 91372360
# 0011856542
# 01150773_01150774
# 91391286
# 0011859695
# 01191071_1072
# 91391310
# 0011899960
# 01197604
# 91581919
# 0011906503
# 01408099_01408101
# 91856041_6049
# 0011973451
# 11508234
# 91903177
# 0011974919
# 11875011
# 91914407
# 0011976929
# 12052385
# 91939637
# 0012178355
# 12603270
# 91974562
# 0012199830
# 12825369
# 92039708_9710
# 0012529284
# 13149651
# 92081358_1359
# 0012529295
# 71108371
# 92091873
# 0012602424
# 71190280
# 92094746
# 0012947358
# 71202511
# 92094751
# 0013255595
# 71206427
# 92298125
# 0030031163
# 71341634
# 92314414
# 0030041455
# 71366499
# 92327794
# 0060000813
# 71563825
# 92433599_92433601
# 0060007216
# 71601299
# 92586242
# 0060024314
# 80310840a
# 92657311_7313
# 0060025670
# 80707440_7443
# 92657391
# 0060029036
# 80718412_8413
# 93213298
# 0060036622
# 80728670
# 93329540
# 0060068489
# 81186212
# 93351929_93351931
# 0060077689
# 81310636
# 93380187
# 0060080406
# 81574683
# 93455715
# 0060091229
# 81619486_9488
