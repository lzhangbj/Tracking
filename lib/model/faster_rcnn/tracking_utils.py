import torch
import numpy as np


EPISILON = 1e-6


# we can track dynamic infomation
# naively, we can also directly insert all information into LSTM
def TrackingDynamicInfos(prevROIFeature, prevROI, currROIFeature, currROI, kernel=5):
	'''
	calculate the dynamic movement info and feed into our TrackingLocGRU
	input length are all tracking module capacity
	input features should be the same shape for convenience

	we use the matchTrans principle here.
	
	inputs
	@prevROIFeature:    info    : tracking objects' features in previous frame
						type    : torch float tensor
						shape   : (numObjects, C, H, W) h=w=32
	@prevROI:           info    : previous frames tracking objects' rois
						type    : torch tensor int
						shape   : (numObjects, 4)  which dim 2 contains (x1, y1, x2, y2)    
	@currROIFeature:    info    : tracking objects' features in current frame
						type    : torch float tensor
						shape   : (numObjects, C, H, W)
	@currROI:           info    : current frames tracking objects' rois
						type    : torch tensor int
						shape   : (numObjects, 4)  which dim 2 contains (x1, y1, x2, y2)
	return
	@trackingDynamicInfos:  type    : torch float tensor
							shape   : (numObjects, 3*, H, W), dim 1 contains (deltaX, deltaY) wrt previous frame

	'''
	numObjects, C, H, W  = prevROIFeature.size()
	assert prevROIFeature.size() == currROIFeature.size(), [prevROIFeature.size(), currROIFeature.size()]
	# assert H == 16 and W == 32, W
	assert len(prevROI.size()) == 2 and prevROI.size(1) == 4, prevROI.size()
	assert len(currROI.size()) == 2 and currROI.size(1) == 4, currROI.size()
	trackingDynamicInfos = prevROIFeature.new(numObjects, 3*kernel*kernel, H, W).zero_()

	trackingLocInfo = prevROIFeature.new(numObjects, 2, 2, H, W).zero_()

	for i in torch.arange(numObjects):
		# if tracking object exist in last frame
		# we calculate info 
		prevROIXLoc   = torch.arange(W).float()
		prevROIXLoc = prevROIXLoc*(prevROI[i, 2] - prevROI[i, 0])/(W-1) + prevROI[i, 0]
		assert prevROIXLoc.size(0) == W, prevROIXLoc.size(0)
		prevROIXLoc = prevROIXLoc.expand(H, -1)

		currROIXLoc   = torch.arange(W).float()
		currROIXLoc = currROIXLoc*(currROI[i, 2] - currROI[i, 0])/(W-1) + currROI[i, 0]
		assert currROIXLoc.size(0) == W, currROIXLoc.size(0)
		currROIXLoc = currROIXLoc.expand(H, -1)             

		prevROIYLoc   = torch.arange(H).float()
		prevROIYLoc = prevROIYLoc*(prevROI[i, 3] - prevROI[i, 1])/(H-1) + prevROI[i, 1]
		assert prevROIYLoc.size(0) == H, prevROIYLoc.size(0)
		prevROIYLoc = prevROIYLoc.expand(W, -1)
		prevROIYLoc = prevROIYLoc.transpose(1, 0).contiguous()

		currROIYLoc   = torch.arange(H).float()
		currROIYLoc = currROIYLoc*(currROI[i, 3] - currROI[i, 1])/(H-1) + currROI[i, 1]
		assert currROIYLoc.size(0) == H, currROIYLoc.size(0)
		currROIYLoc = currROIYLoc.expand(W, -1)
		currROIYLoc = currROIYLoc.transpose(1, 0).contiguous()


		trackingLocInfo[i, 0, 0] = prevROIXLoc
		trackingLocInfo[i, 0, 1] = prevROIYLoc                
		trackingLocInfo[i, 1, 0] = currROIXLoc
		trackingLocInfo[i, 1, 1] = currROIYLoc
	k_min = int(-(kernel-1)/2)
	k_max = int((kernel+1)/2)
	for i in torch.arange(k_min, k_max):
		for j in torch.arange(k_min, k_max):
			compare_prev_features = prevROIFeature.new(prevROIFeature.size()).zero_()
			compare_prev_loc = trackingLocInfo.new(numObjects, 2, H, W).zero_()
			compare_prev_features[:, :, max(0, -i):min(H-i, H), max(0,-j):min(W-j, W)] = \
						prevROIFeature[:, :, max(0,i):min(H+i, H), max(0,j):min(W+j,W)]
			# assert compare_prev_loc[:, 0].size() == trackingLocInfo[:, 0, 0].size() and trackingLocInfo[:, 0, 0].size() == prevROI[:, 2].size(),\
			# 	[compare_prev_loc.size(), trackingLocInfo.size(), prevROI.size()]
			compare_prev_loc[:, 0] = trackingLocInfo[:, 0, 0] +(i.float()*(prevROI[:, 2] - prevROI[:, 0])/(W-1)).view(-1, 1, 1)
			compare_prev_loc[:, 1] = trackingLocInfo[:, 0, 1] +(j.float()*(prevROI[:, 3] - prevROI[:, 1])/(H-1)).view(-1, 1, 1)

			# print([ (3*((i-k_min)*kernel + (j-k_min))).item(), (3*((i-k_min)*kernel + (j-k_min))+2).item()])
			# print(trackingDynamicInfos[:, 3*((i-k_min)*kernel + (j-k_min)):3*((i-k_min)*kernel + (j-k_min))+2].size())
			trackingDynamicInfos[:, 3*((i-k_min)*kernel + (j-k_min)):3*((i-k_min)*kernel + (j-k_min))+2] = \
				trackingLocInfo[:, 1]-compare_prev_loc
			temp = compare_prev_features*currROIFeature
			trackingDynamicInfos[:, 3*((i-k_min)*kernel + (j-k_min))+2] = torch.sum(temp, dim=1)
			del compare_prev_features
			del compare_prev_loc
			del temp
			# torch.cuda.empty_cache()

	return trackingDynamicInfos


def clip_tracking_boxes(boxes, im_info):
	'''
	im_info : [h,w]
	'''
	boxes[:,0::4].clamp_(0, im_info[1]-1)
	boxes[:,1::4].clamp_(0, im_info[0]-1)
	boxes[:,2::4].clamp_(0, im_info[1]-1)
	boxes[:,3::4].clamp_(0, im_info[0]-1)
	return boxes

def tracking_boxes_validation_check(boxes):
	count=0
	valid_indexes =-boxes.new(boxes.size(0)).fill_(1).long()
	for i in torch.arange(boxes.size(0)):
		if boxes[i, 2]<=boxes[i, 0] or boxes[i, 3]<=boxes[i, 1]:
			boxes[i] = 0
		else:
			valid_indexes[count] = i
			count+=1
	valid_indexes = valid_indexes[:count]
	return boxes, valid_indexes
