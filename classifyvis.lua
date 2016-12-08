--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require 'transforms'
local imagenetLabel = require './imagenet'




-- Load the model
local model = torch.load('resnet-101.t7'):cuda()
local softMaxLayer = cudnn.SoftMax():cuda()

-- add Softmax layer
model:add(softMaxLayer)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local N = 5

--for i=2,#arg do
   -- load the image as a RGB float tensor with values 0..1
   local img = image.load('img.jpg', 3, 'float')
   --local name = arg[i]:match( "([^/]+)$" )

   -- Scale, normalize, and crop the image
--itorch.image(img)
   img = transform(img)
itorch.image(img)
    print(img:size())

   -- View as mini-batch of size 1
   local batch = img:view(1, table.unpack(img:size():totable()))
    --print(model.modules[3].weight:size())
   -- Get the output of the softmax
   local output = model:forward(batch:cuda()):squeeze()
    layers=model:size()
    for j=1,layers do
        img=model:get(j).output
        if(img:nDimension()>2) then
            if(img:size(3)>1) then
                itorch.image(img[1])
            end
        end
    end
        

   -- Get the top 5 class indexes and probabilities
   local probs, indexes = output:topk(N, true, true)
   --print('Classes for', arg[i])
   for n=1,N do
     print(probs[n], imagenetLabel[indexes[n]])
   end
   print('')

--end
