# Visualize
This script is based off the classification script present in https://github.com/facebook/fb.resnet.torch . This file also uses the  transforms.lua script provided in the above directory.

Copy the image to be passed to the network to the sa:me directory as that of classifyvis.lua and name it as img.jpg.

Also copy the transforms.lua function to the same directory.

Also, copy the model for example this is tested on 'resnet-101.t7' to the same folder.

Requirements:-
itorch has to be installed because we display the output using itorch.
Installation instructions can be found at https://github.com/facebook/iTorch 

Once it is installed, start itorch from the command line and enter the following command
dofile('classifyvis.lua')
