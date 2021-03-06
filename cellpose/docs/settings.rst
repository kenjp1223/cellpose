Settings
--------------------------

The important settings are described on this page. 
See the :ref:`cpclass` for all run options.

Here is an example of calling the Cellpose class and
running a list of images for reference:
::
    from cellpose import models
    import skimage.io

    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=False, model_type='cyto')

    files = ['img0.tif', 'img1.tif']
    imgs = [skimage.io.imread(f) for f in files]
    masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0], 
                                             threshold=0.4, do_3D=False)

You can make lists of channels/diameter for each image, or set the same channels/diameter for all images
as shown in the example above.

Diameter 
~~~~~~~~~~~~~~~~~~~~~~~~

The cellpose models have been trained on images which were rescaled 
to all have the same diameter (30 pixels in the case of the `cyto` 
model and 17 pixels in the case of the `nuclei` model). Therefore, 
cellpose needs a user-defined cell diameter (in pixels) as input, or to estimate 
the object size of an image-by-image basis.

The automated estimation of the diameter is a two-step process using the `style` vector 
from the network, a 64-dimensional summary of the input image. We trained a 
linear regression model to predict the size of objects from these style vectors 
on the training data. On a new image the procedure is as follows.

1. Run the image through the cellpose network and obtain the style vector. Predict the size using the linear regression model from the style vector.
2. Resize the image based on the predicted size and run cellpose again, and produce masks. Take the final estimated size as the median diameter of the predicted masks.

For automated estimation set ``diameter = None``. 
However, if this estimate is incorrect please set the diameter by hand.

Changing the diameter will change the results that the algorithm 
outputs. When the diameter is set smaller than the true size 
then cellpose may over-split cells. Similarly, if the diameter 
is set too big then cellpose may over-merge cells.

Channels
~~~~~~~~~~~~~~~~~~~~~~~~

Cytoplasm model (`'cyto'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cytoplasm model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
an optional nuclear channel. Here are the options for each:
1. 0=grayscale, 1=red, 2=green, 3=blue 
2. 0=None (will set to zero), 1=red, 2=green, 3=blue

Set channels to a list with each of these elements, e.g.
``channels = [0,0]`` if you want to segment cells in grayscale or for single channel images, or
``channels = [2,3]`` if you green cells with blue nuclei.

Nucleus model (`'nuclei'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nuclear model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
always set to an array of zeros. Therefore set the first channel as 
0=grayscale, 1=red, 2=green, 3=blue; and set the second channel to zero, e.g.
``channels = [0,0]`` if you want to segment nuclei in grayscale or for single channel images, or 
``channels = [3,0]`` if you want to segment blue nuclei.

Threshold
~~~~~~~~~~~~~~~~~~~~~~

Note there is nothing keeping the neural network from predicting 
horizontal and vertical flows that do not correspond to any real 
shapes at all. In practice, most predicted flows are consistent with 
real shapes, because the network was only trained on image flows 
that are consistent with real shapes, but sometimes when the network 
is uncertain it may output inconsistent flows. To check that the 
recovered shapes after the flow dynamics step are consistent with 
real masks, we recompute the flow gradients for these putative 
predicted masks, and compute the mean squared error between them and
the flows predicted by the network. 

The ``threshold`` parameter is the maximum allowed error of the flows 
for each mask. The default is ``threshold=0.4``. Increase this threshold 
if cellpose is not returning as many masks as you'd expect. 
Similarly, decrease this threshold if cellpose is returning too many 
ill-shaped masks.

