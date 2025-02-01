# handydandy
Emily Colleran's course project for Computer Vision (CSE 40535)

Hand Gesture Recognition Program

## Hand Gesture Recognition: High-Level Solution

#### Problem Definition

The goal of this project is to develop a real-time hand gesture recognition system that can classify at least three predefined gestures using video input from a laptop camera. The system must segment the hand from the background, extract meaningful features, and classify the gesture in real-time (≥1 FPS). The solution will use segmentation, feature extraction, and classification techniques to achieve reliable performance.

### Approach

First, the system will need to isolate the hand from the rest of a uniform backgroud to  simplify feature extraction and classification. This will be done using an HSV colro space for skin tone mapping. The system will create a binary mask of the hand to highlight it. The system will also detect contors in the hand to find the largest connected comoponents and refine the segmentation. There might also be some clean up steps here to elimate noise from the images. 

To distinguish between different gestures, contour analysis, convex hull, and Hu moments can be used. Fingertip position will be a very important aspect of this extraction to differentiate one gesture from another. Aspect ratio, centroid position, and palm size will also help enhance identification of gestures.

This proejct will likely use machine leaning to classifly hand gestrues from the given dataset. Supprt Vector Mahcneis seem like a good option in this case because they work well on high-dimensional spaces. This model will be trained with supervised learning where the dataset helps map input features to each gesture. 

To reduce computational laod, the system might only process video frames periodically since gastures will be mostly still. 

As stated in the project description the dataset will include at least 20 training samples per gesture for training.  The two suggested data soruces along with the CVOnline databases all have promising content. 

### Next Steps:
- Conduct preliminary tests on segmentation techniques.
- Review existing datasets and determine feasibility of self-collection.
- Experiment with feature extraction methods and evaluate their effectiveness.
- Begin implementing classification models and assess their real-time performance.
