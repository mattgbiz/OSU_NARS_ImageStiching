# OSU_NARS_ImageStitching
Image stitching code to combine radiographs with mutual information. Designed for radiographs with very limited features in the overlapping region. This code was used on fast and thermal neutron radiographs acquired at Ohio State University Research Reactor.

ReadMe for Ohio State University Nuclear Analysis and Radiation Sensor (NARS) Code for the Image Stitching with Limited Features 
*******************************************************************************************************
Original Author: Matthew Bisbee
Affiliations: Ohio State University Dept. of Mechanical and Aerospace Engineering, Nuclear Engineering
	      Nuclear Analysis and Radiation Sensor (NARS) Laboratory
	      DOE NEUP Fellowship FY19
	      Points of Contact: Advisor Dr. Raymond Cao - cao.152@osu.edu
			         Author Matthew Bisbee - bisbee.11@osu.edu
********************************************************************************************************

PythonScript: RadiographStitchingv3.py
InputFile: StitchInputEDM1_June.txt

********************************************************************************************************
General Information:

This code is written in Python3.X based language and was run on a Windows device. Small changes may need to be made for Mac/Linux machines in terms of navigating to files. Running the code requires an input file (see example input file for structure) and stacks of images to be stitched for left and right hand side. Future implementations of the code can involve stitching objects top and bottom side or extending the stitching to work for images instead of just two. 

The code requires user input of several regions of interest (ROI) through several pop-ups. A description of what region the code needs for each pop-up. Once the ROI has been selected for each feature of hte first image on either side to be stitched, the same ROI will be used for the other images. This code requires there to be at least some subset of images (suggested is at least 8 images) with overlapping features that can be used for minimizing the sum of squares error between left and right portions. THe user will specify what projection angles include the mutual information to be compared. The assumption is that the detector positioning will not change between object translations so once the mutual information is determine for the subset of images, it can be extended to the full stack. If the stitching fails because the measured size of the object and calculated stitched size do not match within 2% then the user must try again possibly expanding or shrinking the size of the ROI with mutual information. 

Note: one issue that had been previously seen in OSU imaging setup is if the mutual information is too close to the edge of the circular beam, the ESF on the edge may not overlap well and lead to failure in stitched size to match measured size. Changing the overlapping ROI to stay away from the circular beam edge is suggested.

********************************************************************************************************
Further Information:
     
How the code actually works can be seen if you take a look at Chapter 8 of the dissertation "Advancing Radiographic Acquisition and Post-Processing Capabilities for a University Research Reactor Fast and Thermal Neutron Radiography and Tomography Instrument" as that goes into more detail on logic and decisions. Also there is a paper that can be cited for this work titled "Improved image stitching method for neutron imaging of large objects with small beam size," doi: 10.1117/12.2677150. Otherwise, the code is well commented, good luck!
